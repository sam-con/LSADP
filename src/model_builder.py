"""Canonical model building and validation."""

from __future__ import annotations

from datetime import date, datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.adp import ADPDataProvider, BeatADPProvider, FantasyCalcADPProvider, load_saved_canonical_adp_paths, match_players_by_identity
from src.analysis import build_environment_from_player_weeks, default_modeling_config, load_league_environment
from src.baseline_artifacts import CanonicalArtifactManager
from src.calibration import calibrate_market_values
from src.canonical import (
    canonical_configuration,
    canonical_environment_key_for_league,
    classify_transformation_type,
    detect_qb_format,
    detect_reception_value,
    detect_reception_format,
    directed_transform_pairs,
    ordered_canonical_environment_keys,
    validate_canonical_environment_keys,
    validate_canonical_configuration,
)
from src.config import APP_VERSION, CANONICAL_ENVIRONMENTS, CANONICAL_LABELS, CANONICAL_LEAGUES, HISTORICAL_DONOR_FILE
from src.donors import (
    load_history_seed_leagues,
)
from src.history_library import build_league_environment_from_library, build_position_history_library
from src.models import ConfigError, HistoricalDataError, HistoricalLeagueSummary, LeagueSettings, ModelingConfig
from src.replacement import calculate_starter_demand_replacement
from src.sleeper import SleeperClient
from src.transform import apply_league_transformation
from src.validation import positional_error_breakdown, score_prediction
from src.vorp import build_vorp_table
from src.utils import adp_utility, inverse_adp_utility, rank_players_within_position, required_completed_seasons

UTC = timezone.utc


def validate_environment_identity(environment_key: str, league) -> None:
    """Validate that a configured canonical league matches its intended format."""

    detected = canonical_environment_key_for_league(league)
    if detected != environment_key:
        raise ConfigError(
            f"Configured canonical league `{environment_key}` loaded as `{detected}` instead of the intended format."
        )


def canonical_team_counts(
    environment_bundle: dict[str, dict[str, Any]],
    environment_keys: tuple[str, ...] | list[str] | None = None,
) -> dict[str, int]:
    """Return canonical team counts keyed by environment."""

    environment_keys = ordered_canonical_environment_keys(environment_keys or environment_bundle.keys())
    return {
        environment_key: int(environment_bundle[environment_key]["league"].total_rosters)
        for environment_key in environment_keys
    }


def validate_canonical_team_counts(
    environment_bundle: dict[str, dict[str, Any]],
    environment_keys: tuple[str, ...] | list[str] | None = None,
) -> tuple[int, dict[str, int]]:
    """Ensure the canonical environments share one calibration team count."""

    counts = canonical_team_counts(environment_bundle, environment_keys)
    unique_counts = sorted(set(counts.values()))
    if len(unique_counts) != 1:
        mismatch = ", ".join(f"{environment_key}={count}" for environment_key, count in counts.items())
        raise ConfigError(
            "Active canonical Sleeper environments do not share one team count. "
            f"Found: {mismatch}"
        )
    return unique_counts[0], counts


def summarize_canonical_market_distinctness(source_adp_by_environment: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Measure whether the canonical ADP feeds are materially distinct."""

    environment_keys = ordered_canonical_environment_keys(source_adp_by_environment)
    rows: list[dict[str, Any]] = []
    for left_key, right_key in combinations(environment_keys, 2):
        left = source_adp_by_environment[left_key][["player_id", "normalized_name", "position", "player_name", "adp"]].copy()
        right = source_adp_by_environment[right_key][["player_id", "normalized_name", "position", "player_name", "adp"]].rename(
            columns={"adp": "comparison_adp"}
        )
        merged = match_players_by_identity(left, right)
        matched = merged[merged["comparison_adp"].notna()].copy()
        if matched.empty:
            rows.append(
                {
                    "left_environment": left_key,
                    "right_environment": right_key,
                    "matched_players": 0,
                    "identical_share": 1.0,
                    "mean_absolute_diff": 0.0,
                    "max_absolute_diff": 0.0,
                    "status": "No overlap",
                }
            )
            continue
        matched["absolute_diff"] = (matched["adp"] - matched["comparison_adp"]).abs()
        identical_share = float((matched["absolute_diff"] < 1e-9).mean())
        mean_absolute_diff = float(matched["absolute_diff"].mean())
        rows.append(
            {
                "left_environment": left_key,
                "right_environment": right_key,
                "matched_players": int(len(matched)),
                "identical_share": identical_share,
                "mean_absolute_diff": mean_absolute_diff,
                "max_absolute_diff": float(matched["absolute_diff"].max()),
                "status": "Distinct" if identical_share < 0.95 and mean_absolute_diff > 0.05 else "Suspiciously similar",
            }
        )
    return pd.DataFrame(rows).sort_values(["status", "left_environment", "right_environment"]).reset_index(drop=True)


def validate_canonical_market_distinctness(source_adp_by_environment: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Block calibration if the canonical market inputs appear effectively identical."""

    diagnostics = summarize_canonical_market_distinctness(source_adp_by_environment)
    suspicious = diagnostics[diagnostics["status"] == "Suspiciously similar"]
    if not suspicious.empty:
        offenders = ", ".join(
            f"{row.left_environment} vs {row.right_environment}"
            for row in suspicious.itertuples(index=False)
        )
        raise ConfigError(
            "Canonical ADP markets appear effectively identical. "
            f"Check the saved BeatADP datasets before calibrating: {offenders}"
        )
    return diagnostics


def load_source_adp_by_environment(
    *,
    canonical_adp_paths: dict[str, Path] | None = None,
    adp_provider: Any | None = None,
    force_adp_refresh: bool = False,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Load canonical ADP inputs either from CSV fixtures or a provider bundle."""

    if canonical_adp_paths is not None:
        frames: dict[str, pd.DataFrame] = {}
        metadata: dict[str, dict[str, Any]] = {}
        environment_keys = ordered_canonical_environment_keys(canonical_adp_paths)
        for environment_key in environment_keys:
            frame, entry = ADPDataProvider(canonical_adp_paths[environment_key]).load_with_metadata()
            frame["canonical_format"] = environment_key
            frames[environment_key] = frame
            metadata[environment_key] = entry
        frames, metadata = synthesize_sf_ppr_from_square(frames, metadata)
        return frames, {
            "source": "csv",
            "status": "Saved",
            "last_refresh": None,
            "available_environments": list(ordered_canonical_environment_keys(frames)),
            "missing_environments": [
                environment_key for environment_key in CANONICAL_ENVIRONMENTS if environment_key not in frames
            ],
            "formats": metadata,
        }

    provider = adp_provider or BeatADPProvider()
    try:
        bundle = provider.load_canonical_markets(force_refresh=force_adp_refresh)
    except TypeError:
        raise ConfigError(
            "An ADP provider without CSV paths must implement load_canonical_markets(force_refresh=...)."
        ) from None
    frames, metadata = synthesize_sf_ppr_from_square(bundle["frames"], bundle.get("formats", {}))
    bundle["frames"] = frames
    bundle["formats"] = metadata
    bundle["available_environments"] = list(ordered_canonical_environment_keys(frames))
    bundle["missing_environments"] = [environment_key for environment_key in CANONICAL_ENVIRONMENTS if environment_key not in frames]
    return bundle["frames"], bundle


def _frame_with_identity_key(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    working["player_id"] = working["player_id"].astype("string")
    working["normalized_name"] = working["normalized_name"].astype("string")
    working["identity_key"] = working["player_id"].where(
        working["player_id"].notna(),
        working["normalized_name"] + "|" + working["position"].astype(str),
    )
    return working.drop_duplicates("identity_key", keep="first").reset_index(drop=True)


def _estimate_market_via_square_path(
    *,
    base_frame: pd.DataFrame,
    delta_from_frame: pd.DataFrame,
    delta_to_frame: pd.DataFrame,
    path_name: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    base = _frame_with_identity_key(base_frame)
    delta_from = _frame_with_identity_key(delta_from_frame)[["identity_key", "position", "adp"]].rename(
        columns={"adp": "adp_delta_from"}
    )
    delta_to = _frame_with_identity_key(delta_to_frame)[["identity_key", "adp"]].rename(columns={"adp": "adp_delta_to"})

    delta = delta_from.merge(delta_to, on="identity_key", how="inner")
    delta["delta_utility"] = adp_utility(delta["adp_delta_to"]) - adp_utility(delta["adp_delta_from"])

    position_delta = delta.groupby("position")["delta_utility"].median().to_dict()
    overall_delta = float(delta["delta_utility"].median()) if not delta.empty else 0.0

    estimated = base.merge(delta[["identity_key", "delta_utility"]], on="identity_key", how="left")
    estimated["delta_fill_kind"] = "direct"
    estimated["delta_utility"] = estimated["delta_utility"].fillna(
        estimated["position"].map(position_delta)
    )
    estimated.loc[estimated["delta_utility"].isna(), "delta_fill_kind"] = "overall_position_fallback"
    estimated["delta_utility"] = estimated["delta_utility"].fillna(overall_delta)
    estimated["estimated_utility"] = adp_utility(estimated["adp"]) + estimated["delta_utility"]
    estimated["estimated_adp"] = inverse_adp_utility(estimated["estimated_utility"]).astype(float)
    estimated["square_path"] = path_name
    diagnostics = {
        "path_name": path_name,
        "base_rows": int(len(base)),
        "direct_delta_rows": int((estimated["delta_fill_kind"] == "direct").sum()),
        "fallback_delta_rows": int((estimated["delta_fill_kind"] != "direct").sum()),
        "overall_delta_utility": overall_delta,
        "position_delta_utility": {key: float(value) for key, value in position_delta.items()},
    }
    return estimated, diagnostics


def synthesize_sf_ppr_from_square(
    frames: dict[str, pd.DataFrame],
    metadata: dict[str, dict[str, Any]],
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, Any]]]:
    """Estimate the missing SF PPR market from the other three canonical corners."""

    required = ("1qb_half_ppr", "1qb_ppr", "sf_half_ppr")
    if "sf_ppr" in frames or any(environment_key not in frames for environment_key in required):
        return frames, metadata

    scoring_path, scoring_diagnostics = _estimate_market_via_square_path(
        base_frame=frames["sf_half_ppr"],
        delta_from_frame=frames["1qb_half_ppr"],
        delta_to_frame=frames["1qb_ppr"],
        path_name="sf_half_ppr_plus_scoring_delta",
    )
    scarcity_path, scarcity_diagnostics = _estimate_market_via_square_path(
        base_frame=frames["1qb_ppr"],
        delta_from_frame=frames["1qb_half_ppr"],
        delta_to_frame=frames["sf_half_ppr"],
        path_name="1qb_ppr_plus_superflex_delta",
    )

    left = scoring_path.rename(
        columns={
            "player_id": "player_id_left",
            "sleeper_id": "sleeper_id_left",
            "player_name": "player_name_left",
            "position": "position_left",
            "team": "team_left",
            "normalized_name": "normalized_name_left",
            "estimated_utility": "estimated_utility_left",
            "estimated_adp": "estimated_adp_left",
        }
    )
    right = scarcity_path.rename(
        columns={
            "player_id": "player_id_right",
            "sleeper_id": "sleeper_id_right",
            "player_name": "player_name_right",
            "position": "position_right",
            "team": "team_right",
            "normalized_name": "normalized_name_right",
            "estimated_utility": "estimated_utility_right",
            "estimated_adp": "estimated_adp_right",
        }
    )
    merged = left.merge(
        right[
            [
                "identity_key",
                "player_id_right",
                "sleeper_id_right",
                "player_name_right",
                "position_right",
                "team_right",
                "normalized_name_right",
                "estimated_utility_right",
                "estimated_adp_right",
            ]
        ],
        on="identity_key",
        how="outer",
    )

    final = pd.DataFrame(
        {
            "player_id": merged["player_id_left"].fillna(merged["player_id_right"]).astype("string"),
            "sleeper_id": merged["sleeper_id_left"].fillna(merged["sleeper_id_right"]).astype("string"),
            "player_name": merged["player_name_left"].fillna(merged["player_name_right"]),
            "position": merged["position_left"].fillna(merged["position_right"]),
            "team": merged["team_left"].fillna(merged["team_right"]).fillna(""),
            "normalized_name": merged["normalized_name_left"].fillna(merged["normalized_name_right"]),
        }
    )
    utility_columns = merged[["estimated_utility_left", "estimated_utility_right"]]
    final["synthetic_path_count"] = utility_columns.notna().sum(axis=1)
    final["synthetic_utility"] = utility_columns.mean(axis=1, skipna=True)
    final["adp"] = inverse_adp_utility(final["synthetic_utility"]).astype(float)
    final["canonical_format"] = "sf_ppr"
    final["source"] = "Synthetic Canonical SF PPR"
    final["adp_source_field"] = "synthetic_complete_square"
    final["retrieved_at"] = datetime.now(UTC).isoformat()
    final["synthetic_method"] = "complete_square"
    final["synthetic_source_environments"] = "1qb_half_ppr,1qb_ppr,sf_half_ppr"
    final = final.dropna(subset=["player_name", "position", "adp"]).reset_index(drop=True)
    final = rank_players_within_position(final)
    frames = frames.copy()
    metadata = metadata.copy()
    frames["sf_ppr"] = final
    metadata["sf_ppr"] = {
        "source": "synthetic_complete_square",
        "status": "Estimated",
        "synthetic": True,
        "player_count": int(len(final)),
        "path": "synthetic://sf_ppr_complete_square",
        "source_environments": list(required),
        "generated_at": datetime.now(UTC).isoformat(),
        "blended_rows": int((final["synthetic_path_count"] == 2).sum()),
        "single_path_rows": int((final["synthetic_path_count"] == 1).sum()),
        "scoring_path": scoring_diagnostics,
        "scarcity_path": scarcity_diagnostics,
    }
    return frames, metadata


def build_canonical_environment_bundle(
    client: SleeperClient,
    canonical_leagues: dict[str, str] | None = None,
    environment_keys: tuple[str, ...] | list[str] | None = None,
    modeling_config: ModelingConfig | None = None,
    today: date | None = None,
) -> dict[str, dict[str, Any]]:
    """Build historical environments for all canonical leagues."""

    canonical_leagues = canonical_leagues or CANONICAL_LEAGUES
    environment_keys = ordered_canonical_environment_keys(environment_keys or CANONICAL_ENVIRONMENTS)
    validate_canonical_configuration(canonical_leagues, required_environment_keys=environment_keys)
    modeling_config = modeling_config or default_modeling_config()

    bundle: dict[str, dict[str, Any]] = {}
    for environment_key in environment_keys:
        environment = load_league_environment(
            client=client,
            league_id=canonical_leagues[environment_key],
            modeling_config=modeling_config,
            today=today,
            replacement_method="starter_demand",
        )
        validate_environment_identity(environment_key, environment["league"])
        bundle[environment_key] = environment
    return bundle


def load_canonical_league_settings_bundle(
    client: SleeperClient,
    canonical_leagues: dict[str, str] | None = None,
    environment_keys: tuple[str, ...] | list[str] | None = None,
) -> dict[str, Any]:
    """Load only the canonical league settings without requiring historical chains."""

    canonical_leagues = canonical_leagues or CANONICAL_LEAGUES
    environment_keys = ordered_canonical_environment_keys(environment_keys or CANONICAL_ENVIRONMENTS)
    validate_canonical_configuration(canonical_leagues, required_environment_keys=environment_keys)
    bundle: dict[str, Any] = {}
    for environment_key in environment_keys:
        league = client.get_league(canonical_leagues[environment_key])
        validate_environment_identity(environment_key, league)
        bundle[environment_key] = {"league": league}
    return bundle


def build_canonical_environment_bundle_from_donors(
    client: SleeperClient,
    canonical_leagues: dict[str, str] | None = None,
    environment_keys: tuple[str, ...] | list[str] | None = None,
    donor_configuration: pd.DataFrame | None = None,
    modeling_config: ModelingConfig | None = None,
    today: date | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Build canonical environments using the position-specific history library."""

    canonical_leagues = canonical_leagues or CANONICAL_LEAGUES
    environment_keys = ordered_canonical_environment_keys(environment_keys or CANONICAL_ENVIRONMENTS)
    modeling_config = modeling_config or default_modeling_config()
    today = today or date.today()
    canonical_settings = load_canonical_league_settings_bundle(client, canonical_leagues, environment_keys=environment_keys)
    seed_leagues, seed_metadata = load_history_seed_leagues(
        today=today,
        donor_configuration=donor_configuration,
    )
    history_library = build_position_history_library(
        client=client,
        seed_leagues=seed_leagues,
        today=today,
        modeling_config=modeling_config,
    )

    bundle: dict[str, dict[str, Any]] = {}
    for environment_key in environment_keys:
        league = canonical_settings[environment_key]["league"]
        environment = build_league_environment_from_library(
            league=league,
            replacement_method="starter_demand",
            library_bundle=history_library,
        )
        validate_environment_identity(environment_key, environment["league"])
        environment["historical_source"] = "position_history_library"
        bundle[environment_key] = environment
    history_library["seed_metadata"] = seed_metadata
    return bundle, history_library


def candidate_model_specs() -> list[dict[str, Any]]:
    """Return the candidate model specifications to compare."""

    specs: list[dict[str, Any]] = [
        {
            "model_name": "No Adjustment",
            "metric_mode": "none",
            "replacement_method": "starter_demand",
            "utility_transform": "neg_log",
            "weight_power": 1.0,
        }
    ]
    for utility_transform in ("neg_log", "inverse_adp"):
        for weight_power in (1.0, 1.5):
            specs.extend(
                [
                    {
                        "model_name": "Curve Only",
                        "metric_mode": "expected_ppg",
                        "replacement_method": "starter_demand",
                        "utility_transform": utility_transform,
                        "weight_power": weight_power,
                    },
                    {
                        "model_name": "Curve + Starter VORP",
                        "metric_mode": "vorp",
                        "replacement_method": "starter_demand",
                        "utility_transform": utility_transform,
                        "weight_power": weight_power,
                    },
                    {
                        "model_name": "Curve + Roster VORP",
                        "metric_mode": "vorp",
                        "replacement_method": "historical_roster",
                        "utility_transform": utility_transform,
                        "weight_power": weight_power,
                    },
                ]
            )
    return specs


def _metric_frame_for_environment(environment: dict[str, Any], metric_mode: str, replacement_method: str) -> pd.DataFrame:
    if metric_mode == "expected_ppg":
        return environment["vorp_variants"]["starter_demand"][["position", "rank", "expected_ppg"]].copy()
    if metric_mode == "vorp":
        return environment["vorp_variants"][replacement_method][["position", "rank", "vorp", "expected_ppg"]].copy()
    raise ValueError(f"Unsupported metric mode {metric_mode}")


def _prepare_prediction_frame(
    source_adp: pd.DataFrame,
    source_environment: dict[str, Any],
    target_environment: dict[str, Any],
    model_spec: dict[str, Any],
    source_label: str,
) -> pd.DataFrame:
    metric_mode = str(model_spec["metric_mode"])
    replacement_method = str(model_spec["replacement_method"])

    if metric_mode == "expected_ppg":
        source_metric = _metric_frame_for_environment(source_environment, metric_mode, replacement_method).rename(
            columns={"rank": "pos_rank", "expected_ppg": "source_metric"}
        )
        target_metric = _metric_frame_for_environment(target_environment, metric_mode, replacement_method).rename(
            columns={"rank": "pos_rank", "expected_ppg": "target_metric"}
        )
        calibrated_source = source_adp.merge(source_metric, on=["position", "pos_rank"], how="left")
        _, calibrated_source = calibrate_market_values(
            calibrated_source,
            metric_column="source_metric",
            metric_name="expected_ppg",
            utility_transform=str(model_spec["utility_transform"]),
            weight_power=float(model_spec["weight_power"]),
        )
    else:
        source_metric = _metric_frame_for_environment(source_environment, metric_mode, replacement_method).rename(
            columns={"rank": "pos_rank", "vorp": "source_metric", "expected_ppg": "canonical_expected_ppg"}
        )
        target_metric = _metric_frame_for_environment(target_environment, metric_mode, replacement_method).rename(
            columns={"rank": "pos_rank", "vorp": "target_metric", "expected_ppg": "league_expected_ppg"}
        )
        calibrated_source = source_adp.merge(source_metric, on=["position", "pos_rank"], how="left")
        _, calibrated_source = calibrate_market_values(
            calibrated_source,
            metric_column="source_metric",
            metric_name="vorp",
            utility_transform=str(model_spec["utility_transform"]),
            weight_power=float(model_spec["weight_power"]),
        )

    prediction_frame = calibrated_source.merge(target_metric, on=["position", "pos_rank"], how="left")
    prediction_frame["canonical_environment_label"] = source_label
    prediction_frame["canonical_metric"] = prediction_frame["source_metric"]
    return prediction_frame


def predict_between_canonical_environments(
    source_key: str,
    target_key: str,
    source_environment: dict[str, Any],
    target_environment: dict[str, Any],
    source_adp: pd.DataFrame,
    model_spec: dict[str, Any],
) -> pd.DataFrame:
    """Predict one canonical ADP market from another."""

    if model_spec["metric_mode"] == "none":
        prediction = source_adp.copy()
        prediction["league_adjusted_adp"] = prediction["adp"].astype(float)
        prediction["canonical_environment_label"] = CANONICAL_LABELS[source_key]
        prediction["delta_metric"] = 0.0
        prediction["canonical_metric"] = 0.0
        prediction["league_metric"] = 0.0
        prediction["league_expected_ppg"] = np.nan
        prediction["canonical_expected_ppg"] = np.nan
        prediction["canonical_vorp"] = np.nan
        prediction["league_vorp"] = np.nan
        prediction["short_explanation"] = (
            prediction["canonical_environment_label"] + " anchor retained without transformation."
        )
        return prediction

    prepared = _prepare_prediction_frame(
        source_adp=source_adp,
        source_environment=source_environment,
        target_environment=target_environment,
        model_spec=model_spec,
        source_label=CANONICAL_LABELS[source_key],
    )
    explanation_metric_label = "PPG" if model_spec["metric_mode"] == "expected_ppg" else "VORP"
    transformed = apply_league_transformation(
        calibrated_adp=prepared.rename(columns={"source_metric": "baseline_metric", "target_metric": "target_metric_internal"}),
        baseline_metric_column="baseline_metric",
        target_metric_column="target_metric_internal",
        explanation_metric_label=explanation_metric_label,
        utility_transform=str(model_spec["utility_transform"]),
        anchor_label=CANONICAL_LABELS[source_key],
    )
    transformed["canonical_metric"] = transformed["baseline_metric"]
    transformed["league_metric"] = transformed["target_metric_internal"]
    if model_spec["metric_mode"] == "vorp":
        transformed["canonical_vorp"] = transformed["canonical_metric"]
        transformed["league_vorp"] = transformed["league_metric"]
    else:
        transformed["canonical_vorp"] = np.nan
        transformed["league_vorp"] = np.nan
    return transformed


def evaluate_candidate_spec(
    source_adp_by_environment: dict[str, pd.DataFrame],
    environment_bundle: dict[str, dict[str, Any]],
    model_spec: dict[str, Any],
) -> tuple[pd.DataFrame, dict[tuple[str, str], pd.DataFrame]]:
    """Run all directed validations for one candidate spec."""

    environment_keys = ordered_canonical_environment_keys(source_adp_by_environment)
    validation_rows: list[dict[str, Any]] = []
    predictions: dict[tuple[str, str], pd.DataFrame] = {}

    for source_key, target_key in directed_transform_pairs(environment_keys):
        prediction = predict_between_canonical_environments(
            source_key=source_key,
            target_key=target_key,
            source_environment=environment_bundle[source_key],
            target_environment=environment_bundle[target_key],
            source_adp=source_adp_by_environment[source_key],
            model_spec=model_spec,
        )
        predictions[(source_key, target_key)] = prediction
        metrics = score_prediction(
            prediction,
            source_adp_by_environment[target_key],
            model_name=str(model_spec["model_name"]),
        )
        metrics["source_environment"] = source_key
        metrics["target_environment"] = target_key
        metrics["transform_type"] = classify_transformation_type(source_key, target_key)
        metrics["utility_transform"] = model_spec["utility_transform"]
        metrics["weight_power"] = model_spec["weight_power"]
        metrics["replacement_method"] = model_spec["replacement_method"]
        metrics["metric_mode"] = model_spec["metric_mode"]
        metrics["composite_score"] = composite_validation_score(metrics)
        validation_rows.append(metrics)

    return pd.DataFrame(validation_rows), predictions


def composite_validation_score(metric_row: dict[str, Any] | pd.Series) -> float:
    """Compute a global model-selection score; lower is better."""

    spearman = float(metric_row["spearman"]) if pd.notna(metric_row["spearman"]) else 0.0
    top_24 = float(metric_row["top_24_overlap"])
    top_50 = float(metric_row["top_50_overlap"])
    top_100 = float(metric_row["top_100_overlap"])
    pairwise = float(metric_row["pairwise_accuracy"])
    weighted_mae = float(metric_row["weighted_mae"])
    return (
        weighted_mae
        + 18.0 * (1.0 - spearman)
        + 12.0 * (1.0 - top_24)
        + 8.0 * (1.0 - top_50)
        + 5.0 * (1.0 - top_100)
        + 10.0 * (1.0 - pairwise)
    )


def summarize_validation_by_group(validation_frame: pd.DataFrame, group_column: str) -> pd.DataFrame:
    """Aggregate validation metrics by category or model grouping."""

    summary = (
        validation_frame.groupby(group_column, as_index=False)
        .agg(
            spearman=("spearman", "mean"),
            mae=("mae", "mean"),
            weighted_mae=("weighted_mae", "mean"),
            top_24_overlap=("top_24_overlap", "mean"),
            top_50_overlap=("top_50_overlap", "mean"),
            top_100_overlap=("top_100_overlap", "mean"),
            pairwise_accuracy=("pairwise_accuracy", "mean"),
            composite_score=("composite_score", "mean"),
        )
        .sort_values("composite_score")
        .reset_index(drop=True)
    )
    return summary


def build_leave_one_environment_out_summary(validation_frame: pd.DataFrame) -> pd.DataFrame:
    """Estimate generalization by holding out each target environment during spec selection."""

    if validation_frame.empty:
        return pd.DataFrame()
    environment_keys = ordered_canonical_environment_keys(validation_frame["target_environment"].dropna().unique().tolist())
    rows: list[dict[str, Any]] = []
    for held_out_key in environment_keys:
        training = validation_frame[validation_frame["target_environment"] != held_out_key]
        if training.empty:
            continue
        grouped = summarize_validation_by_group(training, "spec_id")
        best_spec_id = grouped.iloc[0]["spec_id"]
        holdout = validation_frame[
            (validation_frame["target_environment"] == held_out_key) & (validation_frame["spec_id"] == best_spec_id)
        ]
        rows.append(
            {
                "held_out_environment": held_out_key,
                "selected_spec_id": best_spec_id,
                "weighted_mae": float(holdout["weighted_mae"].mean()),
                "spearman": float(holdout["spearman"].mean()),
                "top_50_overlap": float(holdout["top_50_overlap"].mean()),
                "composite_score": float(holdout["composite_score"].mean()),
            }
        )
    return pd.DataFrame(rows)


def build_candidate_model(
    client: SleeperClient,
    canonical_leagues: dict[str, str] | None = None,
    canonical_adp_paths: dict[str, Path] | None = None,
    adp_provider: Any | None = None,
    donor_configuration: pd.DataFrame | None = None,
    modeling_config: ModelingConfig | None = None,
    today: date | None = None,
    force_adp_refresh: bool = False,
) -> dict[str, Any]:
    """Build, validate, and package a canonical candidate model."""

    canonical_leagues = canonical_leagues or CANONICAL_LEAGUES
    modeling_config = modeling_config or default_modeling_config()
    if canonical_adp_paths is None and adp_provider is None:
        canonical_adp_paths, saved_adp_metadata = load_saved_canonical_adp_paths()
    else:
        saved_adp_metadata = None

    source_adp_by_environment, adp_source_summary = load_source_adp_by_environment(
        canonical_adp_paths=canonical_adp_paths,
        adp_provider=adp_provider,
        force_adp_refresh=force_adp_refresh,
    )
    active_environment_keys = validate_canonical_environment_keys(source_adp_by_environment)
    synthetic_environment_keys = [
        environment_key
        for environment_key, entry in adp_source_summary.get("formats", {}).items()
        if bool(entry.get("synthetic"))
    ]
    validate_canonical_configuration(
        canonical_leagues,
        canonical_adp_paths,
        required_environment_keys=active_environment_keys,
        allow_missing_adp_paths=synthetic_environment_keys,
    )
    if saved_adp_metadata is not None:
        adp_source_summary["last_refresh"] = saved_adp_metadata.get("fetched_at")
        adp_source_summary["source"] = saved_adp_metadata.get("source", adp_source_summary["source"])
        adp_source_summary["status"] = saved_adp_metadata.get("status", adp_source_summary["status"])
        if "formats" in saved_adp_metadata:
            merged_formats = dict(saved_adp_metadata["formats"])
            merged_formats.update(adp_source_summary["formats"])
            adp_source_summary["formats"] = merged_formats
        adp_source_summary["available_environments"] = list(ordered_canonical_environment_keys(active_environment_keys))
        adp_source_summary["missing_environments"] = [
            environment_key for environment_key in CANONICAL_ENVIRONMENTS if environment_key not in active_environment_keys
        ]

    environment_bundle, donor_history_bundle = build_canonical_environment_bundle_from_donors(
        client=client,
        canonical_leagues=canonical_leagues,
        environment_keys=active_environment_keys,
        donor_configuration=donor_configuration,
        modeling_config=modeling_config,
        today=today,
    )
    donor_metadata = {
        "source": str(HISTORICAL_DONOR_FILE) if donor_configuration is None else "provided_dataframe",
        "seed_metadata": donor_history_bundle.get("seed_metadata", {}),
        "seed_leagues": donor_history_bundle["seed_leagues"].to_dict(orient="records"),
        "library_metadata": donor_history_bundle["library_metadata"],
        "position_environment_count": int(len(donor_history_bundle["position_scoring_environments"])),
    }
    canonical_team_count, canonical_team_counts_by_environment = validate_canonical_team_counts(
        environment_bundle,
        active_environment_keys,
    )
    market_distinctness = validate_canonical_market_distinctness(source_adp_by_environment)

    spec_rows: list[dict[str, Any]] = []
    validation_frames: list[pd.DataFrame] = []
    selected_predictions: dict[tuple[str, str], pd.DataFrame] = {}
    best_spec: dict[str, Any] | None = None
    best_score = float("inf")

    for index, model_spec in enumerate(candidate_model_specs(), start=1):
        spec_id = f"spec_{index:02d}"
        validation_frame, predictions = evaluate_candidate_spec(
            source_adp_by_environment=source_adp_by_environment,
            environment_bundle=environment_bundle,
            model_spec=model_spec,
        )
        validation_frame["spec_id"] = spec_id
        validation_frames.append(validation_frame)
        overall = summarize_validation_by_group(validation_frame.assign(group="overall"), "group").iloc[0]
        spec_row = {
            "spec_id": spec_id,
            **model_spec,
            "composite_score": float(overall["composite_score"]),
            "weighted_mae": float(overall["weighted_mae"]),
            "spearman": float(overall["spearman"]),
            "top_24_overlap": float(overall["top_24_overlap"]),
            "top_50_overlap": float(overall["top_50_overlap"]),
            "top_100_overlap": float(overall["top_100_overlap"]),
        }
        spec_rows.append(spec_row)
        if spec_row["composite_score"] < best_score:
            best_score = spec_row["composite_score"]
            best_spec = spec_row
            selected_predictions = predictions

    if best_spec is None:
        raise ConfigError("No candidate model specifications could be evaluated.")

    all_validation = pd.concat(validation_frames, ignore_index=True)
    best_validation = all_validation[all_validation["spec_id"] == best_spec["spec_id"]].copy()

    curves_rows: list[pd.DataFrame] = []
    replacement_rows: list[pd.DataFrame] = []
    calibration_rows: list[pd.DataFrame] = []
    for environment_key in active_environment_keys:
        environment = environment_bundle[environment_key]
        source_adp = source_adp_by_environment[environment_key]

        curves_frame = environment["evaluated_curves"].copy()
        curves_frame["environment_key"] = environment_key
        curves_rows.append(curves_frame)

        for method_name, replacement_frame in environment["replacement_variants"].items():
            frame = replacement_frame.copy()
            frame["environment_key"] = environment_key
            frame["replacement_method"] = method_name
            replacement_rows.append(frame)

        if best_spec["metric_mode"] == "none":
            continue

        metric_frame = _prepare_prediction_frame(
            source_adp=source_adp,
            source_environment=environment,
            target_environment=environment,
            model_spec=best_spec,
            source_label=CANONICAL_LABELS[environment_key],
        )
        calibration = (
            metric_frame[["position", "market_intercept", "market_coefficient", "market_metric_name", "utility_transform", "weight_power"]]
            .drop_duplicates("position")
            .rename(columns={"market_intercept": "intercept", "market_coefficient": "coefficient", "market_metric_name": "metric_name"})
        )
        calibration["environment_key"] = environment_key
        calibration["model_name"] = best_spec["model_name"]
        calibration["replacement_method"] = best_spec["replacement_method"]
        calibration_rows.append(calibration)

    canonical_config_payload = canonical_configuration(
        canonical_leagues,
        adp_paths=canonical_adp_paths,
        adp_source=str(adp_source_summary["source"]),
        adp_details=adp_source_summary["formats"],
        environment_keys=active_environment_keys,
    )
    for environment_key in active_environment_keys:
        canonical_config_payload[environment_key]["scoring_settings"] = environment_bundle[environment_key]["league"].scoring_settings
        canonical_config_payload[environment_key]["roster_positions"] = environment_bundle[environment_key]["league"].roster_positions
        canonical_config_payload[environment_key]["team_count"] = environment_bundle[environment_key]["league"].total_rosters
    metadata = {
        "selected_model_name": best_spec["model_name"],
        "selected_spec_id": best_spec["spec_id"],
        "selected_metric_mode": best_spec["metric_mode"],
        "selected_replacement_method": best_spec["replacement_method"],
        "selected_utility_transform": best_spec["utility_transform"],
        "selected_weight_power": best_spec["weight_power"],
        "selected_model_score": best_spec["composite_score"],
        "generated_timestamp": datetime.now(UTC).isoformat(),
        "model_version": APP_VERSION,
        "available_canonical_environments": list(active_environment_keys),
        "canonical_environments": canonical_config_payload,
        "adp_snapshot": {
            "source": adp_source_summary["source"],
            "status": adp_source_summary["status"],
            "last_refresh": adp_source_summary.get("last_refresh"),
            "canonical_team_count": canonical_team_count,
            "team_counts_by_environment": canonical_team_counts_by_environment,
            "available_environments": list(active_environment_keys),
            "missing_environments": [
                environment_key for environment_key in CANONICAL_ENVIRONMENTS if environment_key not in active_environment_keys
            ],
            "formats": adp_source_summary["formats"],
            "market_distinctness": market_distinctness.to_dict(orient="records"),
        },
        "historical_seasons": {
            environment_key: [league.season for league in environment_bundle[environment_key]["historical_leagues"]]
            for environment_key in active_environment_keys
        },
        "historical_donor_source": donor_metadata,
        "validation_complete": True,
    }

    leave_one_out = build_leave_one_environment_out_summary(all_validation)
    grouped_type = summarize_validation_by_group(best_validation, "transform_type")
    best_validation["source_label"] = best_validation["source_environment"].map(CANONICAL_LABELS)
    best_validation["target_label"] = best_validation["target_environment"].map(CANONICAL_LABELS)

    return {
        "environment_bundle": environment_bundle,
        "donor_history_bundle": donor_history_bundle,
        "source_adp_by_environment": source_adp_by_environment,
        "model_parameters": pd.DataFrame(spec_rows).sort_values("composite_score").reset_index(drop=True),
        "selected_spec": best_spec,
        "validation": all_validation,
        "selected_validation": best_validation,
        "validation_by_type": grouped_type,
        "leave_one_out": leave_one_out,
        "adp_source_summary": adp_source_summary,
        "market_distinctness": market_distinctness,
        "curves": pd.concat(curves_rows, ignore_index=True),
        "replacement": pd.concat(replacement_rows, ignore_index=True),
        "market_calibration": pd.concat(calibration_rows, ignore_index=True) if calibration_rows else pd.DataFrame(),
        "canonical_config": canonical_config_payload,
        "metadata": metadata,
        "predictions": selected_predictions,
        "history_position_environments": donor_history_bundle["position_scoring_environments"],
        "history_environment_seasons": donor_history_bundle["environment_seasons"],
        "history_curve_models": donor_history_bundle["curve_models"],
        "history_curves": donor_history_bundle["fitted_curves"],
        "history_library_metadata": donor_history_bundle["library_metadata"],
    }


def save_candidate_model(candidate_manager: CanonicalArtifactManager, candidate_bundle: dict[str, Any]) -> None:
    """Persist a built candidate model without touching production."""

    candidate_manager.save(
        curves=candidate_bundle["curves"],
        replacement=candidate_bundle["replacement"],
        market_calibration=candidate_bundle["market_calibration"],
        model_parameters=candidate_bundle["model_parameters"],
        validation=candidate_bundle["validation"],
        canonical_config=candidate_bundle["canonical_config"],
        metadata=candidate_bundle["metadata"],
        history_position_environments=candidate_bundle.get("history_position_environments"),
        history_environment_seasons=candidate_bundle.get("history_environment_seasons"),
        history_curve_models=candidate_bundle.get("history_curve_models"),
        history_curves=candidate_bundle.get("history_curves"),
        history_library_metadata=candidate_bundle.get("history_library_metadata"),
    )


def promote_candidate_model(
    candidate_manager: CanonicalArtifactManager,
    production_manager: CanonicalArtifactManager,
) -> None:
    """Promote a validated candidate model to production."""

    candidate_artifacts = candidate_manager.load()
    if not bool(candidate_artifacts.metadata.get("validation_complete")):
        raise ConfigError("Candidate model validation is incomplete. Promotion is blocked.")
    production_manager.promote_from(candidate_manager)


def build_public_target_environment_from_anchor(
    *,
    artifacts,
    source_key: str,
    target_league: LeagueSettings,
) -> dict[str, Any]:
    """Build a best-effort public target environment without requiring league history."""

    fitted_curves, empirical_curves, runtime_mode = build_public_target_curve_bundle(
        artifacts=artifacts,
        source_key=source_key,
        target_league=target_league,
    )
    if fitted_curves.empty:
        raise ConfigError(f"Production curves are missing for canonical anchor {source_key}.")

    fitted_curves = fitted_curves.drop(columns=["environment_key"], errors="ignore").reset_index(drop=True)
    empirical_curves = empirical_curves.drop(columns=["environment_key"], errors="ignore").reset_index(drop=True)
    if empirical_curves.empty:
        empirical_curves = fitted_curves.copy()
        empirical_curves["dataset"] = "empirical"

    evaluated_curves = pd.concat([fitted_curves, empirical_curves], ignore_index=True)
    starter_demand_replacement = calculate_starter_demand_replacement(target_league, fitted_curves)
    fallback_historical_replacement = starter_demand_replacement.copy()
    fallback_historical_replacement["method"] = "Starter Demand Replacement (No History Fallback)"

    replacement_variants = {
        "starter_demand": starter_demand_replacement,
        "historical_roster": fallback_historical_replacement,
    }
    vorp_variants = {
        name: build_vorp_table(fitted_curves, replacement_frame)
        for name, replacement_frame in replacement_variants.items()
    }
    selected_replacement_method = str(artifacts.metadata.get("selected_replacement_method", "starter_demand"))
    replacement = replacement_variants.get(selected_replacement_method, starter_demand_replacement)
    vorp_table = vorp_variants.get(selected_replacement_method, vorp_variants["starter_demand"])

    return {
        "league": target_league,
        "historical_leagues": [],
        "coverage": [],
        "player_weeks": pd.DataFrame(),
        "season_player_ppg": pd.DataFrame(),
        "empirical_curve": empirical_curves,
        "candidate_curves": pd.DataFrame(),
        "selected_curves": pd.DataFrame(),
        "evaluated_curves": evaluated_curves,
        "replacement_variants": replacement_variants,
        "vorp_variants": vorp_variants,
        "active_replacement_method": selected_replacement_method,
        "replacement": replacement,
        "vorp_table": vorp_table,
        "historical_source": "canonical_anchor_fallback",
        "public_runtime_mode": runtime_mode,
        "position_match_summary": pd.DataFrame(),
        "matched_position_environments": pd.DataFrame(),
        "coverage_frame": pd.DataFrame(),
    }


def _curve_dataset_for_environment(artifacts, environment_key: str, dataset: str) -> pd.DataFrame:
    return artifacts.curves[
        (artifacts.curves["environment_key"] == environment_key) & (artifacts.curves["dataset"] == dataset)
    ][["position", "rank", "expected_ppg", "dataset"]].copy()


def _interpolate_curve_dataset(
    lower_frame: pd.DataFrame,
    upper_frame: pd.DataFrame,
    *,
    lower_value: float,
    upper_value: float,
    target_value: float,
    dataset: str,
) -> pd.DataFrame:
    merged = lower_frame.merge(
        upper_frame[["position", "rank", "expected_ppg"]].rename(columns={"expected_ppg": "upper_expected_ppg"}),
        on=["position", "rank"],
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame(columns=["position", "rank", "expected_ppg", "dataset"])
    if abs(upper_value - lower_value) < 1e-9:
        weight = 0.0
    else:
        weight = (target_value - lower_value) / (upper_value - lower_value)
    merged["expected_ppg"] = merged["expected_ppg"] + weight * (merged["upper_expected_ppg"] - merged["expected_ppg"])
    merged["dataset"] = dataset
    return merged[["position", "rank", "expected_ppg", "dataset"]].sort_values(["position", "rank"]).reset_index(drop=True)


def build_public_target_curve_bundle(
    *,
    artifacts,
    source_key: str,
    target_league: LeagueSettings,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Build public target curves using canonical scoring interpolation when possible."""

    qb_format = detect_qb_format(target_league)
    target_reception = detect_reception_value(target_league)
    if qb_format == "sf":
        lower_key, upper_key = "sf_half_ppr", "sf_ppr"
        lower_value, upper_value = 0.5, 1.0
    else:
        lower_key, upper_key = "1qb_half_ppr", "1qb_ppr"
        lower_value, upper_value = 0.5, 1.0

    available_curve_keys = set(artifacts.curves["environment_key"].astype(str).unique())
    if lower_key in available_curve_keys and upper_key in available_curve_keys:
        lower_fitted = _curve_dataset_for_environment(artifacts, lower_key, "fitted")
        upper_fitted = _curve_dataset_for_environment(artifacts, upper_key, "fitted")
        lower_empirical = _curve_dataset_for_environment(artifacts, lower_key, "empirical")
        upper_empirical = _curve_dataset_for_environment(artifacts, upper_key, "empirical")
        fitted_curves = _interpolate_curve_dataset(
            lower_fitted,
            upper_fitted,
            lower_value=lower_value,
            upper_value=upper_value,
            target_value=target_reception,
            dataset="fitted",
        )
        empirical_curves = _interpolate_curve_dataset(
            lower_empirical,
            upper_empirical,
            lower_value=lower_value,
            upper_value=upper_value,
            target_value=target_reception,
            dataset="empirical",
        )
        if not fitted_curves.empty:
            return fitted_curves, empirical_curves, "no_history_scoring_interpolated"

    return (
        _curve_dataset_for_environment(artifacts, source_key, "fitted"),
        _curve_dataset_for_environment(artifacts, source_key, "empirical"),
        "no_history",
    )


def build_public_anchor_projection(
    source_key: str,
    target_environment: dict[str, Any],
    artifacts,
    canonical_adp_paths: dict[str, Path] | None = None,
    adp_provider: Any | None = None,
    force_adp_refresh: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Project a user league from the selected canonical production anchor."""

    if canonical_adp_paths is None and adp_provider is None:
        canonical_adp_paths, saved_adp_metadata = load_saved_canonical_adp_paths()
    else:
        saved_adp_metadata = None

    if canonical_adp_paths is not None:
        source_adp_by_environment, adp_source_summary = load_source_adp_by_environment(
            canonical_adp_paths=canonical_adp_paths,
        )
        if source_key not in source_adp_by_environment:
            raise ConfigError(f"Canonical ADP is unavailable for {source_key}.")
        source_adp = source_adp_by_environment[source_key]
        source_adp_metadata = adp_source_summary.get("formats", {}).get(source_key, {})
        if saved_adp_metadata is not None:
            merged_metadata = dict(saved_adp_metadata.get("formats", {}).get(source_key, {}))
            merged_metadata.update(source_adp_metadata)
            source_adp_metadata = merged_metadata or source_adp_metadata
    else:
        provider = adp_provider or BeatADPProvider()
        canonical_config = artifacts.metadata["canonical_environments"][source_key]
        source_adp, source_adp_metadata = provider.load_environment(
            source_key,
            num_teams=int(canonical_config["team_count"]),
            force_refresh=force_adp_refresh,
        )
    spec = {
        "model_name": artifacts.metadata["selected_model_name"],
        "metric_mode": artifacts.metadata["selected_metric_mode"],
        "replacement_method": artifacts.metadata["selected_replacement_method"],
        "utility_transform": artifacts.metadata["selected_utility_transform"],
        "weight_power": float(artifacts.metadata["selected_weight_power"]),
    }
    curves = artifacts.curves[artifacts.curves["environment_key"] == source_key]
    fitted_curves = curves[curves["dataset"] == "fitted"].copy()
    replacement_variants: dict[str, pd.DataFrame] = {}
    vorp_variants: dict[str, pd.DataFrame] = {}
    for replacement_method in ("starter_demand", "historical_roster"):
        replacement = artifacts.replacement[
            (artifacts.replacement["environment_key"] == source_key)
            & (artifacts.replacement["replacement_method"] == replacement_method)
        ].copy()
        replacement_variants[replacement_method] = replacement
        vorp_variants[replacement_method] = build_vorp_table(fitted_curves, replacement)

    source_environment = {
        "vorp_variants": vorp_variants,
        "replacement_variants": replacement_variants,
    }
    target_prediction = predict_between_canonical_environments(
        source_key=source_key,
        target_key=source_key,
        source_environment=source_environment,
        target_environment=target_environment,
        source_adp=source_adp,
        model_spec=spec,
    )
    target_prediction["selected_canonical_key"] = source_key
    target_prediction["selected_canonical_label"] = CANONICAL_LABELS[source_key]
    return target_prediction, source_adp_metadata


def build_error_by_adp_bucket(predicted: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
    """Summarize error by draft range."""

    actual_adp = actual[["player_name", "position", "adp"]].rename(columns={"adp": "actual_adp"})
    merged = predicted.merge(actual_adp, on=["player_name", "position"], how="inner")
    merged["absolute_error"] = (merged["league_adjusted_adp"] - merged["actual_adp"]).abs()
    merged["bucket"] = pd.cut(
        merged["actual_adp"],
        bins=[0, 12, 24, 50, 100, float("inf")],
        labels=["1-12", "13-24", "25-50", "51-100", "101+"],
        include_lowest=True,
    )
    return (
        merged.groupby("bucket", observed=False, as_index=False)
        .agg(player_count=("player_name", "count"), mae=("absolute_error", "mean"))
        .fillna({"mae": 0.0})
    )


def build_aggregated_positional_errors(
    predictions: dict[tuple[str, str], pd.DataFrame],
    source_adp_by_environment: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Aggregate positional errors across directed canonical transformations."""

    rows: list[pd.DataFrame] = []
    for (source_key, target_key), prediction in predictions.items():
        breakdown = positional_error_breakdown(prediction, source_adp_by_environment[target_key])
        breakdown["source_environment"] = source_key
        breakdown["target_environment"] = target_key
        rows.append(breakdown)
    if not rows:
        return pd.DataFrame()
    merged = pd.concat(rows, ignore_index=True)
    return (
        merged.groupby("position", as_index=False)
        .agg(
            player_count=("player_count", "sum"),
            mae=("mae", "mean"),
            median_absolute_error=("median_absolute_error", "mean"),
        )
        .sort_values("position")
        .reset_index(drop=True)
    )


def run_public_canonical_analysis(
    client: SleeperClient,
    production_manager: CanonicalArtifactManager,
    target_league_id: str,
    canonical_adp_paths: dict[str, Path] | None = None,
    adp_provider: Any | None = None,
    modeling_config: ModelingConfig | None = None,
    today: date | None = None,
    force_adp_refresh: bool = False,
) -> dict[str, Any]:
    """Run the public user flow using the selected production canonical model."""

    artifacts = production_manager.load()
    modeling_config = modeling_config or default_modeling_config()
    selected_replacement_method = str(artifacts.metadata.get("selected_replacement_method", "starter_demand"))
    target_league = client.get_league(target_league_id)
    metadata_environment_keys = ordered_canonical_environment_keys(
        artifacts.metadata.get("available_canonical_environments") or artifacts.metadata["canonical_environments"].keys()
    )
    curve_environment_keys = ordered_canonical_environment_keys(set(artifacts.curves["environment_key"].astype(str).unique()))
    available_environment_keys = tuple(
        environment_key for environment_key in metadata_environment_keys if environment_key in set(curve_environment_keys)
    )
    if not available_environment_keys:
        raise ConfigError("Production canonical model does not contain any usable canonical anchors.")
    requested_anchor_key = canonical_environment_key_for_league(
        target_league,
        environment_keys=CANONICAL_ENVIRONMENTS,
    )
    anchor_key = canonical_environment_key_for_league(
        target_league,
        environment_keys=available_environment_keys,
    )
    history_library_bundle = {
        "position_scoring_environments": artifacts.history_position_environments,
        "environment_seasons": artifacts.history_environment_seasons,
        "curve_models": artifacts.history_curve_models,
        "fitted_curves": artifacts.history_curves,
    }
    try:
        target_environment = build_league_environment_from_library(
            league=target_league,
            library_bundle=history_library_bundle,
            replacement_method=selected_replacement_method,
        )
    except ConfigError as exc:
        target_environment = build_public_target_environment_from_anchor(
            artifacts=artifacts,
            source_key=anchor_key,
            target_league=target_league,
        )
        target_environment["fallback_reason"] = str(exc)
    results, source_adp_metadata = build_public_anchor_projection(
        source_key=anchor_key,
        target_environment=target_environment,
        artifacts=artifacts,
        canonical_adp_paths=canonical_adp_paths,
        adp_provider=adp_provider,
        force_adp_refresh=force_adp_refresh,
    )
    return {
        "artifacts": artifacts,
        "target_environment": target_environment,
        "results": results,
        "adp_source_metadata": source_adp_metadata,
        "requested_canonical_key": requested_anchor_key,
        "requested_canonical_label": CANONICAL_LABELS[requested_anchor_key],
        "selected_canonical_key": anchor_key,
        "selected_canonical_label": CANONICAL_LABELS[anchor_key],
        "selected_canonical_fallback": requested_anchor_key != anchor_key,
    }
