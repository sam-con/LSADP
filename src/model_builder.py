"""Six-format canonical model building and validation."""

from __future__ import annotations

from datetime import UTC, date, datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.adp import ADPDataProvider, FantasyCalcADPProvider, match_players_by_identity
from src.analysis import default_modeling_config, load_league_environment
from src.baseline_artifacts import CanonicalArtifactManager
from src.calibration import calibrate_market_values
from src.canonical import (
    canonical_configuration,
    canonical_environment_key_for_league,
    classify_transformation_type,
    detect_qb_format,
    detect_reception_format,
    directed_transform_pairs,
    validate_canonical_configuration,
)
from src.config import APP_VERSION, CANONICAL_ENVIRONMENTS, CANONICAL_LABELS, CANONICAL_LEAGUES
from src.models import ConfigError, ModelingConfig
from src.sleeper import SleeperClient
from src.transform import apply_league_transformation
from src.validation import positional_error_breakdown, score_prediction
from src.vorp import build_vorp_table


def validate_environment_identity(environment_key: str, league) -> None:
    """Validate that a configured canonical league matches its intended format."""

    detected = canonical_environment_key_for_league(league)
    if detected != environment_key:
        raise ConfigError(
            f"Configured canonical league `{environment_key}` loaded as `{detected}` instead of the intended format."
        )


def canonical_team_counts(environment_bundle: dict[str, dict[str, Any]]) -> dict[str, int]:
    """Return canonical team counts keyed by environment."""

    return {
        environment_key: int(environment_bundle[environment_key]["league"].total_rosters)
        for environment_key in CANONICAL_ENVIRONMENTS
    }


def validate_canonical_team_counts(environment_bundle: dict[str, dict[str, Any]]) -> tuple[int, dict[str, int]]:
    """Ensure the canonical environments share one calibration team count."""

    counts = canonical_team_counts(environment_bundle)
    unique_counts = sorted(set(counts.values()))
    if len(unique_counts) != 1:
        mismatch = ", ".join(f"{environment_key}={count}" for environment_key, count in counts.items())
        raise ConfigError(
            "Canonical Sleeper leagues do not share one team count, so FantasyCalc cannot be queried against a single "
            f"canonical market size. Found: {mismatch}"
        )
    return unique_counts[0], counts


def summarize_canonical_market_distinctness(source_adp_by_environment: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Measure whether the canonical ADP feeds are materially distinct."""

    rows: list[dict[str, Any]] = []
    for left_key, right_key in combinations(CANONICAL_ENVIRONMENTS, 2):
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
    """Block calibration if FantasyCalc appears to be returning identical markets."""

    diagnostics = summarize_canonical_market_distinctness(source_adp_by_environment)
    suspicious = diagnostics[diagnostics["status"] == "Suspiciously similar"]
    if not suspicious.empty:
        offenders = ", ".join(
            f"{row.left_environment} vs {row.right_environment}"
            for row in suspicious.itertuples(index=False)
        )
        raise ConfigError(
            "FantasyCalc returned canonical ADP markets that appear effectively identical. "
            f"Check the provider/query parameters before calibrating: {offenders}"
        )
    return diagnostics


def load_source_adp_by_environment(
    environment_bundle: dict[str, dict[str, Any]],
    *,
    canonical_adp_paths: dict[str, Path] | None = None,
    adp_provider: FantasyCalcADPProvider | None = None,
    force_adp_refresh: bool = False,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Load canonical ADP inputs either from CSV fixtures or live FantasyCalc."""

    if canonical_adp_paths is not None:
        frames: dict[str, pd.DataFrame] = {}
        metadata: dict[str, dict[str, Any]] = {}
        for environment_key in CANONICAL_ENVIRONMENTS:
            frame, entry = ADPDataProvider(canonical_adp_paths[environment_key]).load_with_metadata()
            frame["canonical_format"] = environment_key
            frames[environment_key] = frame
            metadata[environment_key] = entry
        return frames, {
            "source": "csv",
            "status": "Healthy",
            "last_refresh": None,
            "canonical_team_count": sorted(set(canonical_team_counts(environment_bundle).values())),
            "formats": metadata,
        }

    provider = adp_provider or FantasyCalcADPProvider()
    _, counts = validate_canonical_team_counts(environment_bundle)
    bundle = provider.load_canonical_markets(counts, force_refresh=force_adp_refresh)
    return bundle["frames"], bundle


def build_canonical_environment_bundle(
    client: SleeperClient,
    canonical_leagues: dict[str, str] | None = None,
    modeling_config: ModelingConfig | None = None,
    today: date | None = None,
) -> dict[str, dict[str, Any]]:
    """Build historical environments for all six canonical leagues."""

    canonical_leagues = canonical_leagues or CANONICAL_LEAGUES
    validate_canonical_configuration(canonical_leagues)
    modeling_config = modeling_config or default_modeling_config()

    bundle: dict[str, dict[str, Any]] = {}
    for environment_key in CANONICAL_ENVIRONMENTS:
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
    """Run the 30 directed validations for one candidate spec."""

    validation_rows: list[dict[str, Any]] = []
    predictions: dict[tuple[str, str], pd.DataFrame] = {}

    for source_key, target_key in directed_transform_pairs():
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

    rows: list[dict[str, Any]] = []
    for held_out_key in CANONICAL_ENVIRONMENTS:
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
    adp_provider: FantasyCalcADPProvider | None = None,
    modeling_config: ModelingConfig | None = None,
    today: date | None = None,
    force_adp_refresh: bool = False,
) -> dict[str, Any]:
    """Build, validate, and package a six-format candidate model."""

    canonical_leagues = canonical_leagues or CANONICAL_LEAGUES
    validate_canonical_configuration(canonical_leagues, canonical_adp_paths)
    modeling_config = modeling_config or default_modeling_config()

    environment_bundle = build_canonical_environment_bundle(
        client=client,
        canonical_leagues=canonical_leagues,
        modeling_config=modeling_config,
        today=today,
    )
    canonical_team_count, canonical_team_counts_by_environment = validate_canonical_team_counts(environment_bundle)
    source_adp_by_environment, adp_source_summary = load_source_adp_by_environment(
        environment_bundle,
        canonical_adp_paths=canonical_adp_paths,
        adp_provider=adp_provider,
        force_adp_refresh=force_adp_refresh,
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
    for environment_key in CANONICAL_ENVIRONMENTS:
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
    )
    for environment_key in CANONICAL_ENVIRONMENTS:
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
        "canonical_environments": canonical_config_payload,
        "adp_snapshot": {
            "source": adp_source_summary["source"],
            "status": adp_source_summary["status"],
            "last_refresh": adp_source_summary.get("last_refresh"),
            "canonical_team_count": canonical_team_count,
            "team_counts_by_environment": canonical_team_counts_by_environment,
            "formats": adp_source_summary["formats"],
            "market_distinctness": market_distinctness.to_dict(orient="records"),
        },
        "historical_seasons": {
            environment_key: [league.season for league in environment_bundle[environment_key]["historical_leagues"]]
            for environment_key in CANONICAL_ENVIRONMENTS
        },
        "validation_complete": True,
    }

    leave_one_out = build_leave_one_environment_out_summary(all_validation)
    grouped_type = summarize_validation_by_group(best_validation, "transform_type")
    best_validation["source_label"] = best_validation["source_environment"].map(CANONICAL_LABELS)
    best_validation["target_label"] = best_validation["target_environment"].map(CANONICAL_LABELS)

    return {
        "environment_bundle": environment_bundle,
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


def build_public_anchor_projection(
    source_key: str,
    target_environment: dict[str, Any],
    artifacts,
    canonical_adp_paths: dict[str, Path] | None = None,
    adp_provider: FantasyCalcADPProvider | None = None,
    force_adp_refresh: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Project a user league from the selected canonical production anchor."""

    if canonical_adp_paths is not None:
        source_adp, source_adp_metadata = ADPDataProvider(canonical_adp_paths[source_key]).load_with_metadata()
    else:
        provider = adp_provider or FantasyCalcADPProvider()
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

    merged = predicted.merge(actual[["player_name", "position", "adp"]], on=["player_name", "position"], how="inner")
    merged["absolute_error"] = (merged["league_adjusted_adp"] - merged["adp"]).abs()
    merged["bucket"] = pd.cut(
        merged["adp"],
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
    adp_provider: FantasyCalcADPProvider | None = None,
    modeling_config: ModelingConfig | None = None,
    today: date | None = None,
    force_adp_refresh: bool = False,
) -> dict[str, Any]:
    """Run the public user flow using the selected production canonical model."""

    artifacts = production_manager.load()
    modeling_config = modeling_config or default_modeling_config()
    target_environment = load_league_environment(
        client=client,
        league_id=target_league_id,
        modeling_config=modeling_config,
        today=today,
        replacement_method="starter_demand",
    )
    anchor_key = canonical_environment_key_for_league(target_environment["league"])
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
        "selected_canonical_key": anchor_key,
        "selected_canonical_label": CANONICAL_LABELS[anchor_key],
    }
