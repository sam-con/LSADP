"""End-to-end analysis orchestration."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

import pandas as pd

from src.adp import ADPDataProvider
from src.baseline_artifacts import BaselineArtifactManager
from src.calibration import calibrate_market_values
from src.config import (
    ADP_1QB_PATH,
    ADP_SUPERFLEX_PATH,
    APP_VERSION,
    BASELINE_1QB_LEAGUE_ID,
    BASELINE_SF_LEAGUE_ID,
    CORE_POSITIONS,
    CURVE_SELECTION_RELATIVE_IMPROVEMENT,
    DEFAULT_COVERAGE_MIN_WEEKS,
    DEFAULT_MIN_GAMES,
    DEFAULT_MIN_PLAYER_WEEKS_BY_POSITION,
    DEFAULT_MIN_PLAYERS_BY_POSITION,
)
from src.curves import aggregate_rank_curves, build_season_player_ppg, evaluate_curve_series, select_curve_models
from src.historical_scores import (
    enrich_player_weeks_with_metadata,
    load_historical_player_weeks,
    summarize_historical_coverage,
    validate_historical_coverage,
)
from src.models import ConfigError, HistoricalCoverage, LeagueSettings, ModelingConfig
from src.replacement import attach_replacement_ppg, calculate_historical_roster_replacement, calculate_starter_demand_replacement
from src.sleeper import SleeperClient
from src.sleeper_history import load_league_chain, select_required_history, validate_scoring_consistency
from src.transform import apply_league_transformation
from src.utils import adp_utility
from src.validation import positional_error_breakdown, score_prediction
from src.vorp import build_vorp_table, merge_metric_into_players


def default_modeling_config() -> ModelingConfig:
    return ModelingConfig(
        min_games=DEFAULT_MIN_GAMES,
        aggregation="median",
        recency_weights={},
        min_coverage_weeks=DEFAULT_COVERAGE_MIN_WEEKS,
        min_player_weeks_by_position=DEFAULT_MIN_PLAYER_WEEKS_BY_POSITION.copy(),
        min_players_by_position=DEFAULT_MIN_PLAYERS_BY_POSITION.copy(),
        curve_selection_relative_improvement=CURVE_SELECTION_RELATIVE_IMPROVEMENT,
    )


def _evaluate_selected_curves(
    selected_curves: pd.DataFrame,
    empirical_curve: pd.DataFrame,
    minimum_max_rank: int = 80,
) -> pd.DataFrame:
    evaluated_frames: list[pd.DataFrame] = []
    for _, row in selected_curves.iterrows():
        empirical_max = int(empirical_curve[empirical_curve["position"] == row["position"]]["rank"].max())
        max_rank = max(empirical_max, minimum_max_rank)
        evaluated = evaluate_curve_series(
            curve_fit=_curve_row_to_object(row),
            max_rank=max_rank,
        )
        evaluated["dataset"] = "fitted"
        evaluated_frames.append(evaluated)

    empirical = empirical_curve.copy()
    empirical["dataset"] = "empirical"
    return pd.concat([*evaluated_frames, empirical], ignore_index=True)


def _curve_row_to_object(row: pd.Series):
    from src.models import CurveFitResult

    return CurveFitResult(
        position=str(row["position"]),
        model_name=str(row["model_name"]),
        a=float(row["a"]),
        c=float(row["c"]),
        k=float(row["k"]),
        rmse=float(row["rmse"]),
        mae=float(row["mae"]),
        r2=float(row["r2"]),
        aic=float(row["aic"]),
        bic=float(row["bic"]),
        cv_rmse=float(row["cv_rmse"]),
        historical_window=str(row["historical_window"]),
        replacement_rank=int(row["replacement_rank"]) if pd.notna(row.get("replacement_rank")) else None,
        replacement_ppg=float(row["replacement_ppg"]) if pd.notna(row.get("replacement_ppg")) else None,
    )


def load_league_environment(
    client: SleeperClient,
    league_id: str,
    modeling_config: ModelingConfig | None = None,
    today: date | None = None,
    replacement_method: str = "starter_demand",
) -> dict[str, Any]:
    """Build the full historical environment for a target Sleeper league."""

    modeling_config = modeling_config or default_modeling_config()
    today = today or date.today()

    league = client.get_league(league_id)
    chain = load_league_chain(client, league_id=league_id)
    historical_leagues = select_required_history(chain=chain, today=today)
    validate_scoring_consistency(current_league=league, historical_leagues=historical_leagues)

    players_payload = client.get_players("nfl")

    seasonal_frames: list[pd.DataFrame] = []
    coverage: list[HistoricalCoverage] = []
    for historical_league in historical_leagues:
        season_frame, weeks_loaded = load_historical_player_weeks(client, historical_league)
        enriched = enrich_player_weeks_with_metadata(season_frame, players_payload)
        seasonal_frames.append(enriched)
        coverage.append(
            summarize_historical_coverage(
                enriched,
                season=historical_league.season,
                weeks_loaded=weeks_loaded,
                min_games=modeling_config.min_games,
            )
        )

    validate_historical_coverage(
        coverage_summaries=coverage,
        min_weeks=modeling_config.min_coverage_weeks,
        min_player_weeks_by_position=modeling_config.min_player_weeks_by_position,
        min_players_by_position=modeling_config.min_players_by_position,
    )

    player_weeks = pd.concat(seasonal_frames, ignore_index=True)
    season_player_ppg = build_season_player_ppg(player_weeks, min_games=modeling_config.min_games)
    empirical_curve = aggregate_rank_curves(
        season_player_ppg=season_player_ppg,
        aggregation=modeling_config.aggregation,
        recency_weights=modeling_config.recency_weights,
    )
    candidate_curves, selected_curves = select_curve_models(
        season_player_ppg=season_player_ppg,
        aggregated_curve=empirical_curve,
        historical_window=f"{historical_leagues[0].season}-{historical_leagues[-1].season}",
        minimum_relative_improvement=modeling_config.curve_selection_relative_improvement,
    )
    evaluated_curves = _evaluate_selected_curves(selected_curves, empirical_curve)

    fitted_curves = evaluated_curves[evaluated_curves["dataset"] == "fitted"].copy()
    starter_replacement = calculate_starter_demand_replacement(league=league, curves=fitted_curves)
    historical_roster_replacement = calculate_historical_roster_replacement(player_weeks)
    historical_roster_replacement = attach_replacement_ppg(historical_roster_replacement, fitted_curves)

    replacement_variants = {
        "starter_demand": starter_replacement,
        "historical_roster": historical_roster_replacement,
    }
    vorp_variants = {
        key: build_vorp_table(fitted_curves, replacement_frame)
        for key, replacement_frame in replacement_variants.items()
    }
    replacement = replacement_variants[replacement_method]

    replacement_by_position = replacement.set_index("position")
    selected_curves = selected_curves.copy()
    for position in CORE_POSITIONS:
        if position not in replacement_by_position.index:
            continue
        selected_curves.loc[selected_curves["position"] == position, "replacement_rank"] = int(
            replacement_by_position.loc[position, "replacement_rank"]
        )
        selected_curves.loc[selected_curves["position"] == position, "replacement_ppg"] = float(
            replacement_by_position.loc[position, "replacement_ppg"]
        )

    return {
        "league": league,
        "historical_leagues": historical_leagues,
        "coverage": coverage,
        "player_weeks": player_weeks,
        "season_player_ppg": season_player_ppg,
        "empirical_curve": empirical_curve,
        "candidate_curves": candidate_curves,
        "selected_curves": selected_curves,
        "evaluated_curves": evaluated_curves,
        "replacement_variants": replacement_variants,
        "vorp_variants": vorp_variants,
        "active_replacement_method": replacement_method,
        "replacement": replacement,
        "vorp_table": vorp_variants[replacement_method],
    }


def build_baseline_artifacts(
    client: SleeperClient,
    baseline_league_id: str,
    current_adp_path=ADP_1QB_PATH,
    known_superflex_league_id: str = BASELINE_SF_LEAGUE_ID,
    modeling_config: ModelingConfig | None = None,
    today: date | None = None,
) -> dict[str, Any]:
    """Build and return baseline artifacts plus diagnostics."""

    environment = load_league_environment(
        client=client,
        league_id=baseline_league_id,
        modeling_config=modeling_config,
        today=today,
        replacement_method="starter_demand",
    )

    adp = ADPDataProvider(current_adp_path).load()
    baseline_players = merge_metric_into_players(adp, environment["vorp_table"], metric_column="vorp", output_column="baseline_vorp")
    baseline_players = merge_metric_into_players(
        baseline_players,
        environment["vorp_table"],
        metric_column="expected_ppg",
        output_column="baseline_expected_ppg",
    )
    calibration_model, calibrated_players = calibrate_market_values(
        baseline_players,
        metric_column="baseline_vorp",
        metric_name="vorp",
    )

    metadata = {
        "baseline_1qb_league_id": baseline_league_id,
        "baseline_superflex_league_id": known_superflex_league_id,
        "historical_seasons": [league.season for league in environment["historical_leagues"]],
        "baseline_scoring_settings": environment["league"].scoring_settings,
        "roster_positions": environment["league"].roster_positions,
        "curve_model_selection": environment["selected_curves"].to_dict(orient="records"),
        "replacement_methodology": "Starter Demand Replacement",
        "adp_source": str(current_adp_path),
        "generated_timestamp": datetime.now(UTC).isoformat(),
        "model_version": APP_VERSION,
    }

    return {
        "environment": environment,
        "adp": adp,
        "calibrated_players": calibrated_players,
        "calibration_model": calibration_model,
        "metadata": metadata,
    }


def save_baseline_artifacts(
    artifact_manager: BaselineArtifactManager,
    baseline_bundle: dict[str, Any],
) -> None:
    """Persist a computed baseline bundle to disk."""

    environment = baseline_bundle["environment"]
    artifact_manager.save(
        curves=environment["evaluated_curves"],
        replacement=environment["replacement"],
        model=baseline_bundle["calibration_model"],
        metadata=baseline_bundle["metadata"],
    )


def load_current_market_context(
    artifact_manager: BaselineArtifactManager,
    current_adp_path=ADP_1QB_PATH,
) -> dict[str, Any]:
    """Load saved baseline artifacts plus the current 1QB ADP market."""

    artifacts = artifact_manager.load()
    current_adp = ADPDataProvider(current_adp_path).load()
    fitted_curves = artifacts.curves[artifacts.curves["dataset"] == "fitted"].copy()
    empirical_curves = artifacts.curves[artifacts.curves["dataset"] == "empirical"].copy()
    baseline_vorp = build_vorp_table(fitted_curves, artifacts.replacement)
    baseline_players = merge_metric_into_players(current_adp, baseline_vorp, metric_column="vorp", output_column="baseline_vorp")
    baseline_players = merge_metric_into_players(
        baseline_players,
        baseline_vorp,
        metric_column="expected_ppg",
        output_column="baseline_expected_ppg",
    )
    baseline_players = baseline_players.merge(
        artifacts.model.rename(
            columns={
                "intercept": "market_intercept",
                "coefficient": "market_coefficient",
            }
        ),
        on="position",
        how="left",
    )
    baseline_players["utility"] = adp_utility(baseline_players["adp"]).to_numpy(dtype=float)

    return {
        "artifacts": artifacts,
        "current_adp": current_adp,
        "baseline_players": baseline_players,
        "baseline_vorp": baseline_vorp,
        "baseline_fitted_curves": fitted_curves,
        "baseline_empirical_curves": empirical_curves,
    }


def run_public_analysis(
    client: SleeperClient,
    artifact_manager: BaselineArtifactManager,
    target_league_id: str,
    current_adp_path=ADP_1QB_PATH,
    modeling_config: ModelingConfig | None = None,
    today: date | None = None,
) -> dict[str, Any]:
    """Run the public app workflow using saved baseline artifacts."""

    baseline_context = load_current_market_context(artifact_manager=artifact_manager, current_adp_path=current_adp_path)
    target_environment = load_league_environment(
        client=client,
        league_id=target_league_id,
        modeling_config=modeling_config,
        today=today,
        replacement_method="starter_demand",
    )
    target_players = merge_metric_into_players(
        baseline_context["baseline_players"],
        target_environment["vorp_table"],
        metric_column="vorp",
        output_column="league_vorp",
    )
    target_players = merge_metric_into_players(
        target_players,
        target_environment["vorp_table"],
        metric_column="expected_ppg",
        output_column="league_expected_ppg",
    )
    results = apply_league_transformation(
        calibrated_adp=target_players,
        baseline_metric_column="baseline_vorp",
        target_metric_column="league_vorp",
        explanation_metric_label="VORP",
    )
    return {
        "baseline_context": baseline_context,
        "target_environment": target_environment,
        "results": results,
    }


def run_superflex_validation(
    client: SleeperClient,
    baseline_1qb_league_id: str = BASELINE_1QB_LEAGUE_ID,
    target_superflex_league_id: str = BASELINE_SF_LEAGUE_ID,
    adp_1qb_path=ADP_1QB_PATH,
    adp_superflex_path=ADP_SUPERFLEX_PATH,
    modeling_config: ModelingConfig | None = None,
    today: date | None = None,
) -> dict[str, Any]:
    """Run the 1QB -> Superflex validation and ablation comparison."""

    if not baseline_1qb_league_id or not target_superflex_league_id:
        raise ConfigError("Configure BASELINE_1QB_LEAGUE_ID and BASELINE_SF_LEAGUE_ID before running validation.")

    modeling_config = modeling_config or default_modeling_config()
    baseline_bundle = build_baseline_artifacts(
        client=client,
        baseline_league_id=baseline_1qb_league_id,
        current_adp_path=adp_1qb_path,
        known_superflex_league_id=target_superflex_league_id,
        modeling_config=modeling_config,
        today=today,
    )
    target_environment = load_league_environment(
        client=client,
        league_id=target_superflex_league_id,
        modeling_config=modeling_config,
        today=today,
        replacement_method="starter_demand",
    )
    historical_roster_target = load_league_environment(
        client=client,
        league_id=target_superflex_league_id,
        modeling_config=modeling_config,
        today=today,
        replacement_method="historical_roster",
    )
    actual_superflex_adp = ADPDataProvider(adp_superflex_path).load()
    baseline_players = baseline_bundle["calibrated_players"].copy()

    target_vorp_players = merge_metric_into_players(
        baseline_players,
        target_environment["vorp_table"],
        metric_column="vorp",
        output_column="league_vorp",
    )
    target_vorp_players = merge_metric_into_players(
        target_vorp_players,
        target_environment["vorp_table"],
        metric_column="expected_ppg",
        output_column="league_expected_ppg",
    )
    target_roster_players = merge_metric_into_players(
        baseline_players,
        historical_roster_target["vorp_table"],
        metric_column="vorp",
        output_column="league_vorp",
    )
    target_roster_players = merge_metric_into_players(
        target_roster_players,
        historical_roster_target["vorp_table"],
        metric_column="expected_ppg",
        output_column="league_expected_ppg",
    )

    curve_only_model, _ = calibrate_market_values(
        baseline_bundle["calibrated_players"].copy(),
        metric_column="baseline_expected_ppg",
        metric_name="expected_ppg",
    )
    curve_only_target = target_vorp_players.merge(
        curve_only_model[["position", "coefficient"]].rename(columns={"coefficient": "curve_only_coefficient"}),
        on="position",
        how="left",
    )
    curve_only_target["utility"] = baseline_bundle["calibrated_players"]["utility"].to_numpy(dtype=float)
    curve_only_target["market_coefficient"] = curve_only_target["curve_only_coefficient"].to_numpy(dtype=float)

    models = {
        "No Adjustment": baseline_bundle["adp"].rename(columns={"adp": "league_adjusted_adp"}),
        "Curve Only": apply_league_transformation(
            curve_only_target,
            baseline_metric_column="baseline_expected_ppg",
            target_metric_column="league_expected_ppg",
            explanation_metric_label="PPG",
        ),
        "Curve + Starter VORP": apply_league_transformation(
            target_vorp_players,
            baseline_metric_column="baseline_vorp",
            target_metric_column="league_vorp",
            explanation_metric_label="VORP",
        ),
        "Curve + Roster VORP": apply_league_transformation(
            target_roster_players,
            baseline_metric_column="baseline_vorp",
            target_metric_column="league_vorp",
            explanation_metric_label="VORP",
        ),
    }

    metrics = []
    positional_breakdowns = {}
    for model_name, prediction in models.items():
        if model_name == "No Adjustment":
            prediction = prediction.copy()
            prediction["league_adjusted_adp"] = prediction["league_adjusted_adp"].astype(float)
        metric_row = score_prediction(prediction, actual_superflex_adp, model_name=model_name)
        metrics.append(metric_row)
        positional_breakdowns[model_name] = positional_error_breakdown(prediction, actual_superflex_adp)

    return {
        "baseline_bundle": baseline_bundle,
        "target_environment": target_environment,
        "historical_roster_target": historical_roster_target,
        "actual_superflex_adp": actual_superflex_adp,
        "predictions": models,
        "metrics": pd.DataFrame(metrics).sort_values("weighted_mae").reset_index(drop=True),
        "positional_breakdowns": positional_breakdowns,
    }
