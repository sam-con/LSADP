"""Position-specific historical scoring library construction and matching."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

import pandas as pd

from src.config import APP_VERSION, CORE_POSITIONS, DEFAULT_MIN_GAMES
from src.curves import aggregate_rank_curves, build_season_player_ppg, evaluate_curve_series, select_curve_models
from src.historical_scores import enrich_player_weeks_with_metadata, load_historical_player_weeks
from src.models import ConfigError, CurveFitResult, HistoricalLeagueSummary, LeagueSettings, ModelingConfig
from src.replacement import attach_replacement_ppg, calculate_historical_roster_replacement, calculate_starter_demand_replacement
from src.sleeper import SleeperClient, as_historical_summary
from src.utils import normalize_scoring_settings, required_completed_seasons

POSITION_SIGNATURE_VERSION = "position-signature-v1"

POSITION_SCORING_KEYS = {
    "QB": {
        "bonus_pass_yd_300",
        "bonus_pass_yd_400",
        "bonus_rush_yd_100",
        "bonus_rush_yd_200",
        "fum_lost",
        "pass_2pt",
        "pass_att",
        "pass_cmp",
        "pass_fd",
        "pass_icmp",
        "pass_int",
        "pass_td",
        "pass_yd",
        "rush_2pt",
        "rush_fd",
        "rush_td",
        "rush_yd",
    },
    "RB": {
        "bonus_rec_yd_100",
        "bonus_rec_yd_200",
        "bonus_rush_yd_100",
        "bonus_rush_yd_200",
        "fum_lost",
        "rec",
        "rec_2pt",
        "rec_fd",
        "rec_td",
        "rec_yd",
        "rush_2pt",
        "rush_fd",
        "rush_td",
        "rush_yd",
    },
    "WR": {
        "bonus_rec_yd_100",
        "bonus_rec_yd_200",
        "bonus_rush_yd_100",
        "bonus_rush_yd_200",
        "fum_lost",
        "rec",
        "rec_2pt",
        "rec_fd",
        "rec_td",
        "rec_yd",
        "rush_2pt",
        "rush_fd",
        "rush_td",
        "rush_yd",
    },
    "TE": {
        "bonus_rec_te",
        "bonus_rec_yd_100",
        "bonus_rec_yd_200",
        "bonus_rush_yd_100",
        "bonus_rush_yd_200",
        "fum_lost",
        "rec",
        "rec_2pt",
        "rec_fd",
        "rec_td",
        "rec_te",
        "rec_yd",
        "rush_2pt",
        "rush_fd",
        "rush_td",
        "rush_yd",
    },
}

POSITION_DISTANCE_WEIGHTS = {
    "QB": {
        "pass_td": 4.0,
        "pass_yd": 0.5,
        "pass_int": 2.5,
        "pass_2pt": 1.5,
        "pass_cmp": 0.2,
        "pass_icmp": 0.2,
        "pass_att": 0.1,
        "pass_fd": 1.0,
        "bonus_pass_yd_300": 0.75,
        "bonus_pass_yd_400": 1.0,
        "rush_td": 2.5,
        "rush_yd": 0.6,
        "rush_2pt": 1.2,
        "rush_fd": 0.8,
        "bonus_rush_yd_100": 0.75,
        "bonus_rush_yd_200": 1.0,
        "fum_lost": 1.5,
    },
    "RB": {
        "rec": 4.0,
        "rec_yd": 0.6,
        "rec_td": 2.5,
        "rec_2pt": 1.2,
        "rec_fd": 1.0,
        "bonus_rec_yd_100": 0.75,
        "bonus_rec_yd_200": 1.0,
        "rush_td": 2.5,
        "rush_yd": 0.7,
        "rush_2pt": 1.2,
        "rush_fd": 1.0,
        "bonus_rush_yd_100": 0.75,
        "bonus_rush_yd_200": 1.0,
        "fum_lost": 1.5,
    },
    "WR": {
        "rec": 4.0,
        "rec_yd": 0.6,
        "rec_td": 2.5,
        "rec_2pt": 1.2,
        "rec_fd": 1.0,
        "bonus_rec_yd_100": 0.75,
        "bonus_rec_yd_200": 1.0,
        "rush_td": 2.0,
        "rush_yd": 0.5,
        "rush_2pt": 1.0,
        "rush_fd": 0.8,
        "bonus_rush_yd_100": 0.5,
        "bonus_rush_yd_200": 0.75,
        "fum_lost": 1.5,
    },
    "TE": {
        "rec": 4.0,
        "rec_te": 3.0,
        "bonus_rec_te": 3.0,
        "rec_yd": 0.6,
        "rec_td": 2.5,
        "rec_2pt": 1.2,
        "rec_fd": 1.0,
        "bonus_rec_yd_100": 0.75,
        "bonus_rec_yd_200": 1.0,
        "rush_td": 1.5,
        "rush_yd": 0.4,
        "rush_2pt": 1.0,
        "rush_fd": 0.8,
        "bonus_rush_yd_100": 0.5,
        "bonus_rush_yd_200": 0.75,
        "fum_lost": 1.5,
    },
}

POSITION_DISTANCE_THRESHOLDS = {
    "QB": {"very_close": 1.5, "approximate": 4.0},
    "RB": {"very_close": 1.0, "approximate": 3.0},
    "WR": {"very_close": 1.0, "approximate": 3.0},
    "TE": {"very_close": 1.5, "approximate": 4.0},
}


@dataclass(slots=True)
class PositionEnvironmentMatch:
    position: str
    target_hash: str
    matched_hash: str | None
    match_quality: str
    exact_match: bool
    distance: float | None
    status: str
    target_scoring: dict[str, float]
    matched_scoring: dict[str, float]
    differing_fields: list[dict[str, float]]
    seasons: list[int]
    max_reliable_rank: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize_value(value: Any) -> float:
    return round(float(value), 6)


def _curve_row_to_object(row: pd.Series) -> CurveFitResult:
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


def _evaluate_selected_curves(selected_curves: pd.DataFrame, empirical_curve: pd.DataFrame, minimum_max_rank: int = 80) -> pd.DataFrame:
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
    return pd.concat([*evaluated_frames, empirical], ignore_index=True) if evaluated_frames else empirical


def scoring_key_affects_position(key: str, position: str) -> bool:
    if key in POSITION_SCORING_KEYS[position]:
        return True
    if key.startswith("pass_") or key.startswith("bonus_pass_"):
        return position == "QB"
    if key in {"fum_lost"}:
        return True
    if key.endswith("_te") or key.startswith("bonus_rec_te"):
        return position == "TE"
    if key.startswith("rush_") or key.startswith("bonus_rush_"):
        return True
    if key.startswith("rec_") or key.startswith("bonus_rec_") or key == "rec":
        return position in {"RB", "WR", "TE"}
    return False


def normalize_position_scoring(scoring_settings: dict[str, Any], position: str) -> dict[str, float]:
    normalized = normalize_scoring_settings(scoring_settings)
    keys = {
        key
        for key in normalized
        if scoring_key_affects_position(key, position)
    } | set(POSITION_SCORING_KEYS[position])
    compact: dict[str, float] = {}
    for key in sorted(keys):
        value = _normalize_value(normalized.get(key, 0.0))
        if abs(value) > 1e-9:
            compact[key] = value
    return compact


def serialize_scoring_dict(scoring: dict[str, float]) -> str:
    return json.dumps(scoring, sort_keys=True, separators=(",", ":"))


def build_position_scoring_profile(scoring_settings: dict[str, Any], position: str) -> dict[str, Any]:
    normalized = normalize_position_scoring(scoring_settings, position)
    serialized = serialize_scoring_dict(normalized)
    return {
        "position": position,
        "position_scoring_hash": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        "normalized_scoring": normalized,
        "normalized_scoring_json": serialized,
        "signature_version": POSITION_SIGNATURE_VERSION,
    }


def build_league_position_scoring_profiles(scoring_settings: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        position: build_position_scoring_profile(scoring_settings, position)
        for position in CORE_POSITIONS
    }


def position_scoring_differences(target: dict[str, float], candidate: dict[str, float]) -> list[dict[str, float]]:
    differences: list[dict[str, float]] = []
    for key in sorted(set(target) | set(candidate)):
        target_value = float(target.get(key, 0.0))
        candidate_value = float(candidate.get(key, 0.0))
        if abs(target_value - candidate_value) > 1e-9:
            differences.append(
                {
                    "key": key,
                    "target_value": target_value,
                    "matched_value": candidate_value,
                }
            )
    return differences


def score_position_scoring_distance(position: str, target: dict[str, float], candidate: dict[str, float]) -> float:
    weights = POSITION_DISTANCE_WEIGHTS[position]
    total = 0.0
    for key in sorted(set(target) | set(candidate)):
        diff = abs(float(target.get(key, 0.0)) - float(candidate.get(key, 0.0)))
        if diff <= 1e-9:
            continue
        total += diff * float(weights.get(key, 0.5))
    return round(total, 6)


def classify_match_quality(position: str, *, exact_match: bool, distance: float | None) -> str:
    if exact_match:
        return "Exact"
    if distance is None:
        return "Insufficient"
    thresholds = POSITION_DISTANCE_THRESHOLDS[position]
    if distance <= thresholds["very_close"]:
        return "Very Close"
    if distance <= thresholds["approximate"]:
        return "Approximate"
    return "Insufficient"


def _position_environment_status(season_count: int) -> str:
    if season_count >= 4:
        return "READY_4_SEASONS"
    if season_count == 3:
        return "READY_3_SEASONS"
    if season_count == 2:
        return "LIMITED"
    return "INSUFFICIENT"


def _empty_library_bundle(today: date) -> dict[str, Any]:
    metadata = {
        "generated_at": None,
        "model_version": APP_VERSION,
        "signature_version": POSITION_SIGNATURE_VERSION,
        "required_completed_seasons": required_completed_seasons(today=today, window=4),
    }
    return {
        "player_weeks": pd.DataFrame(),
        "position_scoring_environments": pd.DataFrame(),
        "environment_seasons": pd.DataFrame(),
        "curve_models": pd.DataFrame(),
        "fitted_curves": pd.DataFrame(),
        "seed_leagues": pd.DataFrame(),
        "library_metadata": metadata,
    }


def build_position_history_library(
    client: SleeperClient,
    seed_leagues: pd.DataFrame,
    *,
    today: date | None = None,
    modeling_config: ModelingConfig | None = None,
) -> dict[str, Any]:
    """Build the position-specific historical scoring library from seed leagues."""

    today = today or date.today()
    modeling_config = modeling_config or ModelingConfig(min_games=DEFAULT_MIN_GAMES)
    if seed_leagues.empty:
        return _empty_library_bundle(today)

    players_payload = client.get_players("nfl")
    observed_rows: list[dict[str, Any]] = []
    seed_records: list[dict[str, Any]] = []

    for row in seed_leagues.itertuples(index=False):
        league = client.get_league(str(row.league_id))
        if int(league.season) != int(row.season):
            continue
        historical_league = as_historical_summary(league)
        season_frame, _ = load_historical_player_weeks(client, historical_league)
        enriched = enrich_player_weeks_with_metadata(season_frame, players_payload)
        profiles = build_league_position_scoring_profiles(league.scoring_settings)
        for position in CORE_POSITIONS:
            position_frame = enriched[enriched["position"] == position].copy()
            if position_frame.empty:
                continue
            profile = profiles[position]
            position_frame["position_scoring_hash"] = profile["position_scoring_hash"]
            position_frame["normalized_scoring_json"] = profile["normalized_scoring_json"]
            position_frame["source_league_id"] = str(league.league_id)
            observed_rows.extend(position_frame.to_dict(orient="records"))
        seed_records.append(
            {
                "league_id": str(league.league_id),
                "season": int(league.season),
                "team_count": int(league.total_rosters),
                "league_name": league.name,
                "selected": bool(getattr(row, "selected", True)),
            }
        )

    if not observed_rows:
        return _empty_library_bundle(today)

    player_weeks = pd.DataFrame(observed_rows)
    duplicate_keys = ["season", "week", "player_id", "position", "position_scoring_hash"]
    disagreement = (
        player_weeks.groupby(duplicate_keys)["fantasy_points"]
        .nunique()
        .reset_index(name="value_count")
    )
    bad = disagreement[disagreement["value_count"] > 1]
    if not bad.empty:
        raise ConfigError(
            "Position-specific historical library found conflicting duplicate player-week scores for the same signature. "
            f"Examples: {bad.head(5).to_dict(orient='records')}"
        )
    confirmation = (
        player_weeks.groupby(duplicate_keys, as_index=False)
        .agg(
            fantasy_points=("fantasy_points", "first"),
            player_name=("player_name", "first"),
            team=("team", "first"),
            starter_flag=("starter_flag", "max"),
            roster_id=("roster_id", "first"),
            normalized_scoring_json=("normalized_scoring_json", "first"),
            source_league_id=("source_league_id", "first"),
            confirmation_count=("source_league_id", "nunique"),
        )
        .sort_values(["position", "position_scoring_hash", "season", "week", "player_id"])
        .reset_index(drop=True)
    )

    required_seasons = required_completed_seasons(today=today, window=4)
    environment_rows: list[dict[str, Any]] = []
    season_rows: list[dict[str, Any]] = []
    curve_model_rows: list[pd.DataFrame] = []
    evaluated_curve_rows: list[pd.DataFrame] = []

    for (position, scoring_hash), group in confirmation.groupby(["position", "position_scoring_hash"]):
        normalized_scoring_json = str(group["normalized_scoring_json"].iloc[0])
        normalized_scoring = json.loads(normalized_scoring_json)
        seasons = sorted(int(value) for value in group["season"].unique())
        season_count = len(seasons)
        replacement_rank = None
        replacement_ppg = None
        max_reliable_rank = 0
        selected_model_name = None

        season_player_ppg = build_season_player_ppg(group, min_games=modeling_config.min_games)
        season_position_ppg = season_player_ppg[season_player_ppg["position"] == position].copy()
        if not season_position_ppg.empty:
            empirical_curve = aggregate_rank_curves(
                season_player_ppg=season_position_ppg,
                aggregation=modeling_config.aggregation,
                recency_weights=modeling_config.recency_weights,
            )
            candidate_curves, selected_curves = select_curve_models(
                season_player_ppg=season_position_ppg,
                aggregated_curve=empirical_curve,
                historical_window=f"{seasons[0]}-{seasons[-1]}",
                minimum_relative_improvement=modeling_config.curve_selection_relative_improvement,
            )
            if not selected_curves.empty:
                historical_replacement = calculate_historical_roster_replacement(group)
                historical_replacement = attach_replacement_ppg(historical_replacement, _evaluate_selected_curves(selected_curves, empirical_curve))
                replacement_match = historical_replacement[historical_replacement["position"] == position]
                if not replacement_match.empty:
                    replacement_rank = int(replacement_match.iloc[0]["replacement_rank"])
                    replacement_ppg = float(replacement_match.iloc[0]["replacement_ppg"])
                max_reliable_rank = int(empirical_curve[empirical_curve["position"] == position]["rank"].max())
                selected_model_name = str(selected_curves.iloc[0]["model_name"])
                selected_curves = selected_curves.copy()
                selected_curves["position_scoring_hash"] = scoring_hash
                selected_curves["selected"] = True
                candidate_curves = candidate_curves.copy()
                candidate_curves["position_scoring_hash"] = scoring_hash
                candidate_curves["selected"] = False
                curve_model_rows.extend([candidate_curves, selected_curves])
                evaluated = _evaluate_selected_curves(selected_curves, empirical_curve)
                evaluated["position_scoring_hash"] = scoring_hash
                evaluated["curve_type"] = selected_model_name
                evaluated_curve_rows.append(evaluated)

        status = _position_environment_status(len(set(required_seasons) & set(seasons)))
        environment_rows.append(
            {
                "position": position,
                "position_scoring_hash": scoring_hash,
                "normalized_scoring_json": normalized_scoring_json,
                "league_season_count": int(group[["season", "source_league_id"]].drop_duplicates().shape[0]),
                "season_count": season_count,
                "player_week_count": int(len(group)),
                "unique_player_count": int(group["player_id"].nunique()),
                "first_seen_season": min(seasons),
                "last_seen_season": max(seasons),
                "max_reliable_rank": int(max_reliable_rank),
                "historical_replacement_rank": replacement_rank,
                "historical_replacement_ppg": replacement_ppg,
                "selected_curve_model": selected_model_name,
                "status": status,
                "signature_version": POSITION_SIGNATURE_VERSION,
                "last_updated": today.isoformat(),
            }
        )
        for season in seasons:
            season_group = group[group["season"] == season]
            seasonal_ppg = season_position_ppg[season_position_ppg["season"] == season]
            season_rows.append(
                {
                    "position": position,
                    "position_scoring_hash": scoring_hash,
                    "season": int(season),
                    "player_week_count": int(len(season_group)),
                    "unique_player_count": int(season_group["player_id"].nunique()),
                    "confirmation_count": int(season_group["confirmation_count"].sum()),
                    "max_rank": int(seasonal_ppg["rank"].max()) if not seasonal_ppg.empty else 0,
                }
            )

    environment_frame = pd.DataFrame(environment_rows).sort_values(["position", "position_scoring_hash"]).reset_index(drop=True)
    season_frame = pd.DataFrame(season_rows).sort_values(["position", "position_scoring_hash", "season"]).reset_index(drop=True)
    curve_models = pd.concat(curve_model_rows, ignore_index=True) if curve_model_rows else pd.DataFrame()
    fitted_curves = pd.concat(evaluated_curve_rows, ignore_index=True) if evaluated_curve_rows else pd.DataFrame()

    metadata = {
        "generated_at": today.isoformat(),
        "model_version": APP_VERSION,
        "signature_version": POSITION_SIGNATURE_VERSION,
        "required_completed_seasons": required_seasons,
        "seed_league_count": int(pd.DataFrame(seed_records)["league_id"].nunique()) if seed_records else 0,
    }
    return {
        "player_weeks": confirmation,
        "position_scoring_environments": environment_frame,
        "environment_seasons": season_frame,
        "curve_models": curve_models,
        "fitted_curves": fitted_curves,
        "seed_leagues": pd.DataFrame(seed_records),
        "library_metadata": metadata,
    }


def match_position_scoring_environment(
    position: str,
    scoring_settings: dict[str, Any],
    environments: pd.DataFrame,
) -> PositionEnvironmentMatch:
    target_profile = build_position_scoring_profile(scoring_settings, position)
    target_scoring = target_profile["normalized_scoring"]
    position_envs = environments[environments["position"] == position].copy()
    if position_envs.empty:
        return PositionEnvironmentMatch(
            position=position,
            target_hash=target_profile["position_scoring_hash"],
            matched_hash=None,
            match_quality="Insufficient",
            exact_match=False,
            distance=None,
            status="INSUFFICIENT",
            target_scoring=target_scoring,
            matched_scoring={},
            differing_fields=[],
            seasons=[],
            max_reliable_rank=0,
        )

    exact = position_envs[position_envs["position_scoring_hash"] == target_profile["position_scoring_hash"]]
    if not exact.empty:
        row = exact.iloc[0]
        matched_scoring = json.loads(str(row["normalized_scoring_json"]))
        return PositionEnvironmentMatch(
            position=position,
            target_hash=target_profile["position_scoring_hash"],
            matched_hash=str(row["position_scoring_hash"]),
            match_quality="Exact",
            exact_match=True,
            distance=0.0,
            status=str(row["status"]),
            target_scoring=target_scoring,
            matched_scoring=matched_scoring,
            differing_fields=[],
            seasons=[],
            max_reliable_rank=int(row.get("max_reliable_rank", 0) or 0),
        )

    candidates: list[tuple[float, pd.Series, dict[str, float]]] = []
    for _, row in position_envs.iterrows():
        matched_scoring = json.loads(str(row["normalized_scoring_json"]))
        distance = score_position_scoring_distance(position, target_scoring, matched_scoring)
        candidates.append((distance, row, matched_scoring))
    candidates.sort(key=lambda item: (item[0], -int(item[1].get("season_count", 0)), str(item[1]["position_scoring_hash"])))
    distance, row, matched_scoring = candidates[0]
    quality = classify_match_quality(position, exact_match=False, distance=distance)
    return PositionEnvironmentMatch(
        position=position,
        target_hash=target_profile["position_scoring_hash"],
        matched_hash=str(row["position_scoring_hash"]),
        match_quality=quality,
        exact_match=False,
        distance=distance,
        status=str(row["status"]),
        target_scoring=target_scoring,
        matched_scoring=matched_scoring,
        differing_fields=position_scoring_differences(target_scoring, matched_scoring),
        seasons=[],
        max_reliable_rank=int(row.get("max_reliable_rank", 0) or 0),
    )


def _build_composite_replacement(
    matched_fitted_curves: pd.DataFrame,
    matched_environments: pd.DataFrame,
    league: LeagueSettings,
) -> dict[str, pd.DataFrame]:
    starter_demand = calculate_starter_demand_replacement(league, matched_fitted_curves)
    historical = matched_environments[["position", "historical_replacement_rank"]].copy()
    historical = historical.rename(columns={"historical_replacement_rank": "replacement_rank"})
    historical["replacement_rank"] = historical["replacement_rank"].fillna(1).astype(int)
    historical["method"] = "Historical Position Replacement"
    historical["replacement_ppg"] = 0.0
    historical = attach_replacement_ppg(historical, matched_fitted_curves)
    return {
        "starter_demand": starter_demand,
        "historical_roster": historical[["position", "method", "replacement_rank", "replacement_ppg"]],
    }


def build_league_environment_from_library(
    league: LeagueSettings,
    library_bundle: dict[str, Any],
    *,
    replacement_method: str = "starter_demand",
) -> dict[str, Any]:
    """Build a target league environment by matching each position independently."""

    environments = library_bundle["position_scoring_environments"]
    fitted_curves = library_bundle["fitted_curves"]
    curve_models = library_bundle.get("curve_models", pd.DataFrame())
    season_frame = library_bundle.get("environment_seasons", pd.DataFrame())
    if environments.empty or fitted_curves.empty:
        raise ConfigError("The position-specific history library is empty. Build a candidate model before public analysis.")

    match_rows: list[dict[str, Any]] = []
    matched_environment_rows: list[dict[str, Any]] = []
    selected_curve_rows: list[pd.DataFrame] = []
    evaluated_curve_rows: list[pd.DataFrame] = []
    candidate_curve_rows: list[pd.DataFrame] = []
    coverage_rows: list[dict[str, Any]] = []

    for position in CORE_POSITIONS:
        match = match_position_scoring_environment(position, league.scoring_settings, environments)
        if not match.matched_hash or match.match_quality == "Insufficient":
            raise ConfigError(f"No acceptable {position} historical scoring environment is available for this league.")
        matched_environment = environments[
            (environments["position"] == position) & (environments["position_scoring_hash"] == match.matched_hash)
        ].iloc[0]
        matched_environment_rows.append(matched_environment.to_dict())
        match_rows.append(match.to_dict())

        fitted = fitted_curves[
            (fitted_curves["position"] == position)
            & (fitted_curves["position_scoring_hash"] == match.matched_hash)
            & (fitted_curves["dataset"] == "fitted")
        ].copy()
        empirical = fitted_curves[
            (fitted_curves["position"] == position)
            & (fitted_curves["position_scoring_hash"] == match.matched_hash)
            & (fitted_curves["dataset"] == "empirical")
        ].copy()
        selected_curve_rows.append(
            curve_models[
                (curve_models["position"] == position)
                & (curve_models["position_scoring_hash"] == match.matched_hash)
                & (curve_models["selected"] == True)
            ].copy()
        )
        candidate_curve_rows.append(
            curve_models[
                (curve_models["position"] == position)
                & (curve_models["position_scoring_hash"] == match.matched_hash)
            ].copy()
        )
        evaluated_curve_rows.append(pd.concat([fitted, empirical], ignore_index=True))

        matched_seasons = season_frame[
            (season_frame["position"] == position)
            & (season_frame["position_scoring_hash"] == match.matched_hash)
        ].copy()
        for season_row in matched_seasons.itertuples(index=False):
            coverage_rows.append(
                {
                    "position": position,
                    "season": int(season_row.season),
                    "match_quality": match.match_quality,
                    "weeks_loaded": 0,
                    "unique_player_weeks": int(season_row.player_week_count),
                    "unique_players": int(season_row.unique_player_count),
                    "max_rank": int(season_row.max_rank),
                }
            )

    selected_curves = pd.concat(selected_curve_rows, ignore_index=True) if selected_curve_rows else pd.DataFrame()
    candidate_curves = pd.concat(candidate_curve_rows, ignore_index=True) if candidate_curve_rows else pd.DataFrame()
    evaluated_curves = pd.concat(evaluated_curve_rows, ignore_index=True).reset_index(drop=True)
    matched_env_frame = pd.DataFrame(matched_environment_rows)
    fitted_only = evaluated_curves[evaluated_curves["dataset"] == "fitted"].copy()
    replacement_variants = _build_composite_replacement(fitted_only, matched_env_frame, league)

    from src.vorp import build_vorp_table

    vorp_variants = {
        name: build_vorp_table(fitted_only, replacement_frame)
        for name, replacement_frame in replacement_variants.items()
    }
    active_replacement = replacement_variants.get(replacement_method, replacement_variants["starter_demand"])

    return {
        "league": league,
        "historical_leagues": [
            HistoricalLeagueSummary(
                league_id=f"{row['position']}_{row['position_scoring_hash']}",
                season=int(season),
                scoring_settings={},
                roster_positions=[],
                total_rosters=0,
            )
            for season in sorted({int(item["season"]) for item in coverage_rows})
            for row in matched_environment_rows[:1]
        ],
        "coverage": coverage_rows,
        "coverage_frame": pd.DataFrame(coverage_rows),
        "player_weeks": pd.DataFrame(),
        "season_player_ppg": pd.DataFrame(),
        "empirical_curve": evaluated_curves[evaluated_curves["dataset"] == "empirical"].copy(),
        "candidate_curves": candidate_curves,
        "selected_curves": selected_curves,
        "evaluated_curves": evaluated_curves.drop(columns=["position_scoring_hash"], errors="ignore"),
        "replacement_variants": replacement_variants,
        "vorp_variants": vorp_variants,
        "active_replacement_method": replacement_method,
        "replacement": active_replacement,
        "vorp_table": vorp_variants[replacement_method],
        "historical_source": "position_history_library",
        "public_runtime_mode": "position_history_library",
        "position_match_summary": pd.DataFrame(match_rows),
        "matched_position_environments": matched_env_frame,
    }
