"""Automatic donor-league discovery, validation, persistence, and aggregation."""

from __future__ import annotations

import json
from collections import deque
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from src.analysis import default_modeling_config
from src.canonical import detect_reception_format
from src.config import (
    HISTORICAL_DONOR_FILE,
    HISTORICAL_DONOR_METADATA_FILE,
    SCORING_FORMATS,
)
from src.historical_scores import (
    enrich_player_weeks_with_metadata,
    load_historical_player_weeks,
    summarize_historical_coverage,
    validate_historical_coverage,
)
from src.models import ConfigError, HistoricalLeagueSummary, HistoricalDataError, LeagueSettings, ModelingConfig
from src.sleeper import SleeperClient, as_historical_summary
from src.utils import normalize_scoring_settings, required_completed_seasons

OFFENSIVE_SCORING_KEYS = (
    "pass_yd",
    "pass_td",
    "pass_2pt",
    "pass_int",
    "rush_yd",
    "rush_td",
    "rush_2pt",
    "rec",
    "rec_yd",
    "rec_td",
    "rec_2pt",
    "fum_lost",
)

FLAGGED_SCORING_KEYS = (
    "rec_te",
    "bonus_rec_te",
    "first_down",
    "first_down_pass",
    "first_down_rush",
    "first_down_rec",
    "pass_cmp",
    "pass_icmp",
    "pass_att",
    "bonus_pass_yd_300",
    "bonus_pass_yd_400",
    "bonus_rush_yd_100",
    "bonus_rush_yd_200",
    "bonus_rec_yd_100",
    "bonus_rec_yd_200",
)

DONOR_RESULT_COLUMNS = [
    "league_id",
    "season",
    "scoring_format",
    "status",
    "reason",
    "quality_score",
    "weeks_loaded",
    "unique_player_weeks",
    "unique_players",
    "qb_coverage",
    "rb_coverage",
    "wr_coverage",
    "te_coverage",
    "team_count",
    "league_name",
    "scoring_signature",
    "selected",
    "discovered_from_user_id",
]


def canonical_scoring_signatures(canonical_leagues: dict[str, str], client: SleeperClient) -> dict[str, dict[str, float]]:
    """Build one explicit scoring signature per reception environment."""

    signatures: dict[str, dict[str, float]] = {}
    for league_id in canonical_leagues.values():
        league = client.get_league(league_id)
        scoring_format = detect_reception_format(league)
        signatures.setdefault(scoring_format, scoring_signature(league.scoring_settings))
    missing = [scoring_format for scoring_format in SCORING_FORMATS if scoring_format not in signatures]
    if missing:
        raise ConfigError(f"Canonical leagues do not provide signatures for: {', '.join(missing)}")
    return signatures


def scoring_signature(scoring_settings: dict[str, float]) -> dict[str, float]:
    """Normalize the offensive scoring rules that define a donor environment."""

    normalized = normalize_scoring_settings(scoring_settings)
    return {key: float(normalized.get(key, 0.0)) for key in OFFENSIVE_SCORING_KEYS}


def scoring_signature_text(signature: dict[str, float]) -> str:
    """Serialize a scoring signature deterministically for storage and UI."""

    return json.dumps(signature, sort_keys=True, separators=(",", ":"))


def donor_scoring_mismatches(
    league: LeagueSettings,
    expected_signature: dict[str, float],
) -> tuple[list[str], list[str]]:
    """Return exact signature mismatches plus unusual scoring flags."""

    league_signature = scoring_signature(league.scoring_settings)
    mismatches = [
        key
        for key in OFFENSIVE_SCORING_KEYS
        if abs(float(league_signature.get(key, 0.0)) - float(expected_signature.get(key, 0.0))) > 1e-9
    ]
    normalized = normalize_scoring_settings(league.scoring_settings)
    flagged = [key for key in FLAGGED_SCORING_KEYS if abs(float(normalized.get(key, 0.0))) > 1e-9]
    return mismatches, flagged


def donor_quality_score(coverage, flagged_keys: list[str]) -> float:
    """Rank accepted donor leagues by coverage depth and cleanliness."""

    depth_score = (
        0.2 * float(coverage.deepest_rank_by_position.get("QB", 0))
        + 0.3 * float(coverage.deepest_rank_by_position.get("RB", 0))
        + 0.3 * float(coverage.deepest_rank_by_position.get("WR", 0))
        + 0.2 * float(coverage.deepest_rank_by_position.get("TE", 0))
    )
    completeness_score = 0.05 * float(coverage.unique_player_weeks) + 0.1 * float(coverage.weeks_loaded)
    penalty = 25.0 * float(len(flagged_keys))
    return round(depth_score + completeness_score - penalty, 3)


def validate_donor_league(
    client: SleeperClient,
    league_id: str,
    season: int,
    scoring_format: str,
    expected_signature: dict[str, float],
    modeling_config: ModelingConfig | None = None,
) -> dict[str, Any]:
    """Validate one league as a donor candidate for a season/scoring cell."""

    modeling_config = modeling_config or default_modeling_config()
    league = client.get_league(league_id)
    if int(league.season) != int(season):
        return {"league_id": league_id, "season": season, "scoring_format": scoring_format, "status": "rejected", "reason": "Wrong season"}

    mismatches, flagged_keys = donor_scoring_mismatches(league, expected_signature)
    if mismatches:
        return {
            "league_id": league_id,
            "season": season,
            "scoring_format": scoring_format,
            "status": "rejected",
            "reason": f"Scoring mismatch: {', '.join(mismatches)}",
            "scoring_signature": scoring_signature_text(scoring_signature(league.scoring_settings)),
        }
    if flagged_keys:
        return {
            "league_id": league_id,
            "season": season,
            "scoring_format": scoring_format,
            "status": "rejected",
            "reason": f"Unusual scoring: {', '.join(flagged_keys)}",
            "scoring_signature": scoring_signature_text(scoring_signature(league.scoring_settings)),
        }

    historical_league = as_historical_summary(league)
    try:
        player_weeks, weeks_loaded = load_historical_player_weeks(client, historical_league)
        enriched = enrich_player_weeks_with_metadata(player_weeks, client.get_players("nfl"))
        coverage = summarize_historical_coverage(
            enriched,
            season=int(league.season),
            weeks_loaded=weeks_loaded,
            min_games=modeling_config.min_games,
        )
        validate_historical_coverage(
            [coverage],
            min_weeks=modeling_config.min_coverage_weeks,
            min_player_weeks_by_position=modeling_config.min_player_weeks_by_position,
            min_players_by_position=modeling_config.min_players_by_position,
        )
    except HistoricalDataError as exc:
        return {
            "league_id": league_id,
            "season": season,
            "scoring_format": scoring_format,
            "status": "rejected",
            "reason": str(exc),
            "scoring_signature": scoring_signature_text(scoring_signature(league.scoring_settings)),
        }

    quality = donor_quality_score(coverage, flagged_keys)
    return {
        "league_id": league_id,
        "season": season,
        "scoring_format": scoring_format,
        "status": "accepted",
        "reason": "Good coverage",
        "quality_score": quality,
        "weeks_loaded": int(coverage.weeks_loaded),
        "unique_player_weeks": int(coverage.unique_player_weeks),
        "unique_players": int(coverage.unique_players),
        "qb_coverage": int(coverage.deepest_rank_by_position.get("QB", 0)),
        "rb_coverage": int(coverage.deepest_rank_by_position.get("RB", 0)),
        "wr_coverage": int(coverage.deepest_rank_by_position.get("WR", 0)),
        "te_coverage": int(coverage.deepest_rank_by_position.get("TE", 0)),
        "team_count": int(league.total_rosters),
        "league_name": league.name,
        "scoring_signature": scoring_signature_text(scoring_signature(league.scoring_settings)),
    }


def donor_matrix_summary(donors: pd.DataFrame, required_seasons: list[int], preferred_per_cell: int) -> pd.DataFrame:
    """Build a 4 x 3 matrix summary for the donor UI."""

    rows: list[dict[str, Any]] = []
    for season in required_seasons:
        row: dict[str, Any] = {"season": season}
        for scoring_format in SCORING_FORMATS:
            count = 0
            if not donors.empty:
                count = int(
                    donors[
                        (donors["selected"])
                        & (donors["season"] == season)
                        & (donors["scoring_format"] == scoring_format)
                    ]["league_id"].nunique()
                )
            row[scoring_format] = f"{count} donors {'✓' if count >= preferred_per_cell else '…'}"
        rows.append(row)
    return pd.DataFrame(rows)


def donors_complete(donors: pd.DataFrame, required_seasons: list[int], preferred_per_cell: int) -> bool:
    """Return True when every season/scoring cell has enough selected donors."""

    if donors.empty:
        return False
    for season in required_seasons:
        for scoring_format in SCORING_FORMATS:
            count = int(
                donors[
                    (donors["selected"])
                    & (donors["season"] == season)
                    & (donors["scoring_format"] == scoring_format)
                ]["league_id"].nunique()
            )
            if count < preferred_per_cell:
                return False
    return True


def discover_historical_donors(
    client: SleeperClient,
    canonical_leagues: dict[str, str],
    seed_user: str,
    *,
    max_users: int = 200,
    max_leagues: int = 1000,
    preferred_donors_per_cell: int = 3,
    max_depth: int = 3,
    today: date | None = None,
    modeling_config: ModelingConfig | None = None,
) -> dict[str, Any]:
    """Breadth-first crawl the Sleeper user graph and collect validated donors."""

    today = today or date.today()
    modeling_config = modeling_config or default_modeling_config()
    required_seasons = required_completed_seasons(today=today, window=4)
    signatures = canonical_scoring_signatures(canonical_leagues, client)

    players_payload = client.get_players("nfl")
    user_cache: dict[str, dict[str, Any]] = {}
    user_leagues_cache: dict[tuple[str, int], list[LeagueSettings]] = {}
    league_users_cache: dict[str, list[dict[str, Any]]] = {}
    league_cache: dict[str, LeagueSettings] = {}

    seed_payload = client.get_user(seed_user)
    seed_user_id = str(seed_payload["user_id"])

    queue: deque[tuple[str, int]] = deque([(seed_user_id, 0)])
    seen_users: set[str] = set()
    seen_leagues: set[str] = set()
    accepted_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    progress: list[dict[str, Any]] = []

    while queue and len(seen_users) < max_users and len(seen_leagues) < max_leagues:
        user_id, depth = queue.popleft()
        if user_id in seen_users or depth > max_depth:
            continue
        seen_users.add(user_id)
        progress.append({"event": "visit_user", "user_id": user_id, "depth": depth})

        for season in [today.year, *required_seasons]:
            cache_key = (user_id, int(season))
            if cache_key not in user_leagues_cache:
                user_leagues_cache[cache_key] = client.get_user_leagues(user_id=user_id, season=season)
            leagues = user_leagues_cache[cache_key]
            for league in leagues:
                if league.league_id in seen_leagues:
                    continue
                if len(seen_leagues) >= max_leagues:
                    break
                seen_leagues.add(league.league_id)
                league_cache[league.league_id] = league
                progress.append({"event": "visit_league", "league_id": league.league_id, "season": league.season})

                if int(league.season) in required_seasons:
                    scoring_format = detect_reception_format(league)
                    if scoring_format in signatures:
                        validation = validate_donor_league(
                            client,
                            league_id=league.league_id,
                            season=int(league.season),
                            scoring_format=scoring_format,
                            expected_signature=signatures[scoring_format],
                            modeling_config=modeling_config,
                        )
                        validation["discovered_from_user_id"] = user_id
                        if validation["status"] == "accepted":
                            validation["selected"] = False
                            accepted_rows.append(validation)
                        else:
                            rejected_rows.append(validation)

                if depth >= max_depth:
                    continue
                if league.league_id not in league_users_cache:
                    league_users_cache[league.league_id] = client.get_league_users(league.league_id)
                for user_payload in league_users_cache[league.league_id]:
                    next_user_id = str(user_payload.get("user_id") or "")
                    if next_user_id and next_user_id not in seen_users:
                        queue.append((next_user_id, depth + 1))

        selected = select_top_donors(pd.DataFrame(accepted_rows), required_seasons, preferred_donors_per_cell)
        if donors_complete(selected, required_seasons, preferred_donors_per_cell):
            accepted_rows = selected.to_dict(orient="records")
            break

    accepted = pd.DataFrame(accepted_rows, columns=DONOR_RESULT_COLUMNS)
    selected = select_top_donors(accepted, required_seasons, preferred_donors_per_cell)
    rejected = pd.DataFrame(rejected_rows)
    matrix = donor_matrix_summary(selected, required_seasons, preferred_donors_per_cell)
    return {
        "seed_user_id": seed_user_id,
        "required_seasons": required_seasons,
        "preferred_donors_per_cell": preferred_donors_per_cell,
        "accepted": selected,
        "rejected": rejected,
        "matrix": matrix,
        "progress": pd.DataFrame(progress),
        "signatures": {key: scoring_signature_text(value) for key, value in signatures.items()},
        "crawl_stats": {
            "users_inspected": len(seen_users),
            "leagues_inspected": len(seen_leagues),
            "max_users": max_users,
            "max_leagues": max_leagues,
            "max_depth": max_depth,
        },
    }


def select_top_donors(
    donors: pd.DataFrame,
    required_seasons: list[int],
    preferred_donors_per_cell: int,
) -> pd.DataFrame:
    """Select the best donors per season/scoring cell by quality score."""

    if donors.empty:
        empty = donors.copy()
        if "selected" not in empty.columns:
            empty["selected"] = pd.Series(dtype=bool)
        return empty
    ordered = donors.sort_values(
        ["season", "scoring_format", "quality_score", "league_id"],
        ascending=[True, True, False, True],
    ).copy()
    ordered["selected"] = False
    for season in required_seasons:
        for scoring_format in SCORING_FORMATS:
            mask = (ordered["season"] == season) & (ordered["scoring_format"] == scoring_format)
            top_index = ordered[mask].head(preferred_donors_per_cell).index
            ordered.loc[top_index, "selected"] = True
    return ordered.reset_index(drop=True)


def save_donor_configuration(
    donors: pd.DataFrame,
    *,
    donor_path: Path = HISTORICAL_DONOR_FILE,
    metadata_path: Path = HISTORICAL_DONOR_METADATA_FILE,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Persist the selected donor configuration and metadata."""

    donor_path.parent.mkdir(parents=True, exist_ok=True)
    donors.to_csv(donor_path, index=False)
    metadata_payload = metadata or {}
    metadata_path.write_text(json.dumps(metadata_payload, indent=2, sort_keys=True), encoding="utf-8")


def load_saved_donor_configuration(
    donor_path: Path = HISTORICAL_DONOR_FILE,
    metadata_path: Path = HISTORICAL_DONOR_METADATA_FILE,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load persisted donors from disk."""

    if not donor_path.exists():
        raise ConfigError(
            f"Historical donor configuration is missing at {donor_path}. Discover or manually save donors from the Development page first."
        )
    donors = pd.read_csv(donor_path)
    if "selected" in donors.columns:
        donors["selected"] = donors["selected"].astype(bool)
    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return donors, metadata


def build_historical_player_weeks_from_donors(
    client: SleeperClient,
    donors: pd.DataFrame,
    *,
    modeling_config: ModelingConfig | None = None,
) -> dict[str, Any]:
    """Load, deduplicate, and validate historical player-weeks from selected donors."""

    modeling_config = modeling_config or default_modeling_config()
    selected = donors[donors["selected"]].copy()
    if selected.empty:
        raise ConfigError("No historical donors are selected.")

    players_payload = client.get_players("nfl")
    frames_by_format: dict[str, list[pd.DataFrame]] = {key: [] for key in SCORING_FORMATS}
    coverage_by_format: dict[str, list[Any]] = {key: [] for key in SCORING_FORMATS}
    league_records: list[dict[str, Any]] = []

    for donor in selected.itertuples(index=False):
        league = client.get_league(str(donor.league_id))
        historical_league = HistoricalLeagueSummary(
            league_id=str(league.league_id),
            season=int(donor.season),
            scoring_settings=league.scoring_settings,
            roster_positions=league.roster_positions,
            total_rosters=int(league.total_rosters),
            previous_league_id=league.previous_league_id,
            playoff_week_start=league.playoff_week_start,
        )
        player_weeks, weeks_loaded = load_historical_player_weeks(client, historical_league)
        enriched = enrich_player_weeks_with_metadata(player_weeks, players_payload)
        coverage = summarize_historical_coverage(
            enriched,
            season=int(donor.season),
            weeks_loaded=weeks_loaded,
            min_games=modeling_config.min_games,
        )
        coverage_by_format[str(donor.scoring_format)].append(coverage)
        frames_by_format[str(donor.scoring_format)].append(enriched)
        league_records.append(
            {
                "season": int(donor.season),
                "scoring_format": str(donor.scoring_format),
                "league_id": str(donor.league_id),
                "weeks_loaded": weeks_loaded,
                "unique_players": int(coverage.unique_players),
            }
        )

    combined_by_format: dict[str, pd.DataFrame] = {}
    conflicts: list[dict[str, Any]] = []
    for scoring_format in SCORING_FORMATS:
        frames = frames_by_format[scoring_format]
        if not frames:
            continue
        combined = pd.concat(frames, ignore_index=True)
        duplicate_groups = combined.groupby(["season", "week", "player_id"])["fantasy_points"].nunique().reset_index(name="value_count")
        bad = duplicate_groups[duplicate_groups["value_count"] > 1]
        if not bad.empty:
            conflicts.extend(bad.to_dict(orient="records"))
            continue
        combined = combined.sort_values(["season", "week", "player_id", "league_id"]).drop_duplicates(
            subset=["season", "week", "player_id"],
            keep="first",
        )
        validate_historical_coverage(
            coverage_by_format[scoring_format],
            min_weeks=modeling_config.min_coverage_weeks,
            min_player_weeks_by_position=modeling_config.min_player_weeks_by_position,
            min_players_by_position=modeling_config.min_players_by_position,
        )
        combined_by_format[scoring_format] = combined.reset_index(drop=True)

    if conflicts:
        raise ConfigError(
            "Historical donor leagues disagree on duplicated player-week fantasy points. "
            f"Conflicts: {conflicts[:5]}"
        )

    missing_formats = [scoring_format for scoring_format in SCORING_FORMATS if scoring_format not in combined_by_format]
    if missing_formats:
        raise ConfigError(f"Historical donors are incomplete for: {', '.join(missing_formats)}")

    return {
        "player_weeks_by_format": combined_by_format,
        "coverage_by_format": coverage_by_format,
        "league_records": pd.DataFrame(league_records),
    }
