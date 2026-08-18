"""Historical fantasy-score extraction using Sleeper `players_points`."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import pandas as pd

from src.config import CORE_POSITIONS, DEFAULT_COVERAGE_MIN_WEEKS, REGULAR_SEASON_WEEKS
from src.models import CoverageError, HistoricalCoverage, HistoricalDataError, HistoricalLeagueSummary
from src.sleeper import SleeperClient
from src.utils import coerce_position


def parse_matchup_players_points(
    matchups: list[dict[str, Any]],
    season: int,
    week: int,
    league_id: str,
) -> pd.DataFrame:
    """Parse player-week fantasy points from a Sleeper matchup response."""

    if not isinstance(matchups, list):
        raise HistoricalDataError("Sleeper matchup payload was not a list")

    records_by_player: dict[str, dict[str, Any]] = {}
    for matchup in matchups:
        players_points = matchup.get("players_points")
        if not isinstance(players_points, dict) or not players_points:
            continue
        starters = {str(player_id) for player_id in matchup.get("starters", []) if player_id}
        roster_id = matchup.get("roster_id")
        for player_id, points in players_points.items():
            try:
                fantasy_points = float(points)
            except (TypeError, ValueError):
                continue

            player_key = str(player_id)
            existing = records_by_player.get(player_key)
            payload = {
                "season": season,
                "week": week,
                "league_id": league_id,
                "player_id": player_key,
                "fantasy_points": fantasy_points,
                "starter_flag": player_key in starters,
                "roster_id": roster_id,
            }
            if existing is None:
                records_by_player[player_key] = payload
                continue
            existing["fantasy_points"] = max(existing["fantasy_points"], fantasy_points)
            existing["starter_flag"] = existing["starter_flag"] or payload["starter_flag"]
            if existing.get("roster_id") is None:
                existing["roster_id"] = roster_id

    if not records_by_player:
        raise HistoricalDataError(
            f"Sleeper historical player scoring could not be loaded for {season} week {week}: "
            "`players_points` was missing, empty, or malformed."
        )

    return pd.DataFrame(records_by_player.values())


def load_historical_player_weeks(
    client: SleeperClient,
    historical_league: HistoricalLeagueSummary,
    regular_season_weeks: tuple[int, ...] = REGULAR_SEASON_WEEKS,
) -> tuple[pd.DataFrame, int]:
    """Load all usable regular-season player-weeks for a single historical league."""

    max_week = historical_league.playoff_week_start - 1 if historical_league.playoff_week_start else max(regular_season_weeks)
    weeks_to_load = [week for week in regular_season_weeks if week <= max_week]

    frames: list[pd.DataFrame] = []
    successful_weeks = 0
    for week in weeks_to_load:
        matchups = client.get_matchups(historical_league.league_id, week)
        try:
            frame = parse_matchup_players_points(
                matchups=matchups,
                season=historical_league.season,
                week=week,
                league_id=historical_league.league_id,
            )
        except HistoricalDataError:
            continue
        frames.append(frame)
        successful_weeks += 1

    if not frames:
        raise HistoricalDataError(f"Sleeper historical player scoring could not be loaded for the {historical_league.season} season.")

    season_frame = pd.concat(frames, ignore_index=True)
    season_frame = season_frame.drop_duplicates(subset=["season", "week", "player_id"], keep="first")
    return season_frame, successful_weeks


def build_player_metadata_frame(players_payload: dict[str, Any]) -> pd.DataFrame:
    """Normalize Sleeper player metadata to a compact frame."""

    records: list[dict[str, Any]] = []
    for player_id, player in players_payload.items():
        if not isinstance(player, dict):
            continue
        position = coerce_position(player.get("position"))
        if not position:
            continue
        full_name = player.get("full_name") or " ".join(
            part for part in [player.get("first_name"), player.get("last_name")] if part
        ).strip()
        if not full_name:
            continue
        records.append(
            {
                "player_id": str(player_id),
                "player_name": full_name,
                "team": player.get("team"),
                "position": position,
            }
        )
    return pd.DataFrame(records).drop_duplicates(subset=["player_id"])


def enrich_player_weeks_with_metadata(player_weeks: pd.DataFrame, players_payload: dict[str, Any]) -> pd.DataFrame:
    """Attach position and display metadata to player-week rows."""

    metadata = build_player_metadata_frame(players_payload)
    enriched = player_weeks.merge(metadata, on="player_id", how="left")
    enriched = enriched.dropna(subset=["position"])
    return enriched


def summarize_historical_coverage(
    player_weeks: pd.DataFrame,
    season: int,
    weeks_loaded: int,
    min_games: int,
) -> HistoricalCoverage:
    """Summarize usable historical coverage by season and position."""

    unique_players = player_weeks[["player_id", "position"]].drop_duplicates()
    players_by_position = (
        unique_players.groupby("position")["player_id"]
        .nunique()
        .reindex(list(CORE_POSITIONS), fill_value=0)
        .to_dict()
    )

    season_player_ppg = (
        player_weeks.groupby(["season", "player_id", "player_name", "position"], as_index=False)
        .agg(games=("week", "nunique"), ppg=("fantasy_points", "mean"))
    )
    season_player_ppg = season_player_ppg[season_player_ppg["games"] >= min_games]
    season_player_ppg = season_player_ppg.sort_values(["position", "ppg"], ascending=[True, False]).copy()
    season_player_ppg["rank"] = season_player_ppg.groupby("position").cumcount() + 1
    deepest_rank = (
        season_player_ppg.groupby("position")["rank"]
        .max()
        .reindex(list(CORE_POSITIONS), fill_value=0)
        .astype(int)
        .to_dict()
    )

    return HistoricalCoverage(
        season=season,
        weeks_loaded=weeks_loaded,
        unique_player_weeks=int(len(player_weeks)),
        unique_players=int(unique_players["player_id"].nunique()),
        unique_players_by_position={key: int(value) for key, value in players_by_position.items()},
        deepest_rank_by_position=deepest_rank,
    )


def validate_historical_coverage(
    coverage_summaries: list[HistoricalCoverage],
    min_weeks: int = DEFAULT_COVERAGE_MIN_WEEKS,
    min_player_weeks_by_position: dict[str, int] | None = None,
    min_players_by_position: dict[str, int] | None = None,
) -> None:
    """Raise when coverage is too sparse to fit reliable curves."""

    min_player_weeks_by_position = min_player_weeks_by_position or {}
    min_players_by_position = min_players_by_position or {}

    aggregate_players_by_position: dict[str, int] = defaultdict(int)
    deepest_rank_by_position: dict[str, int] = defaultdict(int)

    for summary in coverage_summaries:
        if summary.weeks_loaded < min_weeks:
            raise CoverageError(f"Historical player coverage is insufficient: only {summary.weeks_loaded} usable weeks in {summary.season}.")
        for position, player_count in summary.unique_players_by_position.items():
            aggregate_players_by_position[position] += player_count
        for position, deepest_rank in summary.deepest_rank_by_position.items():
            deepest_rank_by_position[position] = max(deepest_rank_by_position[position], deepest_rank)

    for position, minimum_players in min_players_by_position.items():
        if aggregate_players_by_position.get(position, 0) < minimum_players:
            raise CoverageError(f"Historical player coverage is insufficient to fit a reliable {position} production curve.")
    for position, minimum_player_weeks in min_player_weeks_by_position.items():
        if deepest_rank_by_position.get(position, 0) < max(1, minimum_player_weeks // min_weeks):
            raise CoverageError(f"Historical player coverage is insufficient to fit a reliable {position} production curve.")

