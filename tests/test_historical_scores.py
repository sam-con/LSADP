from __future__ import annotations

import pytest

from src.historical_scores import parse_matchup_players_points, summarize_historical_coverage, validate_historical_coverage
from src.models import CoverageError, HistoricalDataError


def test_matchup_parser_extracts_players_points() -> None:
    matchups = [
        {
            "roster_id": 1,
            "starters": ["Q1"],
            "players_points": {"Q1": 25.2, "R1": 18.4},
        }
    ]

    frame = parse_matchup_players_points(matchups, season=2025, week=1, league_id="abc")

    assert set(frame["player_id"]) == {"Q1", "R1"}
    assert float(frame.loc[frame["player_id"] == "Q1", "fantasy_points"].iloc[0]) == 25.2


def test_absence_of_players_points_produces_controlled_failure() -> None:
    with pytest.raises(HistoricalDataError):
        parse_matchup_players_points([{"roster_id": 1, "starters": []}], season=2025, week=1, league_id="abc")


def test_duplicate_player_appearances_are_deduplicated() -> None:
    matchups = [
        {"roster_id": 1, "starters": ["Q1"], "players_points": {"Q1": 21.0}},
        {"roster_id": 2, "starters": [], "players_points": {"Q1": 22.5}},
    ]

    frame = parse_matchup_players_points(matchups, season=2025, week=1, league_id="abc")

    assert len(frame) == 1
    assert float(frame.iloc[0]["fantasy_points"]) == 22.5
    assert bool(frame.iloc[0]["starter_flag"]) is True


def test_insufficient_player_score_coverage_fails() -> None:
    summary = summarize_historical_coverage(
        player_weeks=__import__("pandas").DataFrame(
            [
                {"season": 2025, "week": 1, "player_id": "Q1", "player_name": "QB", "position": "QB", "fantasy_points": 20.0}
            ]
        ),
        season=2025,
        weeks_loaded=2,
        min_games=1,
    )

    with pytest.raises(CoverageError):
        validate_historical_coverage([summary], min_weeks=10, min_players_by_position={"QB": 8})

