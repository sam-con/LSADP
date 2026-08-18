from __future__ import annotations

import pytest

from src.models import SleeperAPIError
from src.sleeper import parse_league_settings


def test_parse_league_settings_extracts_core_fields() -> None:
    payload = {
        "league_id": "123",
        "name": "Home League",
        "season": "2026",
        "total_rosters": 12,
        "scoring_settings": {"pass_td": 4, "rec": 1},
        "roster_positions": ["QB", "RB", "WR", "TE", "FLEX", "BN"],
        "previous_league_id": "122",
        "settings": {"playoff_week_start": 15},
    }

    league = parse_league_settings(payload)

    assert league.league_id == "123"
    assert league.season == 2026
    assert league.previous_league_id == "122"
    assert league.playoff_week_start == 15
    assert league.scoring_settings == {"pass_td": 4.0, "rec": 1.0}


def test_parse_league_settings_rejects_missing_fields() -> None:
    with pytest.raises(SleeperAPIError):
        parse_league_settings({"league_id": "123"})

