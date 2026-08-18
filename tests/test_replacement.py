from __future__ import annotations

import pandas as pd

from src.replacement import calculate_starter_demand_replacement
from src.sleeper import parse_league_settings


def _curve_frame() -> pd.DataFrame:
    rows = []
    for position, values in {
        "QB": [26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9],
        "RB": [20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7],
        "WR": [19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6],
        "TE": [14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    }.items():
        for rank, value in enumerate(values, start=1):
            rows.append({"position": position, "rank": rank, "expected_ppg": value})
    return pd.DataFrame(rows)


def test_basic_1qb_league_produces_sensible_qb_demand() -> None:
    league = parse_league_settings(
        {
            "league_id": "1",
            "name": "1QB",
            "season": "2026",
            "total_rosters": 12,
            "scoring_settings": {"rec": 1, "pass_td": 4},
            "roster_positions": ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "BN"],
        }
    )
    replacement = calculate_starter_demand_replacement(league, _curve_frame()).set_index("position")
    assert int(replacement.loc["QB", "replacement_rank"]) == 13


def test_superflex_materially_increases_qb_demand() -> None:
    one_qb = parse_league_settings(
        {
            "league_id": "1",
            "name": "1QB",
            "season": "2026",
            "total_rosters": 12,
            "scoring_settings": {"rec": 1, "pass_td": 4},
            "roster_positions": ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "BN"],
        }
    )
    superflex = parse_league_settings(
        {
            "league_id": "2",
            "name": "SF",
            "season": "2026",
            "total_rosters": 12,
            "scoring_settings": {"rec": 1, "pass_td": 4},
            "roster_positions": ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "SUPER_FLEX", "BN"],
        }
    )
    baseline = calculate_starter_demand_replacement(one_qb, _curve_frame()).set_index("position")
    sf = calculate_starter_demand_replacement(superflex, _curve_frame()).set_index("position")
    assert int(sf.loc["QB", "replacement_rank"]) > int(baseline.loc["QB", "replacement_rank"])


def test_extra_wr_starters_increase_wr_demand() -> None:
    base = parse_league_settings(
        {
            "league_id": "1",
            "name": "Base",
            "season": "2026",
            "total_rosters": 12,
            "scoring_settings": {"rec": 1, "pass_td": 4},
            "roster_positions": ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "BN"],
        }
    )
    extra_wr = parse_league_settings(
        {
            "league_id": "2",
            "name": "3WR",
            "season": "2026",
            "total_rosters": 12,
            "scoring_settings": {"rec": 1, "pass_td": 4},
            "roster_positions": ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "BN"],
        }
    )
    base_replacement = calculate_starter_demand_replacement(base, _curve_frame()).set_index("position")
    wr_replacement = calculate_starter_demand_replacement(extra_wr, _curve_frame()).set_index("position")
    assert int(wr_replacement.loc["WR", "replacement_rank"]) > int(base_replacement.loc["WR", "replacement_rank"])
