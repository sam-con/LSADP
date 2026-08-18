from __future__ import annotations

import math

import pandas as pd

from src.transform import apply_league_transformation
from src.vorp import build_vorp_table


def test_player_at_replacement_has_near_zero_vorp() -> None:
    curves = pd.DataFrame(
        [
            {"position": "QB", "rank": 1, "expected_ppg": 24.0},
            {"position": "QB", "rank": 2, "expected_ppg": 22.0},
            {"position": "QB", "rank": 3, "expected_ppg": 20.0},
        ]
    )
    replacement = pd.DataFrame([{"position": "QB", "replacement_rank": 3, "replacement_ppg": 20.0, "method": "test"}])
    vorp = build_vorp_table(curves, replacement)
    assert abs(float(vorp.loc[vorp["rank"] == 3, "vorp"].iloc[0])) < 1e-9


def test_player_above_replacement_has_positive_vorp() -> None:
    curves = pd.DataFrame(
        [
            {"position": "QB", "rank": 1, "expected_ppg": 24.0},
            {"position": "QB", "rank": 2, "expected_ppg": 22.0},
            {"position": "QB", "rank": 3, "expected_ppg": 20.0},
        ]
    )
    replacement = pd.DataFrame([{"position": "QB", "replacement_rank": 3, "replacement_ppg": 20.0, "method": "test"}])
    vorp = build_vorp_table(curves, replacement)
    assert float(vorp.loc[vorp["rank"] == 1, "vorp"].iloc[0]) > 0


def test_identical_baseline_and_target_leagues_produce_zero_delta_vorp() -> None:
    frame = pd.DataFrame(
        [
            {
                "player_name": "QB Player 1",
                "position": "QB",
                    "adp": 5.0,
                    "pos_rank": 1,
                    "baseline_vorp": 5.0,
                    "league_vorp": 5.0,
                    "utility": -math.log(5.0),
                    "market_coefficient": 0.8,
                }
            ]
    )
    transformed = apply_league_transformation(frame, "baseline_vorp", "league_vorp")
    assert abs(float(transformed.iloc[0]["delta_metric"])) < 1e-9
    assert abs(float(transformed.iloc[0]["league_adjusted_adp"]) - 5.0) < 1e-9
