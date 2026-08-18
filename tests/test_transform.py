from __future__ import annotations

import numpy as np
import pandas as pd

from src.transform import apply_league_transformation


def _calibrated_players() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"player_name": "QB1", "position": "QB", "adp": 30.0, "pos_rank": 1, "baseline_vorp": 5.0, "league_vorp": 8.0, "baseline_expected_ppg": 23.0, "league_expected_ppg": 24.5, "utility": -np.log(30.0), "market_coefficient": 0.12},
            {"player_name": "QB2", "position": "QB", "adp": 42.0, "pos_rank": 2, "baseline_vorp": 4.0, "league_vorp": 6.5, "baseline_expected_ppg": 21.0, "league_expected_ppg": 22.0, "utility": -np.log(42.0), "market_coefficient": 0.12},
            {"player_name": "RB1", "position": "RB", "adp": 6.0, "pos_rank": 1, "baseline_vorp": 7.0, "league_vorp": 6.8, "baseline_expected_ppg": 20.0, "league_expected_ppg": 19.8, "utility": -np.log(6.0), "market_coefficient": 0.10},
            {"player_name": "WR1", "position": "WR", "adp": 9.0, "pos_rank": 1, "baseline_vorp": 6.5, "league_vorp": 6.3, "baseline_expected_ppg": 19.0, "league_expected_ppg": 18.9, "utility": -np.log(9.0), "market_coefficient": 0.10},
            {"player_name": "TE1", "position": "TE", "adp": 28.0, "pos_rank": 1, "baseline_vorp": 4.5, "league_vorp": 6.9, "baseline_expected_ppg": 13.5, "league_expected_ppg": 16.0, "utility": -np.log(28.0), "market_coefficient": 0.11},
        ]
    )


def test_identical_league_settings_produce_unchanged_adp() -> None:
    frame = _calibrated_players()
    frame["league_vorp"] = frame["baseline_vorp"]
    transformed = apply_league_transformation(frame, "baseline_vorp", "league_vorp")
    assert np.allclose(transformed["league_adjusted_adp"], transformed["adp"])


def test_adding_superflex_moves_qbs_upward_on_average() -> None:
    transformed = apply_league_transformation(_calibrated_players(), "baseline_vorp", "league_vorp")
    qb_change = transformed[transformed["position"] == "QB"]["adp_change"].mean()
    rb_change = transformed[transformed["position"] == "RB"]["adp_change"].mean()
    assert qb_change > rb_change


def test_changing_only_te_scoring_primarily_affects_tes() -> None:
    frame = _calibrated_players()
    frame["league_vorp"] = frame["baseline_vorp"]
    frame.loc[frame["position"] == "TE", "league_vorp"] += 2.5
    transformed = apply_league_transformation(frame, "baseline_vorp", "league_vorp")
    te_move = transformed.loc[transformed["position"] == "TE", "adp_change"].abs().mean()
    non_te_move = transformed.loc[transformed["position"] != "TE", "adp_change"].abs().mean()
    assert te_move > non_te_move


def test_output_ordering_is_unique_and_stable() -> None:
    transformed = apply_league_transformation(_calibrated_players(), "baseline_vorp", "league_vorp")
    assert transformed["adjusted_rank"].is_unique
    assert transformed["league_adjusted_adp"].replace([np.inf, -np.inf], np.nan).notna().all()

