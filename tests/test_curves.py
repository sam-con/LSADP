from __future__ import annotations

import numpy as np
import pytest

from src.curves import aggregate_rank_curves, build_season_player_ppg, fit_curve_for_position, select_curve_models
from src.models import CurveFitError


def test_fitted_curve_is_monotonic_decreasing(player_weeks_df) -> None:
    season_player_ppg = build_season_player_ppg(player_weeks_df, min_games=4)
    aggregated = aggregate_rank_curves(season_player_ppg)
    _, selected = select_curve_models(season_player_ppg, aggregated, historical_window="2022-2025")

    qb = selected[selected["position"] == "QB"].iloc[0]
    values = [qb["a"] * np.exp(-qb["c"] * (rank - 1)) + qb["k"] for rank in range(1, 9)]
    assert all(left >= right for left, right in zip(values, values[1:]))
    assert qb["a"] > 0
    assert qb["c"] > 0
    assert qb["k"] >= 0


def test_curve_fitting_handles_missing_ranks(player_weeks_df) -> None:
    trimmed = player_weeks_df[~((player_weeks_df["position"] == "TE") & (player_weeks_df["player_id"] == "T8"))]
    season_player_ppg = build_season_player_ppg(trimmed, min_games=4)
    aggregated = aggregate_rank_curves(season_player_ppg)
    _, selected = select_curve_models(season_player_ppg, aggregated, historical_window="2022-2025")
    assert "TE" in selected["position"].tolist()


def test_model_selection_is_deterministic(player_weeks_df) -> None:
    season_player_ppg = build_season_player_ppg(player_weeks_df, min_games=4)
    aggregated = aggregate_rank_curves(season_player_ppg)
    _, first = select_curve_models(season_player_ppg, aggregated, historical_window="2022-2025")
    _, second = select_curve_models(season_player_ppg, aggregated, historical_window="2022-2025")
    assert first.equals(second)


def test_floor_curve_failure_falls_back_to_exp(player_weeks_df, monkeypatch: pytest.MonkeyPatch) -> None:
    season_player_ppg = build_season_player_ppg(player_weeks_df, min_games=4)
    aggregated = aggregate_rank_curves(season_player_ppg)

    original = fit_curve_for_position

    def patched_fit_curve_for_position(empirical_curve, position, model_name, historical_window):
        if model_name == "exp_k":
            raise CurveFitError("Curve fit failed for model exp_k")
        return original(empirical_curve, position, model_name, historical_window)

    monkeypatch.setattr("src.curves.fit_curve_for_position", patched_fit_curve_for_position)

    candidates, selected = select_curve_models(season_player_ppg, aggregated, historical_window="2022-2025")

    assert set(selected["model_name"]) == {"exp"}
    assert set(candidates["model_name"]) == {"exp"}
