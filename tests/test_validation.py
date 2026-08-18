from __future__ import annotations

import pandas as pd

from src.validation import positional_error_breakdown, score_prediction


def test_validation_metrics_are_computed() -> None:
    predicted = pd.DataFrame(
        [
            {"player_name": "A", "position": "QB", "league_adjusted_adp": 10.0},
            {"player_name": "B", "position": "RB", "league_adjusted_adp": 20.0},
            {"player_name": "C", "position": "WR", "league_adjusted_adp": 30.0},
        ]
    )
    actual = pd.DataFrame(
        [
            {"player_name": "A", "position": "QB", "adp": 12.0},
            {"player_name": "B", "position": "RB", "adp": 18.0},
            {"player_name": "C", "position": "WR", "adp": 33.0},
        ]
    )

    metrics = score_prediction(predicted, actual, model_name="Test")
    breakdown = positional_error_breakdown(predicted, actual)

    assert metrics["mae"] > 0
    assert "spearman" in metrics
    assert list(breakdown["position"]) == ["QB", "RB", "WR"]

