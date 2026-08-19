from __future__ import annotations

import pandas as pd

from src.charts import build_validation_scatter


def test_validation_scatter_handles_predicted_and_actual_adp_columns() -> None:
    predicted = pd.DataFrame(
        [
            {"player_name": "A", "position": "QB", "adp": 9.0, "league_adjusted_adp": 10.0},
            {"player_name": "B", "position": "RB", "adp": 19.0, "league_adjusted_adp": 20.0},
        ]
    )
    actual = pd.DataFrame(
        [
            {"player_name": "A", "position": "QB", "adp": 12.0},
            {"player_name": "B", "position": "RB", "adp": 18.0},
        ]
    )

    figure = build_validation_scatter(predicted, actual, "Validation")

    assert figure.data
    y_values = []
    for trace in figure.data:
        if hasattr(trace, "y"):
            y_values.extend(list(trace.y))
    assert sorted(y_values) == [12.0, 18.0]
