"""Model validation and ablation metrics."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from src.utils import correlation_safe


def _top_overlap(predicted: pd.DataFrame, actual: pd.DataFrame, top_n: int) -> float:
    predicted_names = set(predicted.nsmallest(top_n, "league_adjusted_adp")["player_name"])
    actual_names = set(actual.nsmallest(top_n, "adp")["player_name"])
    return float(len(predicted_names & actual_names) / top_n)


def _pairwise_accuracy(merged: pd.DataFrame, prediction_column: str, actual_column: str) -> float:
    if len(merged) < 2:
        return 0.0
    correct = 0
    total = 0
    for left_index, right_index in combinations(merged.index, 2):
        left = merged.loc[left_index]
        right = merged.loc[right_index]
        predicted_order = left[prediction_column] < right[prediction_column]
        actual_order = left[actual_column] < right[actual_column]
        correct += int(predicted_order == actual_order)
        total += 1
    return float(correct / total) if total else 0.0


def score_prediction(
    predicted: pd.DataFrame,
    actual: pd.DataFrame,
    model_name: str,
    prediction_column: str = "league_adjusted_adp",
    actual_column: str = "adp",
) -> dict[str, float | str]:
    """Score a predicted ADP table against actual ADP."""

    actual_frame = actual[["player_name", "position", actual_column]].rename(columns={actual_column: "actual_adp"})
    merged = predicted.merge(
        actual_frame,
        on=["player_name", "position"],
        how="inner",
    ).copy()
    if merged.empty:
        raise ValueError("No overlapping players between predicted and actual ADP files")

    error = merged[prediction_column] - merged["actual_adp"]
    abs_error = error.abs()
    weights = 1.0 / merged["actual_adp"].clip(lower=1.0)
    actual_alias = merged[["player_name", "actual_adp"]].rename(columns={"actual_adp": "adp"})

    return {
        "model_name": model_name,
        "spearman": correlation_safe(merged[prediction_column], merged["actual_adp"], method="spearman"),
        "pearson": correlation_safe(merged[prediction_column], merged["actual_adp"], method="pearson"),
        "mae": float(abs_error.mean()),
        "median_absolute_error": float(abs_error.median()),
        "weighted_mae": float(np.average(abs_error, weights=weights)),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "pairwise_accuracy": _pairwise_accuracy(merged, prediction_column=prediction_column, actual_column="actual_adp"),
        "top_12_overlap": _top_overlap(merged, actual=actual_alias, top_n=12),
        "top_24_overlap": _top_overlap(merged, actual=actual_alias, top_n=24),
        "top_50_overlap": _top_overlap(merged, actual=actual_alias, top_n=50),
        "top_100_overlap": _top_overlap(merged, actual=actual_alias, top_n=100),
    }


def positional_error_breakdown(
    predicted: pd.DataFrame,
    actual: pd.DataFrame,
    prediction_column: str = "league_adjusted_adp",
    actual_column: str = "adp",
) -> pd.DataFrame:
    """Calculate validation errors by position."""

    actual_frame = actual[["player_name", "position", actual_column]].rename(columns={actual_column: "actual_adp"})
    merged = predicted.merge(
        actual_frame,
        on=["player_name", "position"],
        how="inner",
    )
    merged["absolute_error"] = (merged[prediction_column] - merged["actual_adp"]).abs()
    return (
        merged.groupby("position", as_index=False)
        .agg(
            player_count=("player_name", "count"),
            mae=("absolute_error", "mean"),
            median_absolute_error=("absolute_error", "median"),
        )
        .sort_values("position")
        .reset_index(drop=True)
    )
