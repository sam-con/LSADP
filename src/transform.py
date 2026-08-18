"""League-specific ADP transformation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import inverse_adp_utility, stable_rank_from_metric


def apply_league_transformation(
    calibrated_adp: pd.DataFrame,
    baseline_metric_column: str,
    target_metric_column: str,
    explanation_metric_label: str = "VORP",
    utility_transform: str = "neg_log",
    anchor_label: str | None = None,
) -> pd.DataFrame:
    """Transform current market ADP into league-adjusted ADP."""

    working = calibrated_adp.copy()
    working["delta_metric"] = working[target_metric_column].fillna(0.0) - working[baseline_metric_column].fillna(0.0)
    working["target_utility"] = working["utility"] + working["market_coefficient"] * working["delta_metric"]
    working["league_adjusted_adp"] = inverse_adp_utility(
        working["target_utility"],
        transform=utility_transform,
    ).to_numpy(dtype=float)
    working["canonical_environment_label"] = anchor_label or working.get("canonical_environment_label", "Canonical")

    ordered = stable_rank_from_metric(working, column="target_utility", ascending=False, rank_column="adjusted_rank")
    ordered["adp_change"] = ordered["adp"] - ordered["league_adjusted_adp"]
    ordered["short_explanation"] = ordered.apply(
        lambda row: (
            f"{row['canonical_environment_label']} anchor: {row['position']}{int(row['pos_rank'])} moves from ADP {row['adp']:.1f} -> "
            f"{row['league_adjusted_adp']:.1f} because this league changes expected positional "
            f"advantage by {row['delta_metric']:.2f} {explanation_metric_label}."
        ),
        axis=1,
    )
    return ordered.sort_values("adjusted_rank").reset_index(drop=True)


def compute_positional_impact_summary(results: pd.DataFrame, baseline_metric_column: str, target_metric_column: str) -> pd.DataFrame:
    """Summarize positional impact from baseline to target environment."""

    summary = (
        results.groupby("position", as_index=False)
        .agg(
            baseline_metric=(baseline_metric_column, "mean"),
            target_metric=(target_metric_column, "mean"),
            mean_adp_change=("adp_change", "mean"),
        )
    )
    summary["impact_pct"] = np.where(
        summary["baseline_metric"].abs() > 1e-6,
        (summary["target_metric"] - summary["baseline_metric"]) / summary["baseline_metric"].abs() * 100.0,
        0.0,
    )
    return summary.sort_values("position").reset_index(drop=True)
