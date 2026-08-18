"""VORP computation helpers."""

from __future__ import annotations

import pandas as pd


def build_vorp_table(curves: pd.DataFrame, replacement: pd.DataFrame, rank_limit: int | None = None) -> pd.DataFrame:
    """Evaluate VORP by position and positional rank."""

    limit = rank_limit or int(curves.groupby("position")["rank"].max().max())
    replacement_map = replacement.set_index("position")[["replacement_rank", "replacement_ppg"]].to_dict("index")

    records: list[dict[str, float | int | str]] = []
    for _, row in curves.iterrows():
        position = str(row["position"])
        rank = int(row["rank"])
        if rank > limit:
            continue
        replacement_ppg = float(replacement_map[position]["replacement_ppg"])
        expected_ppg = float(row["expected_ppg"])
        records.append(
            {
                "position": position,
                "rank": rank,
                "expected_ppg": expected_ppg,
                "replacement_rank": int(replacement_map[position]["replacement_rank"]),
                "replacement_ppg": replacement_ppg,
                "vorp": expected_ppg - replacement_ppg,
            }
        )
    return pd.DataFrame(records)


def merge_metric_into_players(
    adp_frame: pd.DataFrame,
    metric_table: pd.DataFrame,
    metric_column: str,
    output_column: str,
) -> pd.DataFrame:
    """Join a rank-based metric onto player ADP rows."""

    merged = adp_frame.merge(
        metric_table[["position", "rank", metric_column]].rename(columns={"rank": "pos_rank", metric_column: output_column}),
        on=["position", "pos_rank"],
        how="left",
    )
    return merged

