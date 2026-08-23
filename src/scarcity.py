"""Positional curves, replacement levels, and scale-resistant scarcity values."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable

import numpy as np
import pandas as pd

from .models import CORE_POSITIONS

FLEX_SHARES = {"RB": 0.40, "WR": 0.45, "TE": 0.15}
SUPERFLEX_SHARES = {"QB": 0.45, "RB": 0.20, "WR": 0.25, "TE": 0.10}


def starter_demand(roster_positions: Iterable[str], teams: int) -> dict[str, float]:
    """Estimate position starters from Sleeper slots, allocating flexible slots explicitly."""
    slots = [str(slot).upper() for slot in (roster_positions or [])]
    direct = Counter(slot for slot in slots if slot in CORE_POSITIONS)
    demand = {position: float(direct[position]) for position in CORE_POSITIONS}
    flex_count = sum(slot in {"FLEX", "REC_FLEX", "WRRB_FLEX", "RBWR_FLEX"} for slot in slots)
    superflex_count = sum(slot in {"SUPER_FLEX", "SUPERFLEX", "OP"} for slot in slots)
    for position, share in FLEX_SHARES.items():
        demand[position] += flex_count * share
    for position, share in SUPERFLEX_SHARES.items():
        demand[position] += superflex_count * share
    return {position: demand[position] * max(int(teams or 0), 1) for position in CORE_POSITIONS}


def _replacement_rank(total_starters: float, player_count: int) -> int:
    # Replacement is the first player after all league-wide starting slots have
    # been filled. Bench/depth value belongs in the observed ADP curve, not in an
    # arbitrary extension of the replacement pool.
    return min(player_count, max(1, int(math.ceil(total_starters)) + 1))


def build_scarcity_frame(players: pd.DataFrame, points_column: str, roster_positions: Iterable[str], teams: int) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Add curve features. Values are later normalized by a pooled curve scale."""
    frame = players.copy()
    demand = starter_demand(roster_positions, teams)
    pieces: list[pd.DataFrame] = []
    summary: dict[str, dict] = {}
    ranges: list[float] = []
    for position in CORE_POSITIONS:
        group = frame[frame["position"] == position].sort_values([points_column, "player_id"], ascending=[False, True]).copy()
        if group.empty:
            continue
        group["pos_rank"] = np.arange(1, len(group) + 1)
        group["pos_percentile"] = 1 - (group["pos_rank"] - 1) / max(len(group) - 1, 1)
        replacement_rank = _replacement_rank(demand[position], len(group))
        replacement_points = float(group.iloc[replacement_rank - 1][points_column])
        group["replacement_rank"] = replacement_rank
        group["replacement_points"] = replacement_points
        # Traditional VORP is value *above* replacement. Depth players remain in
        # the market ADP curve, but do not receive a positional-scarcity boost.
        group["value_above_replacement"] = (group[points_column] - replacement_points).clip(lower=0.0)
        group["elite_separation"] = float(group.iloc[0][points_column]) - replacement_points
        group["replacement_separation"] = replacement_points - float(group.iloc[min(replacement_rank, len(group) - 1)][points_column])
        group["local_slope"] = group[points_column].diff(-1).abs().fillna(0.0)
        group["starter_demand"] = demand[position]
        ranges.append(max(float(group.iloc[0][points_column]) - replacement_points, 1.0))
        summary[position] = {
            "replacement_rank": replacement_rank,
            "replacement_points": replacement_points,
            "elite_separation": float(group.iloc[0][points_column]) - replacement_points,
            "starter_demand": demand[position],
        }
        pieces.append(group)
    if not pieces:
        return frame, summary
    result = pd.concat(pieces, ignore_index=True)
    # Median cross-position elite-to-replacement gap removes league-wide point-scale changes.
    pooled_scale = float(np.median(ranges)) if ranges else 1.0
    result["pooled_curve_scale"] = max(pooled_scale, 1.0)
    # Starter demand is already expressed in each position's replacement rank.
    # Applying a second demand multiplier would double-count roster scarcity.
    result["demand_weight"] = 1.0
    result["scarcity_value"] = result["value_above_replacement"] / result["pooled_curve_scale"]
    return result, summary
