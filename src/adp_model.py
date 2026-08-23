"""Interpretable market-preserving conversion from scarcity changes to adjusted ADP."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .scarcity import build_scarcity_frame


def _strict_market_curve(adps: pd.Series) -> np.ndarray:
    """Retain the market's spacing while ensuring each output rank has a unique ADP."""
    values = np.sort(adps.to_numpy(dtype=float))
    for index in range(1, len(values)):
        values[index] = max(values[index], values[index - 1] + 0.01)
    return values


def estimate_adjusted_adp(players: pd.DataFrame, reference_scoring: str, league_scoring: str, reference_roster: tuple[str, ...] | list[str], league_roster: tuple[str, ...] | list[str], reference_teams: int, league_teams: int, adjustment_strength: float = 0.75) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Re-rank the whole pool from market ADP plus *change* in normalized scarcity value.

    Market ADP supplies the baseline ordering. Only a change between reference and league
    positional environments can move a player, which intentionally downweights raw points.
    """
    reference, reference_summary = build_scarcity_frame(players, reference_scoring, reference_roster, reference_teams)
    ref_cols = ["player_id", "pos_rank", "pos_percentile", "value_above_replacement", "local_slope", "scarcity_value", "replacement_points", "replacement_rank", "elite_separation", "starter_demand"]
    reference = reference[ref_cols].rename(columns={column: f"reference_{column}" for column in ref_cols if column != "player_id"})
    league, league_summary = build_scarcity_frame(players, league_scoring, league_roster, league_teams)
    league_cols = ["player_id", "pos_rank", "pos_percentile", "value_above_replacement", "local_slope", "scarcity_value", "replacement_points", "replacement_rank", "elite_separation", "starter_demand"]
    league = league[league_cols].rename(columns={column: f"league_{column}" for column in league_cols if column != "player_id"})
    result = players.merge(reference, on="player_id", how="inner").merge(league, on="player_id", how="inner")
    result["scarcity_delta"] = result["league_scarcity_value"] - result["reference_scarcity_value"]
    # A robust unit avoids one extreme projection forcing the entire ADP board to reshuffle.
    median = result["scarcity_delta"].median()
    mad = (result["scarcity_delta"] - median).abs().median()
    delta_scale = max(float(mad * 1.4826), 0.05)
    result["normalized_scarcity_delta"] = (result["scarcity_delta"] - median) / delta_scale
    market_order = result.sort_values(["current_adp", "player_id"], kind="stable").index
    market_rank = pd.Series(np.arange(1, len(result) + 1), index=market_order)
    result["current_adp_rank"] = market_rank.reindex(result.index).astype(int)
    # Log ADP expresses market spacing; normalized scarcity deltas are deliberately modest.
    result["draft_score"] = -np.log(result["current_adp"].clip(lower=0.01)) + adjustment_strength * result["normalized_scarcity_delta"]
    result = result.sort_values(["draft_score", "current_adp", "player_id"], ascending=[False, True, True], kind="stable").copy()
    result["league_adjusted_rank"] = np.arange(1, len(result) + 1)
    result["league_adjusted_adp"] = _strict_market_curve(players["current_adp"])
    result["adp_change"] = result["current_adp"] - result["league_adjusted_adp"]
    result["position_impact"] = result.groupby("position")["scarcity_delta"].transform("mean")
    result = result.sort_values("league_adjusted_rank").reset_index(drop=True)
    return result, {"reference": reference_summary, "league": league_summary}
