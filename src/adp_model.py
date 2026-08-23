"""Market-calibrated VORP curves for league-specific ADP estimation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .models import CORE_POSITIONS
from .scarcity import build_scarcity_frame


def _strict_market_curve(adps: pd.Series) -> np.ndarray:
    values = np.sort(adps.to_numpy(dtype=float))
    for index in range(1, len(values)):
        values[index] = max(values[index], values[index - 1] + 0.01)
    return values


def _monotonic_fit(values: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Dependency-free isotonic fit from VORP to observed market strength."""
    data = pd.DataFrame({"value": values, "target": targets}).groupby("value", as_index=False)["target"].mean().sort_values("value")
    blocks: list[dict[str, float]] = []
    for row in data.itertuples(index=False):
        blocks.append({"start": float(row.value), "end": float(row.value), "weight": 1.0, "sum": float(row.target)})
        while len(blocks) >= 2:
            previous, current = blocks[-2], blocks[-1]
            if previous["sum"] / previous["weight"] <= current["sum"] / current["weight"]:
                break
            blocks[-2:] = [{
                "start": previous["start"], "end": current["end"],
                "weight": previous["weight"] + current["weight"], "sum": previous["sum"] + current["sum"],
            }]
    knots: list[float] = []
    fitted: list[float] = []
    for block in blocks:
        mean = block["sum"] / block["weight"]
        knots.extend((block["start"], block["end"]))
        fitted.extend((mean, mean))
    return np.asarray(knots), np.asarray(fitted)


def _apply_fit(values: pd.Series, knots: np.ndarray, fitted: np.ndarray, lower_slope: float = 0.0, upper_slope: float = 0.0) -> np.ndarray:
    """Apply a calibrated curve with bounded local extrapolation at its edges."""
    numeric = values.to_numpy(dtype=float)
    output = np.interp(numeric, knots, fitted)
    below = numeric < knots[0]
    above = numeric > knots[-1]
    output[below] = fitted[0] + lower_slope * (numeric[below] - knots[0])
    output[above] = fitted[-1] + upper_slope * (numeric[above] - knots[-1])
    return np.clip(output, 0.0, 1.0)


def _edge_slopes(values: pd.Series, targets: pd.Series) -> tuple[float, float]:
    """Reference-market slope at the shallow/deep ends of a positional curve."""
    data = pd.DataFrame({"value": values, "target": targets}).groupby("value", as_index=False)["target"].mean().sort_values("value")
    if len(data) < 2:
        return 0.0, 0.0
    lower = max(0.0, float((data.iloc[1].target - data.iloc[0].target) / (data.iloc[1].value - data.iloc[0].value)))
    upper = max(0.0, float((data.iloc[-1].target - data.iloc[-2].target) / (data.iloc[-1].value - data.iloc[-2].value)))
    return lower, upper


def _local_curve_gaps(curve: np.ndarray) -> np.ndarray:
    """Smoothed local VORP-per-market-rank gaps for equivalent-rank nudges."""
    if len(curve) <= 1:
        return np.ones(len(curve))
    gaps = np.abs(np.diff(curve))
    positive = gaps[gaps > 1e-9]
    fallback = float(np.median(positive)) if len(positive) else 1.0
    output: list[float] = []
    for index in range(len(curve)):
        nearby = gaps[max(0, index - 2): min(len(gaps), index + 2)]
        nearby = nearby[nearby > 1e-9]
        output.append(float(np.median(nearby)) if len(nearby) else fallback)
    return np.asarray(output)


def _curve_coordinates(curve: np.ndarray) -> np.ndarray:
    """Make tied curve values strictly ordered without moving the next tier.

    A zero-VORP tail still has a meaningful market order.  Each tied run starts
    at its original value and receives a tiny deterministic decrement within
    that run.  Starting each run at zero (rather than at its absolute positional
    rank) lets, for example, league RB32 map cleanly to the reference's first
    zero-VORP RB slot even if that reference slot is RB37.
    """
    values = np.asarray(curve, dtype=float)
    output = values.copy()
    step = max(float(np.nanmax(np.abs(values))) if len(values) else 0.0, 1.0) * 1e-9
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[end] == values[start]:
            end += 1
        output[start:end] -= np.arange(end - start, dtype=float) * step
        start = end
    return output


def _market_slots(reference: pd.DataFrame, league: pd.DataFrame) -> pd.DataFrame:
    """Expected VORP curves at existing market slots (QB1, QB2, RB1, ...)."""
    pieces: list[pd.DataFrame] = []
    for position in CORE_POSITIONS:
        ref_group = reference[reference["position"] == position]
        league_group = league[league["position"] == position]
        if ref_group.empty or league_group.empty:
            continue
        market = ref_group.sort_values(["current_adp", "player_id"], kind="stable").copy()
        ref_curve = ref_group.sort_values(["scarcity_value", "player_id"], ascending=[False, True], kind="stable")["scarcity_value"].to_numpy()
        league_curve = league_group.sort_values(["scarcity_value", "player_id"], ascending=[False, True], kind="stable")["scarcity_value"].to_numpy()
        market["market_pos_rank"] = np.arange(1, len(market) + 1)
        market["reference_market_curve_value"] = ref_curve
        market["league_market_curve_value"] = league_curve
        market["local_reference_curve_gap"] = _local_curve_gaps(ref_curve)
        # VORP is exactly zero at replacement and can be tied through a long
        # depth segment.  The reference market still prices TE14, TE18, etc.
        # differently.  Preserve each tied slot's order instead of averaging
        # the segment into one ADP strength.
        market["reference_curve_coordinate"] = _curve_coordinates(ref_curve)
        market["league_curve_coordinate"] = _curve_coordinates(league_curve)
        pieces.append(market[["player_id", "market_pos_rank", "reference_market_curve_value", "league_market_curve_value", "reference_curve_coordinate", "league_curve_coordinate", "local_reference_curve_gap"]])
    return pd.concat(pieces, ignore_index=True)


def _calibrate_position_curves(result: pd.DataFrame) -> pd.DataFrame:
    """Fit each position's reference VORP curve to its observed market slots."""
    pieces: list[pd.DataFrame] = []
    for _, group in result.groupby("position", sort=False):
        slots = group.sort_values("market_pos_rank", kind="stable")
        knots, fitted = _monotonic_fit(slots["reference_curve_coordinate"].to_numpy(), slots["market_strength"].to_numpy())
        lower_slope, upper_slope = _edge_slopes(slots["reference_curve_coordinate"], slots["market_strength"])
        group = group.copy()
        group["reference_position_curve_strength"] = _apply_fit(group["reference_curve_coordinate"], knots, fitted, lower_slope, upper_slope)
        group["league_position_curve_strength"] = _apply_fit(group["league_curve_coordinate"], knots, fitted, lower_slope, upper_slope)
        group["reference_curve_strength"] = _apply_fit(group["reference_scarcity_value"], knots, fitted, lower_slope, upper_slope)
        group["league_curve_strength"] = _apply_fit(group["league_scarcity_value"], knots, fitted, lower_slope, upper_slope)
        pieces.append(group)
    output = pd.concat(pieces, ignore_index=True)
    output["position_curve_delta"] = output["league_position_curve_strength"] - output["reference_position_curve_strength"]
    output["curve_strength_delta"] = output["league_curve_strength"] - output["reference_curve_strength"]
    return output


def _position_reliability(result: pd.DataFrame) -> pd.Series:
    """Reference projection/market agreement determines how much to trust a nudge."""
    reliability: dict[str, float] = {}
    for position, group in result.groupby("position"):
        correlation = group["reference_scarcity_value"].corr(-group["market_pos_rank"], method="spearman")
        reliability[position] = float(np.clip(correlation if pd.notna(correlation) else 0.0, 0.0, 1.0))
    return result["position"].map(reliability).astype(float)


def _assign_adjusted_market_slots(result: pd.DataFrame) -> pd.DataFrame:
    """Assign nudged players to nearby existing market slots, not fresh projected ADPs."""
    slot_columns = ["position", "market_pos_rank", "market_strength", "position_curve_delta"]
    slots = result[slot_columns].rename(columns={"market_strength": "slot_market_strength", "position_curve_delta": "slot_position_curve_delta"})
    pieces: list[pd.DataFrame] = []
    for position, group in result.groupby("position", sort=False):
        ordered = group.sort_values(["within_position_score", "market_pos_rank", "player_id"], kind="stable").copy()
        ordered["adjusted_market_pos_rank"] = np.arange(1, len(ordered) + 1)
        position_slots = slots[slots["position"] == position].sort_values("market_pos_rank")
        ordered["slot_market_strength"] = position_slots["slot_market_strength"].to_numpy()
        ordered["slot_position_curve_delta"] = position_slots["slot_position_curve_delta"].to_numpy()
        pieces.append(ordered)
    return pd.concat(pieces, ignore_index=True)


def estimate_adjusted_adp(players: pd.DataFrame, reference_scoring: str, league_scoring: str, reference_roster: tuple[str, ...] | list[str], league_roster: tuple[str, ...] | list[str], reference_teams: int, league_teams: int) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Translate league VORP through position-specific reference market curves."""
    reference, reference_summary = build_scarcity_frame(players, reference_scoring, reference_roster, reference_teams)
    league, league_summary = build_scarcity_frame(players, league_scoring, league_roster, league_teams)
    feature_columns = ["player_id", "pos_rank", "pos_percentile", "value_above_replacement", "local_slope", "scarcity_value", "replacement_points", "replacement_rank", "elite_separation", "starter_demand"]
    reference_features = reference[feature_columns].rename(columns={column: f"reference_{column}" for column in feature_columns if column != "player_id"})
    league_features = league[feature_columns].rename(columns={column: f"league_{column}" for column in feature_columns if column != "player_id"})
    result = players.merge(reference_features, on="player_id", how="inner").merge(league_features, on="player_id", how="inner")
    result = result.merge(_market_slots(reference, league), on="player_id", how="inner")
    result["scarcity_delta"] = result["league_scarcity_value"] - result["reference_scarcity_value"]

    market_order = result.sort_values(["current_adp", "player_id"], kind="stable").index
    market_rank = pd.Series(np.arange(1, len(result) + 1), index=market_order)
    result["current_adp_rank"] = market_rank.reindex(result.index).astype(int)
    result["market_strength"] = 1 - (result["current_adp_rank"] - 1) / max(len(result) - 1, 1)
    result = _calibrate_position_curves(result)

    # Player-specific scoring change beyond the new league's expected curve at
    # that market slot becomes an equivalent number of within-position ranks.
    result["within_position_vor_residual"] = result["scarcity_delta"] - (result["league_market_curve_value"] - result["reference_market_curve_value"])
    result["projection_market_reliability"] = _position_reliability(result)
    result["raw_equivalent_rank_nudge"] = result["within_position_vor_residual"] / result["local_reference_curve_gap"]
    result["equivalent_rank_nudge"] = result["projection_market_reliability"] * result["raw_equivalent_rank_nudge"]
    result["within_position_score"] = result["market_pos_rank"] - result["equivalent_rank_nudge"]
    result = _assign_adjusted_market_slots(result)

    result["draft_score"] = result["slot_market_strength"] + result["slot_position_curve_delta"]
    result = result.sort_values(["draft_score", "current_adp", "player_id"], ascending=[False, True, True], kind="stable").copy()
    result["league_adjusted_rank"] = np.arange(1, len(result) + 1)
    result["league_adjusted_adp"] = _strict_market_curve(players["current_adp"])
    result["adp_change"] = result["current_adp"] - result["league_adjusted_adp"]
    result["position_impact"] = result.groupby("position")["slot_position_curve_delta"].transform("mean")
    return result.sort_values("league_adjusted_rank").reset_index(drop=True), {"reference": reference_summary, "league": league_summary}
