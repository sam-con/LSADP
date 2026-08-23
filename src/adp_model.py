"""Reference-calibrated, market-anchored conversion from VOR to adjusted ADP."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .scarcity import build_scarcity_frame


def _strict_market_curve(adps: pd.Series) -> np.ndarray:
    """Retain market spacing while ensuring each output rank has a unique ADP."""
    values = np.sort(adps.to_numpy(dtype=float))
    for index in range(1, len(values)):
        values[index] = max(values[index], values[index - 1] + 0.01)
    return values


def _monotonic_fit(values: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit a dependency-free isotonic VOR-to-market-strength curve with PAVA."""
    data = (
        pd.DataFrame({"value": values, "target": targets})
        .groupby("value", as_index=False)["target"]
        .mean()
        .sort_values("value")
    )
    blocks: list[dict[str, float]] = []
    for row in data.itertuples(index=False):
        blocks.append({"start": float(row.value), "end": float(row.value), "weight": 1.0, "sum": float(row.target)})
        while len(blocks) >= 2:
            previous, current = blocks[-2], blocks[-1]
            if previous["sum"] / previous["weight"] <= current["sum"] / current["weight"]:
                break
            blocks[-2:] = [{
                "start": previous["start"],
                "end": current["end"],
                "weight": previous["weight"] + current["weight"],
                "sum": previous["sum"] + current["sum"],
            }]
    knots: list[float] = []
    fitted: list[float] = []
    for block in blocks:
        mean = block["sum"] / block["weight"]
        knots.extend((block["start"], block["end"]))
        fitted.extend((mean, mean))
    return np.asarray(knots), np.asarray(fitted)


def _apply_monotonic_fit(values: pd.Series, knots: np.ndarray, fitted: np.ndarray) -> np.ndarray:
    return np.interp(values.to_numpy(dtype=float), knots, fitted, left=fitted[0], right=fitted[-1])


def _market_ranked_curve(frame: pd.DataFrame, value_column: str, output_column: str) -> pd.DataFrame:
    """Apply a positional projection curve to existing positional market ranks."""
    pieces: list[pd.DataFrame] = []
    for _, group in frame.groupby("position", sort=False):
        market_order = group.sort_values(["current_adp", "player_id"], kind="stable").copy()
        projected_curve = group.sort_values([value_column, "player_id"], ascending=[False, True], kind="stable")[value_column].to_numpy()
        market_order[output_column] = projected_curve
        pieces.append(market_order[["player_id", output_column]])
    return pd.concat(pieces, ignore_index=True)


def estimate_adjusted_adp(players: pd.DataFrame, reference_scoring: str, league_scoring: str, reference_roster: tuple[str, ...] | list[str], league_roster: tuple[str, ...] | list[str], reference_teams: int, league_teams: int, position_curve_weight: float = 0.20, player_shuffle_weight: float = 0.05) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Map league VOR through a reference-market curve while anchoring to market ADP.

    Reference projected VOR is monotonically calibrated to the observed market ADP.
    League VOR is passed through that same curve. Only the *difference* between the
    two calibrated values changes the market order. A 20% position-curve weight
    moves positional value; a separate 5% residual permits only modest projection-
    led player shuffling within the position.
    """
    reference, reference_summary = build_scarcity_frame(players, reference_scoring, reference_roster, reference_teams)
    league, league_summary = build_scarcity_frame(players, league_scoring, league_roster, league_teams)
    feature_columns = ["player_id", "pos_rank", "pos_percentile", "value_above_replacement", "local_slope", "scarcity_value", "replacement_points", "replacement_rank", "elite_separation", "starter_demand"]
    reference_features = reference[feature_columns].rename(columns={column: f"reference_{column}" for column in feature_columns if column != "player_id"})
    league_features = league[feature_columns].rename(columns={column: f"league_{column}" for column in feature_columns if column != "player_id"})
    reference_market_curve = _market_ranked_curve(reference, "scarcity_value", "reference_market_curve_value")
    league_market_curve = _market_ranked_curve(league, "scarcity_value", "league_market_curve_value")
    result = players.merge(reference_features, on="player_id", how="inner").merge(league_features, on="player_id", how="inner")
    result = result.merge(reference_market_curve, on="player_id", how="inner").merge(league_market_curve, on="player_id", how="inner")
    result["scarcity_delta"] = result["league_scarcity_value"] - result["reference_scarcity_value"]

    market_order = result.sort_values(["current_adp", "player_id"], kind="stable").index
    market_rank = pd.Series(np.arange(1, len(result) + 1), index=market_order)
    result["current_adp_rank"] = market_rank.reindex(result.index).astype(int)
    result["market_strength"] = 1 - (result["current_adp_rank"] - 1) / max(len(result) - 1, 1)

    # The reference curve learns a constrained VOR-to-market relationship, not a
    # new player ranking. Isotonic fitting makes larger VOR never imply worse ADP.
    knots, fitted = _monotonic_fit(result["reference_scarcity_value"].to_numpy(), result["market_strength"].to_numpy())
    result["reference_curve_strength"] = _apply_monotonic_fit(result["reference_scarcity_value"], knots, fitted)
    result["league_curve_strength"] = _apply_monotonic_fit(result["league_scarcity_value"], knots, fitted)
    result["curve_strength_delta"] = result["league_curve_strength"] - result["reference_curve_strength"]
    result["reference_position_curve_strength"] = _apply_monotonic_fit(result["reference_market_curve_value"], knots, fitted)
    result["league_position_curve_strength"] = _apply_monotonic_fit(result["league_market_curve_value"], knots, fitted)
    result["position_curve_delta"] = result["league_position_curve_strength"] - result["reference_position_curve_strength"]
    result["player_shuffle_delta"] = result["curve_strength_delta"] - result["position_curve_delta"]
    result["draft_score"] = (
        result["market_strength"]
        + position_curve_weight * result["position_curve_delta"]
        + player_shuffle_weight * result["player_shuffle_delta"]
    )

    result = result.sort_values(["draft_score", "current_adp", "player_id"], ascending=[False, True, True], kind="stable").copy()
    result["league_adjusted_rank"] = np.arange(1, len(result) + 1)
    result["league_adjusted_adp"] = _strict_market_curve(players["current_adp"])
    result["adp_change"] = result["current_adp"] - result["league_adjusted_adp"]
    result["position_impact"] = result.groupby("position")["position_curve_delta"].transform("mean")
    result = result.sort_values("league_adjusted_rank").reset_index(drop=True)
    return result, {"reference": reference_summary, "league": league_summary}
