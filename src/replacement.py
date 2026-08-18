"""Replacement-level modeling from roster settings."""

from __future__ import annotations

from collections import Counter

import pandas as pd

from src.config import CORE_POSITIONS, FLEX_POSITIONS, SUPERFLEX_POSITIONS
from src.models import LeagueSettings, ReplacementLevel


def curve_value_at_rank(curves: pd.DataFrame, position: str, rank: int) -> float:
    match = curves[(curves["position"] == position) & (curves["rank"] == rank)]
    if match.empty:
        same_position = curves[curves["position"] == position].sort_values("rank")
        if same_position.empty:
            return 0.0
        return float(same_position.iloc[-1]["expected_ppg"])
    return float(match.iloc[0]["expected_ppg"])


def calculate_starter_demand_replacement(league: LeagueSettings, curves: pd.DataFrame) -> pd.DataFrame:
    """Allocate mandatory and flexible starters league-wide to derive replacement ranks."""

    team_count = league.total_rosters
    mandatory = league.mandatory_starter_counts()
    flex_slots = league.flex_slots() * team_count
    superflex_slots = league.superflex_slots() * team_count

    drafted_counts = Counter({position: mandatory.get(position, 0) * team_count for position in CORE_POSITIONS})

    for _ in range(flex_slots):
        best_position = max(
            FLEX_POSITIONS,
            key=lambda position: curve_value_at_rank(curves, position, drafted_counts[position] + 1),
        )
        drafted_counts[best_position] += 1

    for _ in range(superflex_slots):
        best_position = max(
            SUPERFLEX_POSITIONS,
            key=lambda position: curve_value_at_rank(curves, position, drafted_counts[position] + 1),
        )
        drafted_counts[best_position] += 1

    records: list[dict[str, float | int | str]] = []
    for position in CORE_POSITIONS:
        replacement_rank = max(int(drafted_counts[position]) + 1, 1)
        replacement_ppg = curve_value_at_rank(curves, position, replacement_rank)
        records.append(
            {
                "position": position,
                "method": "Starter Demand Replacement",
                "replacement_rank": replacement_rank,
                "replacement_ppg": replacement_ppg,
            }
        )

    return pd.DataFrame(records)


def calculate_historical_roster_replacement(player_weeks: pd.DataFrame, min_starter_weeks: int = 4) -> pd.DataFrame:
    """Estimate replacement ranks from historical starter usage."""

    starters = player_weeks[player_weeks["starter_flag"]].copy()
    starter_ppg = (
        starters.groupby(["season", "player_id", "player_name", "position"], as_index=False)
        .agg(starter_weeks=("week", "nunique"), ppg=("fantasy_points", "mean"))
    )
    starter_ppg = starter_ppg[starter_ppg["starter_weeks"] >= min_starter_weeks]
    starter_ppg = starter_ppg.sort_values(["season", "position", "ppg"], ascending=[True, True, False]).copy()
    starter_ppg["rank"] = starter_ppg.groupby(["season", "position"]).cumcount() + 1
    replacement = (
        starter_ppg.groupby("position", as_index=False)["rank"]
        .median()
        .rename(columns={"rank": "replacement_rank"})
    )
    replacement["replacement_rank"] = replacement["replacement_rank"].round().astype(int) + 1
    replacement["method"] = "Historical Roster Replacement"
    replacement["replacement_ppg"] = 0.0
    return replacement[["position", "method", "replacement_rank", "replacement_ppg"]]


def attach_replacement_ppg(replacement: pd.DataFrame, curves: pd.DataFrame) -> pd.DataFrame:
    """Add replacement-level PPG using the fitted curve."""

    frame = replacement.copy()
    frame["replacement_ppg"] = frame.apply(
        lambda row: curve_value_at_rank(curves, str(row["position"]), int(row["replacement_rank"])),
        axis=1,
    )
    return frame

