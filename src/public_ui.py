"""Pure presentation helpers for the public Streamlit UI."""

from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd

from src.models import LeagueSettings
from src.transform import compute_positional_impact_summary

SUPPORTED_POSITIONS = ("QB", "RB", "WR", "TE")
IGNORED_ROSTER_SLOTS = {"BN", "IR", "TAXI"}
NON_MODELED_AUXILIARY_SLOTS = {"FLEX", "SUPER_FLEX"}
ROSTER_SLOT_ORDER = (
    "QB",
    "RB",
    "WR",
    "TE",
    "FLEX",
    "SUPER_FLEX",
    "K",
    "DEF",
    "DL",
    "LB",
    "DB",
    "IDP_FLEX",
)
ROSTER_SLOT_LABELS = {
    "SUPER_FLEX": "SUPERFLEX",
    "IDP_FLEX": "IDP FLEX",
}


def ordered_public_positions(positions: list[str] | set[str]) -> list[str]:
    values = {str(position).upper() for position in positions}
    return [position for position in SUPPORTED_POSITIONS if position in values]


def modeled_positions_for_league(league: LeagueSettings, results: pd.DataFrame) -> list[str]:
    league_positions = {slot for slot in league.roster_positions if slot in SUPPORTED_POSITIONS}
    result_positions = {str(position).upper() for position in results.get("position", pd.Series(dtype="object")).dropna().tolist()}
    ordered = [position for position in SUPPORTED_POSITIONS if position in league_positions and position in result_positions]
    if ordered:
        return ordered
    return ordered_public_positions(result_positions)


def unsupported_roster_positions(league: LeagueSettings) -> list[str]:
    unsupported = {
        slot
        for slot in league.roster_positions
        if slot not in SUPPORTED_POSITIONS and slot not in IGNORED_ROSTER_SLOTS and slot not in NON_MODELED_AUXILIARY_SLOTS
    }
    return [slot.replace("_", " ") for slot in sorted(unsupported)]


def missing_modeled_positions(league: LeagueSettings, results: pd.DataFrame) -> list[str]:
    relevant = {slot for slot in league.roster_positions if slot in SUPPORTED_POSITIONS}
    modeled = set(modeled_positions_for_league(league, results))
    return [position for position in SUPPORTED_POSITIONS if position in relevant and position not in modeled]


def league_format_label(league: LeagueSettings) -> str:
    return "Superflex" if league.superflex_slots() > 0 else "1QB"


def scoring_primary_label(league: LeagueSettings) -> str:
    reception = float(league.scoring_settings.get("rec", 0.0))
    if abs(reception - 1.0) < 1e-9:
        return "PPR"
    if abs(reception - 0.5) < 1e-9:
        return "Half-PPR"
    if abs(reception) < 1e-9:
        return "Standard"
    return f"{reception:g} PPR"


def scoring_detail_lines(league: LeagueSettings) -> list[str]:
    scoring = league.scoring_settings
    details: list[str] = []
    pass_td = float(scoring.get("pass_td", 4.0))
    if abs(pass_td - 4.0) > 1e-9:
        details.append(f"{pass_td:g}-point passing TDs")
    te_premium = float(scoring.get("rec_te", 0.0)) + float(scoring.get("bonus_rec_te", 0.0))
    if abs(te_premium) > 1e-9:
        details.append(f"TE +{te_premium:g} reception premium")
    for key, label in (
        ("bonus_pass_yd_300", "300-yard passing bonus"),
        ("bonus_pass_yd_400", "400-yard passing bonus"),
    ):
        value = float(scoring.get(key, 0.0))
        if abs(value) > 1e-9:
            details.append(f"+{value:g} {label}")
    return details


def scoring_summary_text(league: LeagueSettings) -> str:
    primary = scoring_primary_label(league)
    details = scoring_detail_lines(league)
    if not details:
        return f"{primary} scoring"
    return f"{primary}; " + "; ".join(details)


def starting_lineup_text(league: LeagueSettings) -> str:
    counts = Counter(slot for slot in league.roster_positions if slot not in IGNORED_ROSTER_SLOTS)
    parts: list[str] = []
    for slot in ROSTER_SLOT_ORDER:
        count = counts.get(slot, 0)
        if count <= 0:
            continue
        label = ROSTER_SLOT_LABELS.get(slot, slot.replace("_", " "))
        parts.append(f"{count} {label}")
    remaining = sorted(set(counts) - set(ROSTER_SLOT_ORDER))
    for slot in remaining:
        parts.append(f"{counts[slot]} {slot.replace('_', ' ')}")
    return " / ".join(parts)


def _filter_results(results: pd.DataFrame, positions: list[str] | None = None, search: str = "") -> pd.DataFrame:
    filtered = results.copy()
    if positions:
        filtered = filtered[filtered["position"].isin(positions)].copy()
    if search.strip():
        needle = search.strip().lower()
        filtered = filtered[
            filtered["player_name"].str.lower().str.contains(needle)
            | filtered["position"].str.lower().str.contains(needle)
            | filtered["team"].fillna("").astype(str).str.lower().str.contains(needle)
        ].copy()
    return filtered.sort_values(["adjusted_rank", "player_name"]).reset_index(drop=True)


def filter_results_for_display(results: pd.DataFrame, positions: list[str] | None = None, search: str = "") -> pd.DataFrame:
    return _filter_results(results, positions=positions, search=search)


def _round_public_numeric_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rounded = frame.copy()
    for column in columns:
        if column in rounded.columns:
            rounded[column] = rounded[column].astype(float).round(1)
    return rounded


def build_public_rankings_frame(
    results: pd.DataFrame,
    *,
    positions: list[str] | None = None,
    search: str = "",
    include_pos_rank: bool = True,
) -> pd.DataFrame:
    filtered = _filter_results(results, positions=positions, search=search)
    public = pd.DataFrame(
        {
            "Rank": filtered["adjusted_rank"].astype(int),
            "Player": filtered["player_name"].astype(str),
            "Pos": filtered["position"].astype(str),
            "Team": filtered["team"].fillna("").replace("", "-").astype(str),
            "Market ADP": filtered["adp"].astype(float),
            "League ADP": filtered["league_adjusted_adp"].astype(float),
            "Change": filtered["adp_change"].astype(float),
        }
    )
    if include_pos_rank and "pos_rank" in filtered.columns:
        public.insert(4, "Pos Rank", filtered["pos_rank"].astype(int))
    return _round_public_numeric_columns(public, ["Market ADP", "League ADP", "Change"])


def build_public_download_frame(results: pd.DataFrame) -> pd.DataFrame:
    public = pd.DataFrame(
        {
            "Rank": results["adjusted_rank"].astype(int),
            "Player": results["player_name"].astype(str),
            "Position": results["position"].astype(str),
            "Team": results["team"].fillna("").replace("", "-").astype(str),
            "Market ADP": results["adp"].astype(float),
            "League ADP": results["league_adjusted_adp"].astype(float),
            "Change": results["adp_change"].astype(float),
        }
    ).sort_values(["Rank", "Player"]).reset_index(drop=True)
    return _round_public_numeric_columns(public, ["Market ADP", "League ADP", "Change"])


def build_biggest_risers_frame(results: pd.DataFrame, limit: int = 8) -> pd.DataFrame:
    risers = results[results["adp_change"] > 0].nlargest(limit, "adp_change")
    return build_public_rankings_frame(risers, include_pos_rank=False)


def build_biggest_fallers_frame(results: pd.DataFrame, limit: int = 8) -> pd.DataFrame:
    fallers = results[results["adp_change"] < 0].nsmallest(limit, "adp_change")
    return build_public_rankings_frame(fallers, include_pos_rank=False)


def _position_impact_label(mean_change: float) -> str:
    if mean_change >= 5.0:
        return "Much more valuable"
    if mean_change >= 2.0:
        return "More valuable"
    if mean_change <= -5.0:
        return "Much less valuable"
    if mean_change <= -2.0:
        return "Less valuable"
    return "Near baseline"


def build_position_impact_frame(results: pd.DataFrame, positions: list[str]) -> pd.DataFrame:
    summary = compute_positional_impact_summary(results, "canonical_metric", "league_metric")
    summary = summary[summary["position"].isin(positions)].copy()
    summary["Positional Value"] = summary["mean_adp_change"].map(_position_impact_label)
    summary["Avg Change"] = summary["mean_adp_change"].astype(float).round(1)
    return summary.rename(columns={"position": "Position"})[["Position", "Positional Value", "Avg Change"]]


def build_historical_reference_frame(match_summary: pd.DataFrame, positions: list[str]) -> pd.DataFrame:
    if match_summary.empty:
        return pd.DataFrame(columns=["Position", "Historical Reference", "Coverage"])
    filtered = match_summary[match_summary["position"].isin(positions)].copy()
    filtered["Historical Reference"] = filtered["match_quality"].astype(str)
    filtered["Coverage"] = filtered["status"].astype(str)
    return filtered.rename(columns={"position": "Position"})[["Position", "Historical Reference", "Coverage"]]


def describe_scoring_difference(key: str, target_value: float, matched_value: float) -> str:
    if key == "rec":
        return f"This league awards {target_value:g} points per reception. The closest saved reference awards {matched_value:g}."
    if key == "pass_td":
        return f"This league awards {target_value:g} points per passing TD. The closest saved reference awards {matched_value:g}."
    if key in {"rec_te", "bonus_rec_te"}:
        return f"This league gives tight ends {target_value:g} extra reception points. The closest saved reference gives {matched_value:g}."
    if key == "bonus_pass_yd_300":
        return f"This league gives a {target_value:g}-point 300-yard passing bonus. The closest saved reference gives {matched_value:g}."
    if key == "bonus_pass_yd_400":
        return f"This league gives a {target_value:g}-point 400-yard passing bonus. The closest saved reference gives {matched_value:g}."
    return f"{key.replace('_', ' ').title()}: this league uses {target_value:g}; the closest saved reference uses {matched_value:g}."


def build_historical_match_details(match_summary: pd.DataFrame, positions: list[str]) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    if match_summary.empty:
        return details
    filtered = match_summary[match_summary["position"].isin(positions)].copy()
    for row in filtered.to_dict(orient="records"):
        if row.get("match_quality") == "Exact":
            continue
        differences = [
            describe_scoring_difference(
                str(diff["key"]),
                float(diff["target_value"]),
                float(diff["matched_value"]),
            )
            for diff in row.get("differing_fields", [])
        ]
        if not differences:
            continue
        details.append(
            {
                "position": str(row["position"]),
                "match_quality": str(row["match_quality"]),
                "differences": differences,
            }
        )
    return details


def public_player_explanation(player_row: pd.Series) -> str:
    change = float(player_row.get("adp_change", 0.0))
    player_name = str(player_row["player_name"])
    position = str(player_row["position"])
    if abs(change) < 1.5:
        return f"{player_name} stays close to current Sleeper market ADP in this league."
    direction = "earlier" if change > 0 else "later"
    return (
        f"This format changes {position} positional value enough to move {player_name} about "
        f"{abs(change):.1f} picks {direction} than current Sleeper market ADP."
    )


def build_player_advanced_frame(player_row: pd.Series) -> pd.DataFrame:
    rows = [
        {"Metric": "Historical Expected PPG", "Canonical": player_row.get("canonical_expected_ppg"), "League": player_row.get("league_expected_ppg")},
        {"Metric": "Replacement Value", "Canonical": player_row.get("canonical_vorp"), "League": player_row.get("league_vorp")},
    ]
    frame = pd.DataFrame(rows)
    for column in ("Canonical", "League"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").round(2)
    return frame


def public_methodology_lines() -> list[str]:
    return [
        "The app starts with current Sleeper market ADP from saved BeatADP canonical markets.",
        "Historical Sleeper scoring is matched by position to leagues with similar scoring settings.",
        "Your league size and starting lineup determine replacement levels and positional value.",
        "Those changes are used to adjust market ADP into a league-specific draft board.",
    ]
