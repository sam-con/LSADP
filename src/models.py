"""Shared configuration and result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


CORE_POSITIONS = ("QB", "RB", "WR", "TE")


@dataclass(frozen=True)
class ReferenceLeague:
    """All V1 market assumptions live in one easy-to-change place."""

    name: str = "12-team Superflex PPR"
    teams: int = 12
    scoring_settings: Mapping[str, float] = field(
        default_factory=lambda: {
            "pass_yd": 0.04,
            "pass_td": 4.0,
            "pass_2pt": 2.0,
            "pass_int": -2.0,
            "rush_yd": 0.1,
            "rush_td": 6.0,
            "rush_2pt": 2.0,
            "rec": 1.0,
            "rec_yd": 0.1,
            "rec_td": 6.0,
            "rec_2pt": 2.0,
            "fum_lost": -2.0,
        }
    )
    roster_positions: tuple[str, ...] = (
        "QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "FLEX", "SUPER_FLEX", "BN", "BN", "BN", "BN", "BN", "BN",
    )
    adp_field: str = "adp_2qb"


ONE_QB_ROSTER = (
    "QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "FLEX", "BN", "BN", "BN", "BN", "BN", "BN", "BN",
)


def _reference_scoring(receptions: float) -> dict[str, float]:
    return {
        "pass_yd": 0.04,
        "pass_td": 4.0,
        "pass_2pt": 2.0,
        "pass_int": -2.0,
        "rush_yd": 0.1,
        "rush_td": 6.0,
        "rush_2pt": 2.0,
        "rec": receptions,
        "rec_yd": 0.1,
        "rec_td": 6.0,
        "rec_2pt": 2.0,
        "fum_lost": -2.0,
    }


DEFAULT_REFERENCE = ReferenceLeague()


def select_reference_league(scoring_settings: Mapping[str, object], roster_positions: tuple[str, ...] | list[str]) -> ReferenceLeague:
    """Choose the closest Sleeper market baseline from roster and reception scoring.

    Sleeper supplies standard, half-PPR, and PPR ADP for 1QB, and an `adp_2qb`
    market for Superflex/2QB. Other custom scoring is modeled as an adjustment to
    the nearest available market rather than pretending a bespoke market exists.
    """
    slots = [str(slot).upper() for slot in (roster_positions or [])]
    is_two_qb_market = any(slot in {"SUPER_FLEX", "SUPERFLEX", "OP"} for slot in slots) or slots.count("QB") >= 2
    try:
        receptions = float((scoring_settings or {}).get("rec", 0) or 0)
    except (TypeError, ValueError):
        receptions = 0.0
    nearest_reception = min((0.0, 0.5, 1.0), key=lambda value: abs(value - receptions))
    if is_two_qb_market:
        # Sleeper's public season payload exposes a single 2QB/SF ADP market.
        return ReferenceLeague(
            name="12-team Superflex PPR market",
            scoring_settings=_reference_scoring(1.0),
            roster_positions=DEFAULT_REFERENCE.roster_positions,
            adp_field="adp_2qb",
        )
    labels = {0.0: ("Standard", "adp_std"), 0.5: ("Half PPR", "adp_half_ppr"), 1.0: ("PPR", "adp_ppr")}
    label, adp_field = labels[nearest_reception]
    return ReferenceLeague(
        name=f"12-team 1QB {label} market",
        scoring_settings=_reference_scoring(nearest_reception),
        roster_positions=ONE_QB_ROSTER,
        adp_field=adp_field,
    )


@dataclass
class ScoringResult:
    points: float
    applied_rules: dict[str, float]
    unsupported_rules: list[str]
