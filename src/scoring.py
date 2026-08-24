"""Score Sleeper projection statistics with arbitrary supported Sleeper rules."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from .models import ScoringResult

# These are direct counting statistics present in Sleeper's season-projection
# payload. Derived statistics are declared separately below.
DIRECT_STAT_KEYS = frozenset(
    {
        "pass_yd", "pass_td", "pass_2pt", "pass_int", "pass_cmp", "pass_att", "pass_fd", "pass_int_td",
        "rush_yd", "rush_td", "rush_2pt", "rush_att", "rush_fd",
        "rec", "rec_yd", "rec_td", "rec_2pt", "rec_fd",
        "fum_lost", "fum_rec", "blk_kick", "sack", "int", "def_fum_td", "def_kr_td", "pr_td",
        "idp_blk_kick", "idp_ff", "idp_fum_rec", "idp_int", "idp_sack", "idp_safe", "idp_tkl",
        "idp_tkl_ast", "idp_tkl_solo", "xpm", "xpmiss", "fgm_40_49", "fgm_50p", "fgm_yds",
        "fgmiss_40_49", "fgmiss_50p", "pts_allow_0", "yds_allow_0_100",
        "bonus_rec_rb", "bonus_rec_te", "bonus_rec_wr", "bonus_rush_td_qb",
    }
)
DERIVED_STAT_KEYS = frozenset({"pass_inc"})
SUPPORTED_STAT_KEYS = DIRECT_STAT_KEYS | DERIVED_STAT_KEYS


def numeric_rules(scoring_settings: Mapping[str, object]) -> dict[str, float]:
    """Keep numeric, non-zero scoring settings and discard API nulls safely."""
    rules: dict[str, float] = {}
    for key, value in (scoring_settings or {}).items():
        if isinstance(value, bool):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number:
            rules[str(key)] = number
    return rules


def _rule_position(rule: str) -> str | None:
    """Return a non-core position affected only by a scoring rule, when known."""
    key = rule.lower()
    if key.startswith("idp_"):
        return "IDP"
    if key.startswith(("def_", "pts_allow", "yds_allow", "bonus_def_")):
        return "DEF"
    if key.startswith(("fg", "xpm", "bonus_fg", "bonus_xp")):
        return "K"
    return None


def _active_roster_positions(roster_positions: Iterable[str]) -> set[str]:
    """Normalize Sleeper's defensive/kicker roster slot spellings."""
    slots = {str(slot).upper() for slot in roster_positions}
    active: set[str] = set()
    if slots & {"DEF", "DST", "D/ST"}:
        active.add("DEF")
    if "K" in slots:
        active.add("K")
    if any(slot.startswith("IDP") or slot in {"DL", "LB", "DB", "CB", "S"} for slot in slots):
        active.add("IDP")
    return active


def unsupported_scoring_rules(scoring_settings: Mapping[str, object], roster_positions: Iterable[str] | None = None) -> list[str]:
    """Return unmodelled rules that can affect a position used by this league.

    The V1 board intentionally contains QB/RB/WR/TE only.  A league can retain
    dormant DEF/K/IDP scoring settings even when it has no corresponding roster
    slot, so those irrelevant rules should not create a warning for the user.
    Omitting ``roster_positions`` preserves the full diagnostic list for callers
    that need to inspect all settings.
    """
    active = _active_roster_positions(roster_positions or []) if roster_positions is not None else None
    rules = []
    for key in numeric_rules(scoring_settings):
        if key in SUPPORTED_STAT_KEYS:
            continue
        position = _rule_position(key)
        if active is not None and position is not None and position not in active:
            continue
        rules.append(key)
    return sorted(rules)


def _stat_value(stats: Mapping[str, object], key: str) -> float:
    """Return a direct Sleeper projection stat or a supported derived stat."""
    def numeric_stat(name: str) -> float:
        try:
            return float((stats or {}).get(name, 0) or 0)
        except (TypeError, ValueError):
            return 0.0

    if key == "pass_inc":
        # Sleeper projects attempts and completions, but does not expose a
        # separate incompletion field.  Sleeper's pass_inc scoring rule is
        # therefore calculated from the two available counting projections.
        return max(0.0, numeric_stat("pass_att") - numeric_stat("pass_cmp"))
    return numeric_stat(key)


def score_projection(stats: Mapping[str, object], scoring_settings: Mapping[str, object]) -> ScoringResult:
    rules = numeric_rules(scoring_settings)
    applied: dict[str, float] = {}
    points = 0.0
    for key, multiplier in rules.items():
        if key not in SUPPORTED_STAT_KEYS:
            continue
        stat = _stat_value(stats, key)
        contribution = stat * multiplier
        applied[key] = contribution
        points += contribution
    return ScoringResult(points=points, applied_rules=applied, unsupported_rules=unsupported_scoring_rules(scoring_settings))
