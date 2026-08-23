"""Score Sleeper projection statistics with arbitrary supported Sleeper rules."""

from __future__ import annotations

from collections.abc import Mapping

from .models import ScoringResult

# These are counting statistics present in Sleeper's season-projection payload.
# A rule is only modeled when the key has a direct, documented projection stat.
SUPPORTED_STAT_KEYS = frozenset(
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


def unsupported_scoring_rules(scoring_settings: Mapping[str, object]) -> list[str]:
    """Rules requiring event distributions/ranges are intentionally not guessed."""
    return sorted(key for key in numeric_rules(scoring_settings) if key not in SUPPORTED_STAT_KEYS)


def score_projection(stats: Mapping[str, object], scoring_settings: Mapping[str, object]) -> ScoringResult:
    rules = numeric_rules(scoring_settings)
    applied: dict[str, float] = {}
    points = 0.0
    for key, multiplier in rules.items():
        if key not in SUPPORTED_STAT_KEYS:
            continue
        try:
            stat = float((stats or {}).get(key, 0) or 0)
        except (TypeError, ValueError):
            stat = 0.0
        contribution = stat * multiplier
        applied[key] = contribution
        points += contribution
    return ScoringResult(points=points, applied_rules=applied, unsupported_rules=unsupported_scoring_rules(scoring_settings))
