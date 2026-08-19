from __future__ import annotations

import pytest

from src.canonical import (
    canonical_environment_key_for_league,
    classify_transformation_type,
    directed_transform_pairs,
    validate_canonical_configuration,
)
from src.models import ConfigError
from src.sleeper import parse_league_settings


def test_all_four_canonical_environments_are_required(canonical_adp_paths) -> None:
    incomplete = {
        "1qb_half_ppr": "half2026",
    }
    with pytest.raises(ConfigError):
        validate_canonical_configuration(incomplete, canonical_adp_paths)


def test_1qb_ppr_selects_1qb_ppr() -> None:
    league = parse_league_settings(
        {
            "league_id": "1",
            "name": "PPR",
            "season": "2026",
            "total_rosters": 12,
            "scoring_settings": {"rec": 1.0, "pass_td": 4},
            "roster_positions": ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "BN"],
        }
    )
    assert canonical_environment_key_for_league(league) == "1qb_ppr"


def test_sf_half_ppr_selects_sf_half_ppr() -> None:
    league = parse_league_settings(
        {
            "league_id": "1",
            "name": "SF Half",
            "season": "2026",
            "total_rosters": 12,
            "scoring_settings": {"rec": 0.5, "pass_td": 4},
            "roster_positions": ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "SUPER_FLEX", "BN"],
        }
    )
    assert canonical_environment_key_for_league(league) == "sf_half_ppr"


def test_1qb_standard_selects_nearest_active_half_ppr_anchor() -> None:
    league = parse_league_settings(
        {
            "league_id": "1",
            "name": "Standard",
            "season": "2026",
            "total_rosters": 12,
            "scoring_settings": {"rec": 0.0, "pass_td": 4},
            "roster_positions": ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "BN"],
        }
    )
    assert canonical_environment_key_for_league(league) == "1qb_half_ppr"


def test_unusual_reception_scoring_selects_nearest_canonical_environment() -> None:
    league = parse_league_settings(
        {
            "league_id": "1",
            "name": "0.75 PPR",
            "season": "2026",
            "total_rosters": 12,
            "scoring_settings": {"rec": 0.75, "pass_td": 4},
            "roster_positions": ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "BN"],
        }
    )
    assert canonical_environment_key_for_league(league) == "1qb_half_ppr"


def test_te_premium_does_not_change_base_reception_classification() -> None:
    league = parse_league_settings(
        {
            "league_id": "1",
            "name": "TE Premium",
            "season": "2026",
            "total_rosters": 12,
            "scoring_settings": {"rec": 0.0, "rec_te": 1.5, "pass_td": 4},
            "roster_positions": ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "BN"],
        }
    )
    assert canonical_environment_key_for_league(league) == "1qb_half_ppr"


def test_four_environments_generate_exactly_twelve_directed_tests() -> None:
    pairs = directed_transform_pairs()
    assert len(pairs) == 12
    assert all(source != target for source, target in pairs)


def test_transformation_categories_are_classified_correctly() -> None:
    assert classify_transformation_type("1qb_ppr", "1qb_half_ppr") == "Scoring-only"
    assert classify_transformation_type("1qb_ppr", "sf_ppr") == "Scarcity-only"
    assert classify_transformation_type("1qb_ppr", "sf_half_ppr") == "Combined"
