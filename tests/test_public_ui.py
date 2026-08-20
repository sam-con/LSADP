from __future__ import annotations

import pandas as pd

from src.config import SHOW_DEVELOPMENT_PAGE
from src.public_ui import (
    build_historical_match_details,
    build_historical_reference_frame,
    build_position_impact_frame,
    build_public_download_frame,
    build_public_rankings_frame,
    league_format_label,
    modeled_positions_for_league,
    public_methodology_lines,
    scoring_primary_label,
    scoring_summary_text,
    starting_lineup_text,
    unsupported_roster_positions,
)
from src.sleeper import parse_league_settings


def sample_league():
    return parse_league_settings(
        {
            "league_id": "sf-1",
            "name": "Superflex PPR League",
            "season": "2026",
            "total_rosters": 12,
            "scoring_settings": {"rec": 1.0, "pass_td": 4.0, "rec_te": 0.5},
            "roster_positions": ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "SUPER_FLEX", "K", "DEF", "BN"],
        }
    )


def sample_results() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "adjusted_rank": 1,
                "player_name": "QB Alpha",
                "position": "QB",
                "team": "BUF",
                "adp": 10.2,
                "pos_rank": 1,
                "league_adjusted_adp": 7.9,
                "adp_change": 2.3,
                "canonical_metric": 5.0,
                "league_metric": 7.0,
                "canonical_expected_ppg": 22.0,
                "league_expected_ppg": 23.2,
                "canonical_vorp": 5.5,
                "league_vorp": 7.0,
                "delta_metric": 1.5,
                "short_explanation": "internal text",
                "player_id": "Q1",
            },
            {
                "adjusted_rank": 2,
                "player_name": "RB Beta",
                "position": "RB",
                "team": "SF",
                "adp": 18.4,
                "pos_rank": 2,
                "league_adjusted_adp": 21.1,
                "adp_change": -2.7,
                "canonical_metric": 8.0,
                "league_metric": 6.0,
                "canonical_expected_ppg": 17.1,
                "league_expected_ppg": 16.0,
                "canonical_vorp": 8.4,
                "league_vorp": 6.9,
                "delta_metric": -1.5,
                "short_explanation": "internal text",
                "player_id": "R1",
            },
            {
                "adjusted_rank": 3,
                "player_name": "WR Gamma",
                "position": "WR",
                "team": "MIN",
                "adp": 29.0,
                "pos_rank": 3,
                "league_adjusted_adp": 25.0,
                "adp_change": 4.0,
                "canonical_metric": 4.0,
                "league_metric": 5.5,
                "canonical_expected_ppg": 15.0,
                "league_expected_ppg": 15.9,
                "canonical_vorp": 3.7,
                "league_vorp": 5.1,
                "delta_metric": 1.4,
                "short_explanation": "internal text",
                "player_id": "W1",
            },
        ]
    )


def sample_match_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "position": "QB",
                "match_quality": "Exact",
                "status": "complete",
                "position_scoring_hash": "hash-qb",
                "differing_fields": [],
            },
            {
                "position": "TE",
                "match_quality": "Very Close",
                "status": "complete",
                "position_scoring_hash": "hash-te",
                "differing_fields": [
                    {"key": "rec_te", "target_value": 1.5, "matched_value": 1.0},
                ],
            },
        ]
    )


def test_modeled_positions_for_league_omits_absent_positions() -> None:
    positions = modeled_positions_for_league(sample_league(), sample_results())
    assert positions == ["QB", "RB", "WR"]


def test_unsupported_roster_positions_identifies_non_modeled_slots_once() -> None:
    unsupported = unsupported_roster_positions(sample_league())
    assert unsupported == ["DEF", "K"]


def test_public_rankings_frame_only_contains_user_facing_columns() -> None:
    frame = build_public_rankings_frame(sample_results(), positions=["QB", "WR"], search="a")
    assert list(frame.columns) == ["Rank", "Player", "Pos", "Team", "Pos Rank", "Market ADP", "League ADP", "Change"]
    assert "player_id" not in frame.columns
    assert "short_explanation" not in frame.columns
    assert set(frame["Pos"]) == {"QB", "WR"}


def test_public_download_frame_only_contains_user_facing_columns() -> None:
    frame = build_public_download_frame(sample_results())
    assert list(frame.columns) == ["Rank", "Player", "Position", "Team", "Market ADP", "League ADP", "Change"]
    assert "delta_metric" not in frame.columns
    assert "canonical_vorp" not in frame.columns


def test_league_summary_helpers_render_human_readable_superflex_ppr() -> None:
    league = sample_league()
    assert league_format_label(league) == "Superflex"
    assert scoring_primary_label(league) == "PPR"
    assert scoring_summary_text(league).startswith("PPR")
    assert "{" not in scoring_summary_text(league)
    assert starting_lineup_text(league) == "1 QB / 2 RB / 2 WR / 1 TE / 1 FLEX / 1 SUPERFLEX / 1 K / 1 DEF"


def test_historical_reference_frame_filters_to_relevant_positions_without_hashes() -> None:
    frame = build_historical_reference_frame(sample_match_summary(), ["QB"])
    assert list(frame.columns) == ["Position", "Historical Reference", "Coverage"]
    assert frame.to_dict(orient="records") == [
        {"Position": "QB", "Historical Reference": "Exact", "Coverage": "complete"}
    ]


def test_historical_match_details_use_plain_language() -> None:
    details = build_historical_match_details(sample_match_summary(), ["TE"])
    assert len(details) == 1
    assert details[0]["position"] == "TE"
    assert "tight ends" in details[0]["differences"][0].lower()
    assert "hash" not in details[0]["differences"][0].lower()


def test_position_impact_frame_uses_public_labels() -> None:
    frame = build_position_impact_frame(sample_results(), ["QB", "RB", "WR"])
    assert list(frame.columns) == ["Position", "Positional Value", "Avg Change"]
    assert "canonical_metric" not in frame.columns
    assert set(frame["Position"]) == {"QB", "RB", "WR"}


def test_public_methodology_avoids_internal_model_jargon() -> None:
    text = " ".join(public_methodology_lines()).lower()
    for banned in ("latent utility", "beta coefficient", "curve parameter", "objective function", "hash"):
        assert banned not in text


def test_development_page_visibility_flag_still_defaults_false() -> None:
    assert SHOW_DEVELOPMENT_PAGE is False
