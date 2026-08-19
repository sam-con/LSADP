from __future__ import annotations

import pandas as pd

from src.history_library import (
    build_league_environment_from_library,
    build_league_position_scoring_profiles,
    build_position_history_library,
    match_position_scoring_environment,
)


def donor_configuration_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for season, league_id, scoring_format in [
        (2022, "half2022", "half_ppr"),
        (2023, "half2023", "half_ppr"),
        (2024, "half2024", "half_ppr"),
        (2025, "half2025", "half_ppr"),
        (2022, "2022", "ppr"),
        (2023, "2023", "ppr"),
        (2024, "2024", "ppr"),
        (2025, "2025", "ppr"),
    ]:
        rows.append({"season": season, "scoring_format": scoring_format, "league_id": league_id, "selected": True})
    return pd.DataFrame(rows)


def test_passing_td_changes_only_qb_signature() -> None:
    base = build_league_position_scoring_profiles({"pass_td": 4.0, "rec": 1.0})
    changed = build_league_position_scoring_profiles({"pass_td": 6.0, "rec": 1.0})

    assert base["QB"]["position_scoring_hash"] != changed["QB"]["position_scoring_hash"]
    assert base["RB"]["position_scoring_hash"] == changed["RB"]["position_scoring_hash"]
    assert base["WR"]["position_scoring_hash"] == changed["WR"]["position_scoring_hash"]
    assert base["TE"]["position_scoring_hash"] == changed["TE"]["position_scoring_hash"]


def test_reception_scoring_changes_only_skill_positions() -> None:
    base = build_league_position_scoring_profiles({"pass_td": 4.0, "rec": 0.5})
    changed = build_league_position_scoring_profiles({"pass_td": 4.0, "rec": 1.0})

    assert base["QB"]["position_scoring_hash"] == changed["QB"]["position_scoring_hash"]
    assert base["RB"]["position_scoring_hash"] != changed["RB"]["position_scoring_hash"]
    assert base["WR"]["position_scoring_hash"] != changed["WR"]["position_scoring_hash"]
    assert base["TE"]["position_scoring_hash"] != changed["TE"]["position_scoring_hash"]


def test_te_premium_changes_only_te_signature() -> None:
    base = build_league_position_scoring_profiles({"pass_td": 4.0, "rec": 1.0})
    changed = build_league_position_scoring_profiles({"pass_td": 4.0, "rec": 1.0, "rec_te": 0.5})

    assert base["QB"]["position_scoring_hash"] == changed["QB"]["position_scoring_hash"]
    assert base["RB"]["position_scoring_hash"] == changed["RB"]["position_scoring_hash"]
    assert base["WR"]["position_scoring_hash"] == changed["WR"]["position_scoring_hash"]
    assert base["TE"]["position_scoring_hash"] != changed["TE"]["position_scoring_hash"]


def test_signature_normalization_treats_numeric_equivalents_as_equal() -> None:
    left = build_league_position_scoring_profiles({"pass_td": 4, "rec": 1})
    right = build_league_position_scoring_profiles({"rec": 1.0, "pass_td": 4.0})

    assert left["QB"]["position_scoring_hash"] == right["QB"]["position_scoring_hash"]
    assert left["RB"]["position_scoring_hash"] == right["RB"]["position_scoring_hash"]


def test_library_merges_rb_history_even_when_qb_scoring_differs(mock_client) -> None:
    mock_client.leagues["pass6_2025"] = {
        **mock_client.leagues["2025"],
        "league_id": "pass6_2025",
        "scoring_settings": {"rec": 1.0, "pass_td": 6.0},
    }
    for week in range(1, 15):
        mock_client.matchups[("pass6_2025", week)] = mock_client.matchups[("2025", week)]

    library = build_position_history_library(
        client=mock_client,
        seed_leagues=pd.DataFrame(
            [
                {"league_id": "2025", "season": 2025, "selected": True},
                {"league_id": "pass6_2025", "season": 2025, "selected": True},
            ]
        ),
    )

    environments = library["position_scoring_environments"]
    assert len(environments[environments["position"] == "QB"]) == 2
    assert len(environments[environments["position"] == "RB"]) == 1

    rb_weeks = library["player_weeks"][library["player_weeks"]["position"] == "RB"]
    assert int(rb_weeks["confirmation_count"].max()) == 2


def test_position_matching_uses_only_relevant_scoring_fields(mock_client) -> None:
    library = build_position_history_library(
        client=mock_client,
        seed_leagues=donor_configuration_frame()[["league_id", "season", "selected"]],
    )
    mock_client.leagues["tep_target"] = {
        **mock_client.leagues["2026"],
        "league_id": "tep_target",
        "previous_league_id": None,
        "scoring_settings": {"rec": 1.0, "pass_td": 4.0, "rec_te": 0.5},
    }

    te_match = match_position_scoring_environment("TE", mock_client.leagues["tep_target"]["scoring_settings"], library["position_scoring_environments"])
    wr_match = match_position_scoring_environment("WR", mock_client.leagues["tep_target"]["scoring_settings"], library["position_scoring_environments"])

    assert te_match.match_quality in {"Very Close", "Approximate"}
    assert any(diff["key"] == "rec_te" for diff in te_match.differing_fields)
    assert wr_match.match_quality == "Exact"


def test_brand_new_league_builds_environment_from_library(mock_client) -> None:
    library = build_position_history_library(
        client=mock_client,
        seed_leagues=donor_configuration_frame()[["league_id", "season", "selected"]],
    )
    mock_client.leagues["brand_new"] = {
        **mock_client.leagues["2026"],
        "league_id": "brand_new",
        "name": "Brand New",
        "previous_league_id": None,
        "total_rosters": 14,
    }
    league = mock_client.get_league("brand_new")

    environment = build_league_environment_from_library(league, library)

    assert environment["public_runtime_mode"] == "position_history_library"
    assert environment["historical_source"] == "position_history_library"
    assert len(environment["position_match_summary"]) == 4
    assert environment["replacement"]["replacement_rank"].notna().all()
    assert environment["vorp_table"]["vorp"].notna().all()
