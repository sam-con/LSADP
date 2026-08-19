from __future__ import annotations

import json
from datetime import date

import pandas as pd
import pytest

from src.donors import (
    discover_historical_donors,
    load_saved_donor_configuration,
    save_donor_configuration,
    validate_donor_league,
)
from src.model_builder import build_candidate_model


def donor_configuration_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for season, league_id, scoring_format in [
        (2022, "std2022", "standard"),
        (2023, "std2023", "standard"),
        (2024, "std2024", "standard"),
        (2025, "std2025", "standard"),
        (2022, "half2022", "half_ppr"),
        (2023, "half2023", "half_ppr"),
        (2024, "half2024", "half_ppr"),
        (2025, "half2025", "half_ppr"),
        (2022, "2022", "ppr"),
        (2023, "2023", "ppr"),
        (2024, "2024", "ppr"),
        (2025, "2025", "ppr"),
    ]:
        rows.append(
            {
                "season": season,
                "scoring_format": scoring_format,
                "league_id": league_id,
                "status": "accepted",
                "reason": "Good coverage",
                "quality_score": 100.0,
                "weeks_loaded": 14,
                "unique_player_weeks": 1000,
                "unique_players": 40,
                "qb_coverage": 8,
                "rb_coverage": 12,
                "wr_coverage": 12,
                "te_coverage": 8,
                "team_count": 12,
                "league_name": f"{scoring_format} {season}",
                "scoring_signature": "{}",
                "selected": True,
            }
        )
    return pd.DataFrame(rows)


def test_discovery_deduplicates_graph_and_respects_limits(mock_client, canonical_league_ids) -> None:
    mock_client.users = {
        "seed": {"user_id": "u1", "username": "seed"},
        "u1": {"user_id": "u1", "username": "seed"},
        "u2": {"user_id": "u2", "username": "friend"},
    }
    mock_client.user_leagues = {
        ("u1", 2026): ["2026"],
        ("u1", 2022): [],
        ("u1", 2023): [],
        ("u1", 2024): [],
        ("u1", 2025): [],
        ("u2", 2026): ["sf2026"],
        ("u2", 2022): ["2022"],
        ("u2", 2023): ["2023"],
        ("u2", 2024): ["2024"],
        ("u2", 2025): ["2025"],
    }
    mock_client.league_users = {
        "2026": [{"user_id": "u1"}, {"user_id": "u2"}, {"user_id": "u2"}],
        "sf2026": [{"user_id": "u2"}],
        "2022": [{"user_id": "u2"}],
        "2023": [{"user_id": "u2"}],
        "2024": [{"user_id": "u2"}],
        "2025": [{"user_id": "u2"}],
    }

    result = discover_historical_donors(
        client=mock_client,
        canonical_leagues=canonical_league_ids,
        seed_user="seed",
        max_users=2,
        max_leagues=10,
        preferred_donors_per_cell=1,
        max_depth=2,
        today=date(2026, 8, 19),
    )

    assert result["crawl_stats"]["users_inspected"] <= 2
    assert result["crawl_stats"]["leagues_inspected"] <= 10
    assert result["accepted"]["league_id"].is_unique


def test_te_premium_donor_is_rejected(mock_client, canonical_league_ids) -> None:
    mock_client.leagues["teprem2025"] = {
        **mock_client.leagues["2025"],
        "league_id": "teprem2025",
        "scoring_settings": {"rec": 1.0, "pass_td": 4.0, "rec_te": 1.5},
    }
    mock_client.matchups.update({("teprem2025", week): mock_client.matchups[("2025", week)] for week in range(1, 15)})

    from src.donors import canonical_scoring_signatures

    signatures = canonical_scoring_signatures(canonical_league_ids, mock_client)
    result = validate_donor_league(
        mock_client,
        league_id="teprem2025",
        season=2025,
        scoring_format="ppr",
        expected_signature=signatures["ppr"],
    )

    assert result["status"] == "rejected"
    assert "Unusual scoring" in result["reason"]


def test_donor_configuration_persists_and_reloads(tmp_path) -> None:
    donors = donor_configuration_frame()
    metadata = {"required_seasons": [2022, 2023, 2024, 2025], "preferred_donors_per_cell": 3}
    donor_path = tmp_path / "donors.csv"
    metadata_path = tmp_path / "donors.json"

    save_donor_configuration(donors, donor_path=donor_path, metadata_path=metadata_path, metadata=metadata)
    loaded, loaded_metadata = load_saved_donor_configuration(donor_path=donor_path, metadata_path=metadata_path)

    assert loaded.equals(donors)
    assert loaded_metadata == metadata


def test_candidate_model_uses_donor_history_and_shares_scoring_curves(
    mock_client,
    canonical_league_ids,
    canonical_adp_paths,
) -> None:
    donors = donor_configuration_frame()

    bundle = build_candidate_model(
        mock_client,
        canonical_leagues=canonical_league_ids,
        canonical_adp_paths=canonical_adp_paths,
        donor_configuration=donors,
    )

    assert len(bundle["selected_validation"]) == 30
    assert bundle["metadata"]["historical_donor_source"]["source"] == "saved_donors"

    one_qb = bundle["curves"][
        (bundle["curves"]["environment_key"] == "1qb_ppr") & (bundle["curves"]["dataset"] == "fitted")
    ].sort_values(["position", "rank"]).reset_index(drop=True)
    sf = bundle["curves"][
        (bundle["curves"]["environment_key"] == "sf_ppr") & (bundle["curves"]["dataset"] == "fitted")
    ].sort_values(["position", "rank"]).reset_index(drop=True)

    pd.testing.assert_frame_equal(
        one_qb[["position", "rank", "expected_ppg"]].reset_index(drop=True),
        sf[["position", "rank", "expected_ppg"]].reset_index(drop=True),
    )
