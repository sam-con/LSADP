from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.donors import (
    discover_historical_donors,
    load_saved_donor_configuration,
    validate_donor_league,
    validate_historical_donor_configuration,
)
from src.model_builder import build_candidate_model
from src.models import ConfigError


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
        rows.append(
            {
                "season": season,
                "scoring_format": scoring_format,
                "league_id": league_id,
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
        ("u1", 2026): ["half2026"],
        ("u1", 2022): [],
        ("u1", 2023): [],
        ("u1", 2024): [],
        ("u1", 2025): [],
        ("u2", 2026): ["sf2026"],
        ("u2", 2022): ["2022", "half2022"],
        ("u2", 2023): ["2023", "half2023"],
        ("u2", 2024): ["2024", "half2024"],
        ("u2", 2025): ["2025", "half2025"],
    }
    mock_client.league_users = {
        "half2026": [{"user_id": "u1"}, {"user_id": "u2"}, {"user_id": "u2"}],
        "sf2026": [{"user_id": "u2"}],
        "2022": [{"user_id": "u2"}],
        "2023": [{"user_id": "u2"}],
        "2024": [{"user_id": "u2"}],
        "2025": [{"user_id": "u2"}],
        "half2022": [{"user_id": "u2"}],
        "half2023": [{"user_id": "u2"}],
        "half2024": [{"user_id": "u2"}],
        "half2025": [{"user_id": "u2"}],
    }

    result = discover_historical_donors(
        client=mock_client,
        canonical_leagues=canonical_league_ids,
        seed_user="seed",
        max_users=2,
        max_leagues=20,
        preferred_donors_per_cell=1,
        max_depth=2,
        today=date(2026, 8, 19),
    )

    assert result["crawl_stats"]["users_inspected"] <= 2
    assert result["crawl_stats"]["leagues_inspected"] <= 20
    assert result["accepted"]["league_id"].is_unique


def test_load_saved_donor_configuration_parses_schema_and_ignores_standard_duplicates(tmp_path) -> None:
    donor_path = tmp_path / "donor_leagues.csv"
    donor_path.write_text(
        "\n".join(
            [
                "league_id,season,format,selected",
                "std2022,2022,standard,true",
                "half2022,2022,half-ppr,true",
                "half2022,2022,half-ppr,true",
                "half2023,2023,half_ppr,true",
                "half2024,2024,half_ppr,true",
                "half2025,2025,half_ppr,true",
                "2022,2022,ppr,true",
                "2023,2023,ppr,true",
                "2024,2024,ppr,true",
                "2025,2025,ppr,true",
            ]
        ),
        encoding="utf-8",
    )

    loaded, metadata = load_saved_donor_configuration(donor_path=donor_path, today=date(2026, 8, 19))

    assert list(loaded["scoring_format"].unique()) == ["half_ppr", "ppr"]
    assert len(loaded) == 8
    assert metadata["ignored_standard_rows"] == 1
    assert metadata["duplicate_rows_removed"] == 1


def test_load_saved_donor_configuration_supports_headerless_tsv_export(tmp_path) -> None:
    donor_path = tmp_path / "donor_leagues.csv"
    donor_path.write_text(
        "\n".join(
            [
                'half2022\t2022\thalf_ppr\taccepted\tGood coverage\t100\t14\t1000\t100\t10\t20\t30\t10\t12\tHalf League\t"{}"\tTRUE\tseed1',
                'half2023\t2023\thalf_ppr\taccepted\tGood coverage\t100\t14\t1000\t100\t10\t20\t30\t10\t12\tHalf League\t"{}"\tTRUE\tseed1',
                'half2024\t2024\thalf_ppr\taccepted\tGood coverage\t100\t14\t1000\t100\t10\t20\t30\t10\t12\tHalf League\t"{}"\tTRUE\tseed1',
                'half2025\t2025\thalf_ppr\taccepted\tGood coverage\t100\t14\t1000\t100\t10\t20\t30\t10\t12\tHalf League\t"{}"\tTRUE\tseed1',
                '2022\t2022\tppr\taccepted\tGood coverage\t100\t14\t1000\t100\t10\t20\t30\t10\t12\tPPR League\t"{}"\tTRUE\tseed1',
                '2023\t2023\tppr\taccepted\tGood coverage\t100\t14\t1000\t100\t10\t20\t30\t10\t12\tPPR League\t"{}"\tTRUE\tseed1',
                '2024\t2024\tppr\taccepted\tGood coverage\t100\t14\t1000\t100\t10\t20\t30\t10\t12\tPPR League\t"{}"\tTRUE\tseed1',
                '2025\t2025\tppr\taccepted\tGood coverage\t100\t14\t1000\t100\t10\t20\t30\t10\t12\tPPR League\t"{}"\tTRUE\tseed1',
            ]
        ),
        encoding="utf-8",
    )

    loaded, metadata = load_saved_donor_configuration(donor_path=donor_path, today=date(2026, 8, 19))

    assert len(loaded) == 8
    assert set(loaded["scoring_format"]) == {"half_ppr", "ppr"}
    assert metadata["active_rows"] == 8


def test_load_saved_donor_configuration_supports_year_keyed_json(tmp_path) -> None:
    donor_path = tmp_path / "historical_donors_by_year.json"
    donor_path.write_text(
        """
        {
          "2022": {
            "half_ppr": [{"league_id": "half2022", "league_name": "Half 2022"}],
            "ppr": [{"league_id": "2022", "league_name": "PPR 2022"}]
          },
          "2023": {
            "half_ppr": ["half2023"],
            "ppr": ["2023"]
          },
          "2024": {
            "half_ppr": ["half2024"],
            "ppr": ["2024"]
          },
          "2025": {
            "half_ppr": ["half2025"],
            "ppr": ["2025"]
          }
        }
        """.strip(),
        encoding="utf-8",
    )

    loaded, metadata = load_saved_donor_configuration(donor_path=donor_path, today=date(2026, 8, 19))

    assert len(loaded) == 8
    assert set(loaded["scoring_format"]) == {"half_ppr", "ppr"}
    assert metadata["source"].endswith("historical_donors_by_year.json")


def test_empty_donor_csv_fails_clearly(tmp_path) -> None:
    donor_path = tmp_path / "donor_leagues.csv"
    donor_path.write_text("", encoding="utf-8")

    with pytest.raises(ConfigError, match="empty"):
        load_saved_donor_configuration(donor_path=donor_path, today=date(2026, 8, 19))


def test_missing_required_half_ppr_or_ppr_cell_fails_clearly(tmp_path) -> None:
    donor_path = tmp_path / "donor_leagues.csv"
    donor_path.write_text(
        "\n".join(
            [
                "league_id,season,format",
                "half2022,2022,half_ppr",
                "half2023,2023,half_ppr",
                "half2024,2024,half_ppr",
                "half2025,2025,half_ppr",
                "2022,2022,ppr",
                "2023,2023,ppr",
                "2024,2024,ppr",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="2025 ppr"):
        load_saved_donor_configuration(donor_path=donor_path, today=date(2026, 8, 19))


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


def test_donor_season_mismatch_is_rejected(mock_client, canonical_league_ids) -> None:
    from src.donors import canonical_scoring_signatures

    signatures = canonical_scoring_signatures(canonical_league_ids, mock_client)
    result = validate_donor_league(
        mock_client,
        league_id="2025",
        season=2024,
        scoring_format="ppr",
        expected_signature=signatures["ppr"],
    )

    assert result["status"] == "rejected"
    assert result["reason"] == "Wrong season"


def test_missing_players_points_rejects_donor(mock_client, canonical_league_ids) -> None:
    for week in range(1, 15):
        mock_client.matchups[("2025", week)] = [{"roster_id": 1, "players_points": {}, "starters": []}]

    from src.donors import canonical_scoring_signatures

    signatures = canonical_scoring_signatures(canonical_league_ids, mock_client)
    result = validate_donor_league(
        mock_client,
        league_id="2025",
        season=2025,
        scoring_format="ppr",
        expected_signature=signatures["ppr"],
    )

    assert result["status"] == "rejected"
    assert "could not be loaded" in result["reason"]


def test_invalid_donor_does_not_break_other_cells(mock_client, canonical_league_ids) -> None:
    donors = donor_configuration_frame()
    donors.loc[(donors["season"] == 2025) & (donors["scoring_format"] == "ppr"), "league_id"] = "bad2025"
    mock_client.leagues["bad2025"] = {
        **mock_client.leagues["2025"],
        "league_id": "bad2025",
        "scoring_settings": {"rec": 1.0, "pass_td": 4.0, "rec_te": 1.5},
    }
    mock_client.matchups.update({("bad2025", week): mock_client.matchups[("2025", week)] for week in range(1, 15)})

    result = validate_historical_donor_configuration(
        mock_client,
        canonical_league_ids,
        donor_configuration=donors,
        today=date(2026, 8, 19),
    )

    assert not result["accepted"].empty
    assert (2025, "ppr") in result["missing_cells"]
    assert "bad2025" in set(result["rejected"]["league_id"])


def test_candidate_model_uses_donor_history_and_runs_twelve_validations(
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
        today=date(2026, 8, 19),
    )

    assert len(bundle["selected_validation"]) == 12
    assert bundle["metadata"]["historical_donor_source"]["source"] == "provided_dataframe"

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
