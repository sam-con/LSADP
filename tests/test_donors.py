from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.donors import load_history_seed_leagues
from src.model_builder import build_candidate_model
from src.models import ConfigError


def donor_configuration_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for season, league_id in [
        (2022, "half2022"),
        (2023, "half2023"),
        (2024, "half2024"),
        (2025, "half2025"),
        (2022, "2022"),
        (2023, "2023"),
        (2024, "2024"),
        (2025, "2025"),
    ]:
        rows.append({"season": season, "league_id": league_id, "selected": True})
    return pd.DataFrame(rows)


def test_load_history_seed_leagues_parses_csv_and_deduplicates(tmp_path) -> None:
    donor_path = tmp_path / "donor_leagues.csv"
    donor_path.write_text(
        "\n".join(
            [
                "league_id,season,selected",
                "half2022,2022,true",
                "half2022,2022,true",
                "half2023,2023,true",
                "half2024,2024,true",
                "half2025,2025,true",
                "2022,2022,true",
                "2023,2023,true",
                "2024,2024,true",
                "2025,2025,true",
                "future,2027,true",
                "inactive,2025,false",
            ]
        ),
        encoding="utf-8",
    )

    loaded, metadata = load_history_seed_leagues(donor_path=donor_path, today=date(2026, 8, 19))

    assert len(loaded) == 8
    assert set(loaded["season"]) == {2022, 2023, 2024, 2025}
    assert metadata["selected_rows"] == 8
    assert metadata["unique_leagues"] == 8


def test_load_history_seed_leagues_supports_headerless_tsv_export(tmp_path) -> None:
    donor_path = tmp_path / "donor_leagues.csv"
    donor_path.write_text(
        "\n".join(
            [
                'half2022\t2022\taccepted\tGood coverage\tTRUE',
                'half2023\t2023\taccepted\tGood coverage\tTRUE',
                'half2024\t2024\taccepted\tGood coverage\tTRUE',
                'half2025\t2025\taccepted\tGood coverage\tTRUE',
                '2022\t2022\taccepted\tGood coverage\tTRUE',
                '2023\t2023\taccepted\tGood coverage\tTRUE',
                '2024\t2024\taccepted\tGood coverage\tTRUE',
                '2025\t2025\taccepted\tGood coverage\tTRUE',
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="must contain `league_id` and `season` columns"):
        load_history_seed_leagues(donor_path=donor_path, today=date(2026, 8, 19))


def test_load_history_seed_leagues_supports_year_keyed_json(tmp_path) -> None:
    donor_path = tmp_path / "historical_donors_by_year.json"
    donor_path.write_text(
        """
        {
          "2022": {"half_ppr": [{"league_id": "half2022"}], "ppr": ["2022"]},
          "2023": {"half_ppr": ["half2023"], "ppr": ["2023"]},
          "2024": {"half_ppr": ["half2024"], "ppr": ["2024"]},
          "2025": {"half_ppr": ["half2025"], "ppr": ["2025"]}
        }
        """.strip(),
        encoding="utf-8",
    )

    loaded, metadata = load_history_seed_leagues(donor_path=donor_path, today=date(2026, 8, 19))

    assert len(loaded) == 8
    assert set(loaded["season"]) == {2022, 2023, 2024, 2025}
    assert metadata["source"].endswith("historical_donors_by_year.json")


def test_empty_donor_source_fails_clearly(tmp_path) -> None:
    donor_path = tmp_path / "donor_leagues.csv"
    donor_path.write_text("", encoding="utf-8")

    with pytest.raises(ConfigError, match="empty"):
        load_history_seed_leagues(donor_path=donor_path, today=date(2026, 8, 19))


def test_candidate_model_uses_seed_history_source(mock_client, canonical_league_ids, canonical_adp_paths) -> None:
    bundle = build_candidate_model(
        mock_client,
        canonical_leagues=canonical_league_ids,
        canonical_adp_paths=canonical_adp_paths,
        donor_configuration=donor_configuration_frame(),
        today=date(2026, 8, 19),
    )

    assert len(bundle["selected_validation"]) == 12
    assert bundle["metadata"]["historical_donor_source"]["source"] == "provided_dataframe"
    assert not bundle["history_position_environments"].empty
