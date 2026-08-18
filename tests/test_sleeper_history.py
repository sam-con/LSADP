from __future__ import annotations

from datetime import date

import pytest

from src.models import ScoringConsistencyError
from src.sleeper_history import (
    compare_scoring_across_history,
    load_league_chain,
    select_required_history,
    validate_scoring_consistency,
)


def test_historical_league_chain_terminates_safely(mock_client) -> None:
    chain = load_league_chain(mock_client, "2026")
    assert [league.season for league in chain] == [2026, 2025, 2024, 2023, 2022]


def test_exact_previous_four_completed_seasons_are_selected(mock_client) -> None:
    chain = load_league_chain(mock_client, "2026")
    selected = select_required_history(chain, today=date(2026, 8, 18))
    assert [league.season for league in selected] == [2022, 2023, 2024, 2025]


def test_identical_scoring_dicts_with_different_key_order_pass_validation(mock_client) -> None:
    current = mock_client.get_league("2026")
    historical = select_required_history(load_league_chain(mock_client, "2026"), today=date(2026, 8, 18))
    validate_scoring_consistency(current, historical)


def test_material_scoring_setting_changes_fail_validation(mock_client) -> None:
    mock_client.leagues["2024"]["scoring_settings"] = {"pass_td": 6.0, "rec": 1.0, "rec_te": 1.0}
    current = mock_client.get_league("2026")
    historical = select_required_history(load_league_chain(mock_client, "2026"), today=date(2026, 8, 18))

    with pytest.raises(ScoringConsistencyError):
        validate_scoring_consistency(current, historical)


def test_missing_required_historical_seasons_fail_validation(mock_client) -> None:
    del mock_client.leagues["2022"]
    chain = load_league_chain(mock_client, "2026")
    with pytest.raises(ScoringConsistencyError):
        select_required_history(chain, today=date(2026, 8, 18))

