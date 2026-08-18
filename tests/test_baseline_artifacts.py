from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.analysis import build_baseline_artifacts, load_current_market_context
from src.baseline_artifacts import BaselineArtifactManager
from src.config import SHOW_DEVELOPMENT_PAGE
from src.models import ConfigError


def test_saved_baseline_artifacts_can_be_loaded_without_network(tmp_path) -> None:
    manager = BaselineArtifactManager(
        curves_path=tmp_path / "curves.csv",
        replacement_path=tmp_path / "replacement.csv",
        model_path=tmp_path / "model.csv",
        metadata_path=tmp_path / "metadata.json",
    )
    manager.save(
        curves=pd.DataFrame([{"position": "QB", "rank": 1, "expected_ppg": 20.0, "dataset": "fitted"}]),
        replacement=pd.DataFrame([{"position": "QB", "replacement_rank": 13, "replacement_ppg": 10.0, "method": "Starter Demand Replacement"}]),
        model=pd.DataFrame([{"position": "QB", "metric_name": "vorp", "intercept": 0.0, "coefficient": 1.0, "r2": 1.0, "weighted_rmse": 0.0}]),
        metadata={
            "baseline_1qb_league_id": "2026",
            "baseline_superflex_league_id": "sf2026",
            "historical_seasons": [2022, 2023, 2024, 2025],
            "generated_timestamp": "2026-08-18T00:00:00",
            "model_version": "0.1.0",
        },
    )
    loaded = manager.load()
    assert loaded.metadata["baseline_1qb_league_id"] == "2026"


def test_missing_required_baseline_files_raise_clear_error(tmp_path) -> None:
    manager = BaselineArtifactManager(
        curves_path=tmp_path / "curves.csv",
        replacement_path=tmp_path / "replacement.csv",
        model_path=tmp_path / "model.csv",
        metadata_path=tmp_path / "metadata.json",
    )
    with pytest.raises(ConfigError):
        manager.validate()


def test_malformed_metadata_is_detected(tmp_path) -> None:
    manager = BaselineArtifactManager(
        curves_path=tmp_path / "curves.csv",
        replacement_path=tmp_path / "replacement.csv",
        model_path=tmp_path / "model.csv",
        metadata_path=tmp_path / "metadata.json",
    )
    (tmp_path / "curves.csv").write_text("position,rank,expected_ppg,dataset\nQB,1,20.0,fitted\n", encoding="utf-8")
    (tmp_path / "replacement.csv").write_text("position,replacement_rank,replacement_ppg,method\nQB,13,10.0,test\n", encoding="utf-8")
    (tmp_path / "model.csv").write_text("position,metric_name,intercept,coefficient,r2,weighted_rmse\nQB,vorp,0,1,1,0\n", encoding="utf-8")
    (tmp_path / "metadata.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ConfigError):
        manager.validate()


def test_rebuilding_baseline_artifacts_is_deterministic(tmp_path, mock_client, adp_frame) -> None:
    adp_path = tmp_path / "adp_1qb.csv"
    adp_frame.to_csv(adp_path, index=False)
    first = build_baseline_artifacts(mock_client, baseline_league_id="2026", current_adp_path=adp_path, known_superflex_league_id="sf2026", today=date(2026, 8, 18))
    second = build_baseline_artifacts(mock_client, baseline_league_id="2026", current_adp_path=adp_path, known_superflex_league_id="sf2026", today=date(2026, 8, 18))
    assert first["environment"]["replacement"].equals(second["environment"]["replacement"])
    assert first["calibration_model"].equals(second["calibration_model"])


def test_public_analysis_context_uses_saved_baseline_values(tmp_path, adp_frame) -> None:
    adp_path = tmp_path / "adp_1qb.csv"
    adp_frame.to_csv(adp_path, index=False)
    manager = BaselineArtifactManager(
        curves_path=tmp_path / "curves.csv",
        replacement_path=tmp_path / "replacement.csv",
        model_path=tmp_path / "model.csv",
        metadata_path=tmp_path / "metadata.json",
    )
    manager.save(
        curves=pd.DataFrame(
            [
                {"position": "QB", "rank": 1, "expected_ppg": 20.0, "dataset": "fitted"},
                {"position": "QB", "rank": 1, "expected_ppg": 20.0, "dataset": "empirical"},
                {"position": "RB", "rank": 1, "expected_ppg": 18.0, "dataset": "fitted"},
                {"position": "RB", "rank": 1, "expected_ppg": 18.0, "dataset": "empirical"},
                {"position": "WR", "rank": 1, "expected_ppg": 17.0, "dataset": "fitted"},
                {"position": "WR", "rank": 1, "expected_ppg": 17.0, "dataset": "empirical"},
                {"position": "TE", "rank": 1, "expected_ppg": 12.0, "dataset": "fitted"},
                {"position": "TE", "rank": 1, "expected_ppg": 12.0, "dataset": "empirical"},
            ]
        ),
        replacement=pd.DataFrame(
            [
                {"position": "QB", "replacement_rank": 1, "replacement_ppg": 20.0, "method": "Starter Demand Replacement"},
                {"position": "RB", "replacement_rank": 1, "replacement_ppg": 18.0, "method": "Starter Demand Replacement"},
                {"position": "WR", "replacement_rank": 1, "replacement_ppg": 17.0, "method": "Starter Demand Replacement"},
                {"position": "TE", "replacement_rank": 1, "replacement_ppg": 12.0, "method": "Starter Demand Replacement"},
            ]
        ),
        model=pd.DataFrame(
            [
                {"position": "QB", "metric_name": "vorp", "intercept": 0.0, "coefficient": 1.0, "r2": 1.0, "weighted_rmse": 0.0},
                {"position": "RB", "metric_name": "vorp", "intercept": 0.0, "coefficient": 1.0, "r2": 1.0, "weighted_rmse": 0.0},
                {"position": "WR", "metric_name": "vorp", "intercept": 0.0, "coefficient": 1.0, "r2": 1.0, "weighted_rmse": 0.0},
                {"position": "TE", "metric_name": "vorp", "intercept": 0.0, "coefficient": 1.0, "r2": 1.0, "weighted_rmse": 0.0},
            ]
        ),
        metadata={
            "baseline_1qb_league_id": "2026",
            "baseline_superflex_league_id": "sf2026",
            "historical_seasons": [2022, 2023, 2024, 2025],
            "generated_timestamp": "2026-08-18T00:00:00",
            "model_version": "0.1.0",
        },
    )
    context = load_current_market_context(manager, current_adp_path=adp_path)
    assert "baseline_players" in context
    assert not context["baseline_players"].empty


def test_development_page_visibility_respects_default_flag() -> None:
    assert SHOW_DEVELOPMENT_PAGE is False
