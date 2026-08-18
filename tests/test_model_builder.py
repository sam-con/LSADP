from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.baseline_artifacts import CanonicalArtifactManager
from src.replacement import calculate_starter_demand_replacement
from src.model_builder import (
    build_candidate_model,
    predict_between_canonical_environments,
    promote_candidate_model,
    save_candidate_model,
    validate_environment_identity,
)
from src.models import ConfigError


def _manager(root: Path, name: str) -> CanonicalArtifactManager:
    base = root / name
    return CanonicalArtifactManager(
        curves_path=base / "canonical_curves.csv",
        replacement_path=base / "canonical_replacement.csv",
        market_calibration_path=base / "canonical_market_calibration.csv",
        model_parameters_path=base / "model_parameters.csv",
        validation_path=base / "model_validation.csv",
        canonical_config_path=base / "canonical_leagues.json",
        metadata_path=base / "baseline_metadata.json",
    )


def test_invalid_canonical_league_identity_fails_clearly(mock_client) -> None:
    environment = {"league": mock_client.get_league("2026")}
    with pytest.raises(ConfigError):
        validate_environment_identity("1qb_standard", environment["league"])


def test_candidate_build_generates_exactly_thirty_directed_validations(
    mock_client,
    canonical_league_ids,
    canonical_adp_paths,
) -> None:
    bundle = build_candidate_model(mock_client, canonical_leagues=canonical_league_ids, canonical_adp_paths=canonical_adp_paths)
    selected = bundle["selected_validation"]
    assert len(selected) == 30
    assert set(selected["transform_type"]) == {"Scoring-only", "Scarcity-only", "Combined"}


def test_candidate_build_does_not_overwrite_production_files(
    tmp_path,
    mock_client,
    canonical_league_ids,
    canonical_adp_paths,
) -> None:
    candidate_manager = _manager(tmp_path, "candidate")
    production_manager = _manager(tmp_path, "production")
    bundle = build_candidate_model(mock_client, canonical_leagues=canonical_league_ids, canonical_adp_paths=canonical_adp_paths)
    save_candidate_model(candidate_manager, bundle)
    assert candidate_manager.curves_path.exists()
    assert not production_manager.curves_path.exists()


def test_failed_validation_cannot_promote(tmp_path) -> None:
    candidate_manager = _manager(tmp_path, "candidate")
    production_manager = _manager(tmp_path, "production")
    candidate_manager.save(
        curves=pd.DataFrame([{"environment_key": "1qb_ppr", "position": "QB", "rank": 1, "expected_ppg": 20.0, "dataset": "fitted"}]),
        replacement=pd.DataFrame([{"environment_key": "1qb_ppr", "position": "QB", "replacement_method": "starter_demand", "replacement_rank": 13, "replacement_ppg": 10.0, "method": "Starter Demand Replacement"}]),
        market_calibration=pd.DataFrame([{"environment_key": "1qb_ppr", "position": "QB", "model_name": "Curve + Starter VORP", "metric_name": "vorp", "intercept": 0.0, "coefficient": 1.0, "utility_transform": "neg_log", "weight_power": 1.0, "replacement_method": "starter_demand"}]),
        model_parameters=pd.DataFrame([{"spec_id": "spec_01", "model_name": "Curve + Starter VORP", "composite_score": 10.0}]),
        validation=pd.DataFrame([{"spec_id": "spec_01", "source_environment": "1qb_ppr", "target_environment": "sf_ppr", "weighted_mae": 1.0}]),
        canonical_config={"1qb_ppr": {"label": "1QB PPR"}},
        metadata={
            "selected_model_name": "Curve + Starter VORP",
            "selected_utility_transform": "neg_log",
            "selected_weight_power": 1.0,
            "generated_timestamp": "2026-08-18T00:00:00+00:00",
            "model_version": "0.1.0",
            "canonical_environments": {"1qb_ppr": {"label": "1QB PPR"}},
            "validation_complete": False,
            "selected_model_score": 10.0,
        },
    )
    with pytest.raises(ConfigError):
        promote_candidate_model(candidate_manager, production_manager)


def test_successful_promotion_writes_expected_artifacts(
    tmp_path,
    mock_client,
    canonical_league_ids,
    canonical_adp_paths,
) -> None:
    candidate_manager = _manager(tmp_path, "candidate")
    production_manager = _manager(tmp_path, "production")
    bundle = build_candidate_model(mock_client, canonical_leagues=canonical_league_ids, canonical_adp_paths=canonical_adp_paths)
    save_candidate_model(candidate_manager, bundle)
    promote_candidate_model(candidate_manager, production_manager)
    production = production_manager.load()
    assert production.metadata["selected_model_name"] == bundle["metadata"]["selected_model_name"]
    assert not production.validation.empty


def test_saved_model_reloads_deterministically(
    tmp_path,
    mock_client,
    canonical_league_ids,
    canonical_adp_paths,
) -> None:
    candidate_manager = _manager(tmp_path, "candidate")
    bundle = build_candidate_model(mock_client, canonical_leagues=canonical_league_ids, canonical_adp_paths=canonical_adp_paths)
    save_candidate_model(candidate_manager, bundle)
    first = candidate_manager.load()
    second = candidate_manager.load()
    assert first.metadata == second.metadata
    assert first.model_parameters.equals(second.model_parameters)


def test_increasing_team_count_pushes_replacement_deeper(mock_client) -> None:
    league = mock_client.get_league("2026")
    curves = pd.DataFrame(
        [
            {"position": "QB", "rank": rank, "expected_ppg": 25 - rank}
            for rank in range(1, 25)
        ]
        + [{"position": "RB", "rank": rank, "expected_ppg": 21 - 0.7 * rank} for rank in range(1, 40)]
        + [{"position": "WR", "rank": rank, "expected_ppg": 20 - 0.6 * rank} for rank in range(1, 40)]
        + [{"position": "TE", "rank": rank, "expected_ppg": 15 - 0.5 * rank} for rank in range(1, 25)]
    )
    larger = mock_client.get_league("2026")
    smaller = mock_client.get_league("2026")
    larger.total_rosters = 14
    smaller.total_rosters = 10
    larger_replacement = calculate_starter_demand_replacement(larger, curves).set_index("position")
    smaller_replacement = calculate_starter_demand_replacement(smaller, curves).set_index("position")
    assert int(larger_replacement.loc["QB", "replacement_rank"]) > int(smaller_replacement.loc["QB", "replacement_rank"])


def test_directionality_1qb_to_sf_materially_raises_qb_relative_value(
    mock_client,
    canonical_league_ids,
    canonical_adp_paths,
) -> None:
    bundle = build_candidate_model(mock_client, canonical_leagues=canonical_league_ids, canonical_adp_paths=canonical_adp_paths)
    source_adp = bundle["source_adp_by_environment"]["1qb_ppr"]
    prediction = predict_between_canonical_environments(
        "1qb_ppr",
        "sf_ppr",
        bundle["environment_bundle"]["1qb_ppr"],
        bundle["environment_bundle"]["sf_ppr"],
        source_adp,
        {"model_name": "Curve + Starter VORP", "metric_mode": "vorp", "replacement_method": "starter_demand", "utility_transform": "neg_log", "weight_power": 1.0},
    )
    qb_change = prediction[prediction["position"] == "QB"]["adp_change"].mean()
    rb_change = prediction[prediction["position"] == "RB"]["adp_change"].mean()
    assert qb_change > rb_change


def test_directionality_sf_to_1qb_lowers_qb_relative_value(
    mock_client,
    canonical_league_ids,
    canonical_adp_paths,
) -> None:
    bundle = build_candidate_model(mock_client, canonical_leagues=canonical_league_ids, canonical_adp_paths=canonical_adp_paths)
    source_adp = bundle["source_adp_by_environment"]["sf_ppr"]
    prediction = predict_between_canonical_environments(
        "sf_ppr",
        "1qb_ppr",
        bundle["environment_bundle"]["sf_ppr"],
        bundle["environment_bundle"]["1qb_ppr"],
        source_adp,
        {"model_name": "Curve + Starter VORP", "metric_mode": "vorp", "replacement_method": "starter_demand", "utility_transform": "neg_log", "weight_power": 1.0},
    )
    qb_change = prediction[prediction["position"] == "QB"]["adp_change"].mean()
    rb_change = prediction[prediction["position"] == "RB"]["adp_change"].mean()
    assert qb_change < rb_change


def test_standard_to_ppr_behaves_directionally_sensibly(
    mock_client,
    canonical_league_ids,
    canonical_adp_paths,
) -> None:
    bundle = build_candidate_model(mock_client, canonical_leagues=canonical_league_ids, canonical_adp_paths=canonical_adp_paths)
    source_adp = bundle["source_adp_by_environment"]["1qb_standard"]
    prediction = predict_between_canonical_environments(
        "1qb_standard",
        "1qb_ppr",
        bundle["environment_bundle"]["1qb_standard"],
        bundle["environment_bundle"]["1qb_ppr"],
        source_adp,
        {"model_name": "Curve Only", "metric_mode": "expected_ppg", "replacement_method": "starter_demand", "utility_transform": "neg_log", "weight_power": 1.0},
    )
    wr_change = prediction[prediction["position"] == "WR"]["adp_change"].mean()
    qb_change = prediction[prediction["position"] == "QB"]["adp_change"].mean()
    assert wr_change > qb_change
