from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.adp import ADPDataProvider, FantasyCalcADPProvider
from src.baseline_artifacts import CanonicalArtifactManager
from src.model_builder import build_candidate_model, run_public_canonical_analysis, save_candidate_model
from src.models import ConfigError


def sample_fantasycalc_payload(adp_offset: float = 0.0) -> list[dict[str, Any]]:
    return [
        {
            "player": {"name": "Justin Jefferson", "position": "WR", "maybeTeam": "MIN", "sleeperId": "W1"},
            "maybeAdp": 1.0 + adp_offset,
            "overallRank": 1,
            "positionRank": 1,
        },
        {
            "player": {"name": "Christian McCaffrey", "position": "RB", "maybeTeam": "SF", "sleeperId": "R1"},
            "maybeAdp": 2.0 + adp_offset,
            "overallRank": 2,
            "positionRank": 1,
        },
        {
            "player": {"name": "Josh Allen", "position": "QB", "maybeTeam": "BUF", "sleeperId": "Q1"},
            "maybeAdp": None,
            "overallRank": 5 + adp_offset,
            "positionRank": 1,
        },
        {
            "player": {"name": "Josh Allen", "position": "QB", "maybeTeam": "BUF", "sleeperId": "Q1"},
            "maybeAdp": 7.0 + adp_offset,
            "overallRank": 6,
            "positionRank": 2,
        },
        {
            "player": {"name": "Broken Player", "position": None, "maybeTeam": "FA", "sleeperId": "X1"},
            "maybeAdp": 80.0,
            "overallRank": 80,
            "positionRank": 1,
        },
    ]


class DummyResponse:
    def __init__(self, payload: Any) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Any:
        return self.payload


class DummySession:
    def __init__(self, payloads: list[Any] | None = None, error: Exception | None = None) -> None:
        self.payloads = payloads or []
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, params: dict[str, Any], timeout: int) -> DummyResponse:
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        if self.error is not None:
            raise self.error
        if not self.payloads:
            raise AssertionError("No payload queued for DummySession.get")
        return DummyResponse(self.payloads.pop(0))


class StaticFantasyCalcProvider:
    def __init__(self, frames: dict[str, pd.DataFrame]) -> None:
        self.frames = {key: ADPDataProvider._normalize_frame(value.copy()) for key, value in frames.items()}

    def _entry(self, environment_key: str, team_count: int) -> dict[str, Any]:
        frame = self.frames[environment_key]
        return {
            "status": "cache_hit",
            "retrieved_at": "2026-08-18T00:00:00+00:00",
            "team_count": team_count,
            "player_count": int(len(frame)),
            "fallback_adp_count": int((frame.get("adp_source_field", pd.Series(dtype=str)) == "overallRank").sum())
            if "adp_source_field" in frame.columns
            else 0,
        }

    def load_canonical_markets(self, team_counts_by_environment: dict[str, int], *, force_refresh: bool = False) -> dict[str, Any]:
        return {
            "source": "FantasyCalc",
            "status": "Healthy",
            "last_refresh": "2026-08-18T00:00:00+00:00",
            "canonical_team_count": sorted(set(team_counts_by_environment.values())),
            "formats": {
                environment_key: self._entry(environment_key, team_counts_by_environment[environment_key])
                for environment_key in team_counts_by_environment
            },
            "frames": {key: value.copy() for key, value in self.frames.items()},
        }

    def load_environment(self, environment_key: str, *, num_teams: int, force_refresh: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
        return self.frames[environment_key].copy(), self._entry(environment_key, num_teams)


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


def test_fantasycalc_response_parses_and_reports_diagnostics(tmp_path) -> None:
    provider = FantasyCalcADPProvider(
        session=DummySession([sample_fantasycalc_payload()]),
        cache_dir=tmp_path,
        metadata_path=tmp_path / "metadata.json",
    )

    frame, metadata = provider.load_environment("1qb_ppr", num_teams=12)

    assert set(frame["sleeper_id"].dropna()) == {"W1", "R1", "Q1"}
    assert float(frame.loc[frame["sleeper_id"] == "Q1", "adp"].iloc[0]) == 5.0
    assert metadata["fallback_adp_count"] == 1
    assert metadata["missing_position_count"] == 1
    assert metadata["duplicate_player_ids"] == ["Q1"]
    assert metadata["status"] == "refreshed"


def test_parameter_mapping_covers_all_six_formats() -> None:
    assert FantasyCalcADPProvider.parameters_for_environment("1qb_standard", 12) == {
        "isDynasty": "false",
        "numQbs": 1,
        "numTeams": 12,
        "ppr": 0.0,
        "includeAdp": "true",
    }
    assert FantasyCalcADPProvider.parameters_for_environment("1qb_half_ppr", 12)["ppr"] == 0.5
    assert FantasyCalcADPProvider.parameters_for_environment("1qb_ppr", 12)["ppr"] == 1.0
    assert FantasyCalcADPProvider.parameters_for_environment("sf_standard", 12)["numQbs"] == 2
    assert FantasyCalcADPProvider.parameters_for_environment("sf_half_ppr", 12)["ppr"] == 0.5
    assert FantasyCalcADPProvider.parameters_for_environment("sf_ppr", 12)["ppr"] == 1.0


def test_fresh_cache_prevents_unnecessary_http_request(tmp_path) -> None:
    session = DummySession([sample_fantasycalc_payload()])
    provider = FantasyCalcADPProvider(session=session, cache_dir=tmp_path, metadata_path=tmp_path / "metadata.json")

    provider.load_environment("1qb_ppr", num_teams=12)
    provider.load_environment("1qb_ppr", num_teams=12)

    assert len(session.calls) == 1


def test_stale_cache_triggers_refresh_and_updates_cache(tmp_path) -> None:
    metadata_path = tmp_path / "metadata.json"
    provider = FantasyCalcADPProvider(
        session=DummySession([sample_fantasycalc_payload()]),
        cache_dir=tmp_path,
        metadata_path=metadata_path,
    )
    provider.load_environment("1qb_ppr", num_teams=12)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["formats"]["1qb_ppr"]["retrieved_at"] = (datetime.now(UTC) - timedelta(days=2)).isoformat()
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    refreshed_provider = FantasyCalcADPProvider(
        session=DummySession([sample_fantasycalc_payload(adp_offset=10.0)]),
        cache_dir=tmp_path,
        metadata_path=metadata_path,
    )
    frame, runtime_metadata = refreshed_provider.load_environment("1qb_ppr", num_teams=12)

    assert float(frame.loc[frame["sleeper_id"] == "W1", "adp"].iloc[0]) == 11.0
    assert runtime_metadata["status"] == "refreshed"


def test_failed_refresh_uses_valid_cached_copy(tmp_path) -> None:
    metadata_path = tmp_path / "metadata.json"
    provider = FantasyCalcADPProvider(
        session=DummySession([sample_fantasycalc_payload()]),
        cache_dir=tmp_path,
        metadata_path=metadata_path,
    )
    cached_frame, _ = provider.load_environment("1qb_ppr", num_teams=12)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["formats"]["1qb_ppr"]["retrieved_at"] = (datetime.now(UTC) - timedelta(days=2)).isoformat()
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    failing_provider = FantasyCalcADPProvider(
        session=DummySession(error=RuntimeError("FantasyCalc down")),
        cache_dir=tmp_path,
        metadata_path=metadata_path,
    )
    fallback_frame, runtime_metadata = failing_provider.load_environment("1qb_ppr", num_teams=12)

    assert fallback_frame.equals(cached_frame)
    assert runtime_metadata["status"] == "stale_cache_fallback"
    assert "FantasyCalc down" in runtime_metadata["warning"]


def test_failed_refresh_without_cache_fails_clearly(tmp_path) -> None:
    provider = FantasyCalcADPProvider(
        session=DummySession(error=RuntimeError("No network")),
        cache_dir=tmp_path,
        metadata_path=tmp_path / "metadata.json",
    )

    with pytest.raises(ConfigError):
        provider.load_environment("1qb_ppr", num_teams=12)


def test_manual_refresh_bypasses_fresh_cache(tmp_path) -> None:
    metadata_path = tmp_path / "metadata.json"
    first_provider = FantasyCalcADPProvider(
        session=DummySession([sample_fantasycalc_payload()]),
        cache_dir=tmp_path,
        metadata_path=metadata_path,
    )
    first_provider.load_environment("1qb_ppr", num_teams=12)

    second_provider = FantasyCalcADPProvider(
        session=DummySession([sample_fantasycalc_payload(adp_offset=20.0)]),
        cache_dir=tmp_path,
        metadata_path=metadata_path,
    )
    frame, runtime_metadata = second_provider.load_environment("1qb_ppr", num_teams=12, force_refresh=True)

    assert float(frame.loc[frame["sleeper_id"] == "W1", "adp"].iloc[0]) == 21.0
    assert runtime_metadata["status"] == "force_refreshed"


def test_candidate_model_uses_provider_and_public_runtime_uses_latest_adp(
    tmp_path,
    mock_client,
    canonical_league_ids,
    canonical_adp_frames,
) -> None:
    build_provider = StaticFantasyCalcProvider(canonical_adp_frames)
    bundle = build_candidate_model(mock_client, canonical_leagues=canonical_league_ids, adp_provider=build_provider)
    assert len(bundle["selected_validation"]) == 30

    candidate_manager = _manager(tmp_path, "candidate")
    production_manager = _manager(tmp_path, "production")
    save_candidate_model(candidate_manager, bundle)
    production_manager.promote_from(candidate_manager)

    updated_frames = {key: value.copy() for key, value in canonical_adp_frames.items()}
    updated_frames["1qb_ppr"].loc[updated_frames["1qb_ppr"]["player_name"] == "WR Player 1", "adp"] = 42.0
    runtime_provider = StaticFantasyCalcProvider(updated_frames)

    analysis = run_public_canonical_analysis(
        client=mock_client,
        production_manager=production_manager,
        target_league_id="2026",
        adp_provider=runtime_provider,
    )

    result_row = analysis["results"].loc[analysis["results"]["player_name"] == "WR Player 1"].iloc[0]
    assert float(result_row["adp"]) == 42.0
    assert analysis["adp_source_metadata"]["player_count"] == len(updated_frames["1qb_ppr"])


def test_identical_canonical_markets_block_calibration(mock_client, canonical_league_ids, canonical_adp_frames) -> None:
    identical_frame = canonical_adp_frames["1qb_ppr"]
    provider = StaticFantasyCalcProvider({environment_key: identical_frame.copy() for environment_key in canonical_league_ids})

    with pytest.raises(ConfigError):
        build_candidate_model(mock_client, canonical_leagues=canonical_league_ids, adp_provider=provider)
