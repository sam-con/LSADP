from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.adp import (
    ADPDataProvider,
    BEATADP_SOURCE_NAME,
    BeatADPProvider,
    FANTASYCALC_ADP_UNAVAILABLE_MESSAGE,
    FantasyCalcADPProvider,
)
from src.baseline_artifacts import CanonicalArtifactManager
from src.model_builder import build_candidate_model, run_public_canonical_analysis, save_candidate_model
from src.models import ConfigError

UTC = timezone.utc


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
            "maybeAdp": 7.0 + adp_offset,
            "overallRank": 5 + adp_offset,
            "positionRank": 2,
        },
        {
            "player": {"name": "Broken Player", "position": None, "maybeTeam": "FA", "sleeperId": "X1"},
            "maybeAdp": 80.0,
            "overallRank": 80,
            "positionRank": 1,
        },
    ]


def sample_fantasycalc_payload_with_missing_adp() -> list[dict[str, Any]]:
    payload = sample_fantasycalc_payload()
    payload.append(
        {
            "player": {"name": "Lamar Jackson", "position": "QB", "maybeTeam": "BAL", "sleeperId": "Q2"},
            "maybeAdp": None,
            "overallRank": 4,
            "positionRank": 1,
        }
    )
    return payload


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


class DummyHTMLResponse:
    def __init__(self, html: str) -> None:
        self.text = html

    def raise_for_status(self) -> None:
        return None


class DummyHTMLSession:
    def __init__(self, html: str) -> None:
        self.html = html
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, headers: dict[str, Any], timeout: int) -> DummyHTMLResponse:
        self.calls.append({"url": url, "headers": headers, "timeout": timeout})
        return DummyHTMLResponse(self.html)


class StaticCanonicalADPProvider:
    def __init__(self, frames: dict[str, pd.DataFrame]) -> None:
        self.frames = {key: ADPDataProvider._normalize_frame(value.copy()) for key, value in frames.items()}

    def _entry(self, environment_key: str) -> dict[str, Any]:
        frame = self.frames[environment_key]
        return {
            "status": "saved",
            "retrieved_at": "2026-08-19T00:00:00+00:00",
            "recorded_at": "2026-08-15",
            "player_count": int(len(frame)),
            "path": f"{environment_key}.csv",
        }

    def load_canonical_markets(self, *, force_refresh: bool = False) -> dict[str, Any]:
        _ = force_refresh
        return {
            "source": BEATADP_SOURCE_NAME,
            "status": "Healthy",
            "last_refresh": "2026-08-19T00:00:00+00:00",
            "available_environments": list(self.frames),
            "missing_environments": [key for key in ("1qb_half_ppr", "1qb_ppr", "sf_half_ppr", "sf_ppr") if key not in self.frames],
            "formats": {environment_key: self._entry(environment_key) for environment_key in self.frames},
            "frames": {key: value.copy() for key, value in self.frames.items()},
        }

    def load_environment(self, environment_key: str, *, num_teams: int, force_refresh: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
        _ = num_teams
        _ = force_refresh
        return self.frames[environment_key].copy(), self._entry(environment_key)


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


def sample_sleeper_players_payload() -> dict[str, dict[str, Any]]:
    return {
        "Q1": {"player_id": "Q1", "full_name": "QB Player 1", "position": "QB", "team": "T1"},
        "R1": {"player_id": "R1", "full_name": "RB Player 1", "position": "RB", "team": "T1"},
        "W1": {"player_id": "W1", "full_name": "WR Player 1", "position": "WR", "team": "T1"},
        "T1": {"player_id": "T1", "full_name": "TE Player 1", "position": "TE", "team": "T1"},
    }


def test_beatadp_fixture_parses_and_persists_three_markets(tmp_path) -> None:
    fixture_html = (Path(__file__).parent / "fixtures" / "beatadp_platform_adp.html").read_text(encoding="utf-8")
    provider = BeatADPProvider(
        session=DummyHTMLSession(fixture_html),
        metadata_path=tmp_path / "adp_metadata.json",
        canonical_paths={
            "1qb_half_ppr": tmp_path / "adp_1qb_half_ppr.csv",
            "1qb_ppr": tmp_path / "adp_1qb_ppr.csv",
            "sf_half_ppr": tmp_path / "adp_sf_half_ppr.csv",
            "sf_ppr": tmp_path / "adp_sf_ppr.csv",
        },
    )

    bundle = provider.refresh_canonical_markets(sleeper_players_payload=sample_sleeper_players_payload())

    assert bundle["available_environments"] == ("1qb_half_ppr", "1qb_ppr", "sf_half_ppr")
    assert bundle["missing_environments"] == ["sf_ppr"]
    assert set(bundle["frames"]) == {"1qb_half_ppr", "1qb_ppr", "sf_half_ppr"}
    assert bundle["formats"]["1qb_ppr"]["unmatched_rows"] == 1
    assert bundle["formats"]["1qb_ppr"]["missing_sleeper_adp_rows"] == 1
    assert float(bundle["frames"]["1qb_ppr"].iloc[0]["adp"]) == 4.0
    assert bundle["frames"]["1qb_ppr"]["source"].eq(BEATADP_SOURCE_NAME).all()
    assert not (tmp_path / "adp_sf_ppr.csv").exists()


def test_beatadp_parser_accepts_current_escaped_marker_style() -> None:
    html = '<script>{"payload":"{\\"slices\\":[{\\"platform\\":\\"SLEEPER\\",\\"scoringFormat\\":\\"PPR\\",\\"draftType\\":\\"REDRAFT\\",\\"qbType\\":\\"1QB\\",\\"recordedAt\\":\\"2026-08-15\\",\\"playerCount\\":1}],\\"players\\":[{\\"id\\":1,\\"fullName\\":\\"QB Player 1\\",\\"position\\":\\"QB\\",\\"teamId\\":\\"T1\\",\\"adps\\":{\\"SLEEPER|PPR|REDRAFT|1QB\\":10.0}}],\\"latestRecordedAt\\":\\"2026-08-15\\"}"}<\/script>'
    payload = BeatADPProvider().parse_platform_payload(html)

    assert payload["latest_recorded_at"] == "2026-08-15"
    assert payload["slices"][0]["platform"] == "SLEEPER"
    assert payload["players"][0]["fullName"] == "QB Player 1"


def test_beatadp_missing_sleeper_values_never_substitute_other_columns(tmp_path) -> None:
    fixture_html = (Path(__file__).parent / "fixtures" / "beatadp_platform_adp.html").read_text(encoding="utf-8")
    provider = BeatADPProvider(
        session=DummyHTMLSession(fixture_html),
        metadata_path=tmp_path / "adp_metadata.json",
        canonical_paths={
            "1qb_half_ppr": tmp_path / "adp_1qb_half_ppr.csv",
            "1qb_ppr": tmp_path / "adp_1qb_ppr.csv",
            "sf_half_ppr": tmp_path / "adp_sf_half_ppr.csv",
            "sf_ppr": tmp_path / "adp_sf_ppr.csv",
        },
    )

    bundle = provider.refresh_canonical_markets(sleeper_players_payload=sample_sleeper_players_payload())
    player_names = bundle["frames"]["1qb_ppr"]["player_name"].tolist()

    assert "Ghost Player" not in player_names
    assert "Unknown Rookie" not in player_names


def test_fantasycalc_response_parses_and_reports_diagnostics(tmp_path) -> None:
    provider = FantasyCalcADPProvider(
        session=DummySession([sample_fantasycalc_payload()]),
        cache_dir=tmp_path,
        metadata_path=tmp_path / "metadata.json",
    )

    frame, metadata = provider.load_environment("1qb_ppr", num_teams=12)

    assert set(frame["sleeper_id"].dropna()) == {"W1", "R1", "Q1"}
    assert float(frame.loc[frame["sleeper_id"] == "Q1", "adp"].iloc[0]) == 7.0
    assert float(frame.loc[frame["sleeper_id"] == "Q1", "overall_rank"].iloc[0]) == 5.0
    assert float(frame.loc[frame["sleeper_id"] == "Q1", "position_rank_market"].iloc[0]) == 2.0
    assert metadata["non_null_maybe_adp_count"] == 3
    assert metadata["missing_adp_count"] == 0
    assert metadata["missing_position_count"] == 1
    assert metadata["duplicate_player_ids"] == []
    assert metadata["status"] == "refreshed"


def test_null_maybe_adp_never_falls_back_to_overall_rank(tmp_path) -> None:
    provider = FantasyCalcADPProvider(
        session=DummySession([sample_fantasycalc_payload_with_missing_adp()]),
        cache_dir=tmp_path,
        metadata_path=tmp_path / "metadata.json",
    )

    with pytest.raises(ConfigError, match="FantasyCalc ADP is currently unavailable for this configuration."):
        provider.load_environment("1qb_ppr", num_teams=12)


def test_missing_adp_causes_controlled_failure(tmp_path) -> None:
    provider = FantasyCalcADPProvider(
        session=DummySession([sample_fantasycalc_payload_with_missing_adp()]),
        cache_dir=tmp_path,
        metadata_path=tmp_path / "metadata.json",
    )

    with pytest.raises(ConfigError, match=FANTASYCALC_ADP_UNAVAILABLE_MESSAGE):
        provider.load_environment("sf_ppr", num_teams=12)


def test_overall_rank_and_adp_remain_distinct_fields(tmp_path) -> None:
    provider = FantasyCalcADPProvider(
        session=DummySession([sample_fantasycalc_payload()]),
        cache_dir=tmp_path,
        metadata_path=tmp_path / "metadata.json",
    )

    frame, _ = provider.load_environment("1qb_half_ppr", num_teams=12)

    assert "overall_rank" in frame.columns
    assert "adp" in frame.columns
    assert not frame["adp"].equals(frame["overall_rank"])
    assert frame["adp_source_field"].eq("maybeAdp").all()


def test_parameter_mapping_covers_all_four_formats() -> None:
    assert FantasyCalcADPProvider.parameters_for_environment("1qb_half_ppr", 12)["ppr"] == 0.5
    assert FantasyCalcADPProvider.parameters_for_environment("1qb_ppr", 12)["ppr"] == 1.0
    assert FantasyCalcADPProvider.parameters_for_environment("sf_half_ppr", 12)["numQbs"] == 2
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


def test_cached_rows_backed_by_rankings_are_rejected(tmp_path) -> None:
    cache_path = tmp_path / "1qb_ppr.csv"
    pd.DataFrame(
        [
            {
                "player_id": "Q1",
                "sleeper_id": "Q1",
                "player_name": "Josh Allen",
                "position": "QB",
                "team": "BUF",
                "adp": 5.0,
                "overall_rank": 5.0,
                "position_rank_market": 1.0,
                "canonical_format": "1qb_ppr",
                "source": "FantasyCalc",
                "adp_source_field": "overallRank",
                "retrieved_at": "2026-08-19T00:00:00+00:00",
            }
        ]
    ).to_csv(cache_path, index=False)

    provider = FantasyCalcADPProvider(cache_dir=tmp_path, metadata_path=tmp_path / "metadata.json")

    with pytest.raises(ConfigError, match="contains rows not backed by maybeAdp"):
        provider._load_cached_frame(cache_path)


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
    build_provider = StaticCanonicalADPProvider(canonical_adp_frames)
    bundle = build_candidate_model(
        mock_client,
        canonical_leagues=canonical_league_ids,
        adp_provider=build_provider,
        donor_configuration=donor_configuration_frame(),
    )
    assert len(bundle["selected_validation"]) == 12

    candidate_manager = _manager(tmp_path, "candidate")
    production_manager = _manager(tmp_path, "production")
    save_candidate_model(candidate_manager, bundle)
    production_manager.promote_from(candidate_manager)

    updated_frames = {key: value.copy() for key, value in canonical_adp_frames.items()}
    updated_frames["1qb_ppr"].loc[updated_frames["1qb_ppr"]["player_name"] == "WR Player 1", "adp"] = 42.0
    runtime_provider = StaticCanonicalADPProvider(updated_frames)

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
    provider = StaticCanonicalADPProvider({environment_key: identical_frame.copy() for environment_key in canonical_league_ids})

    with pytest.raises(ConfigError):
        build_candidate_model(
            mock_client,
            canonical_leagues=canonical_league_ids,
            adp_provider=provider,
            donor_configuration=donor_configuration_frame(),
        )
