from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.adp import ADPDataProvider, BEATADP_SOURCE_NAME, BeatADPProvider, load_saved_canonical_adp_paths
from src.baseline_artifacts import CanonicalArtifactManager
from src.model_builder import build_candidate_model, run_public_canonical_analysis, save_candidate_model
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
    html = '<script>{"payload":"{\\"slices\\":[{\\"platform\\":\\"SLEEPER\\",\\"scoringFormat\\":\\"PPR\\",\\"draftType\\":\\"REDRAFT\\",\\"qbType\\":\\"1QB\\",\\"recordedAt\\":\\"2026-08-15\\",\\"playerCount\\":1}],\\"players\\":[{\\"id\\":1,\\"fullName\\":\\"QB Player 1\\",\\"position\\":\\"QB\\",\\"teamId\\":\\"T1\\",\\"adps\\":{\\"SLEEPER|PPR|REDRAFT|1QB\\":10.0}}],\\"latestRecordedAt\\":\\"2026-08-15\\"}"}<\\/script>'
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


def test_load_saved_canonical_adp_paths_falls_back_from_stale_absolute_metadata_path(tmp_path, monkeypatch) -> None:
    actual_path = tmp_path / "adp_1qb_half_ppr.csv"
    pd.DataFrame([{"player_name": "RB Player 1", "position": "RB", "team": "T1", "adp": 1.0}]).to_csv(actual_path, index=False)
    metadata_path = tmp_path / "adp_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "available_environments": ["1qb_half_ppr"],
                "formats": {
                    "1qb_half_ppr": {
                        "path": r"Z:\streamlit-cloud-only\adp_1qb_half_ppr.csv"
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("src.adp.CANONICAL_ADP_PATHS", {"1qb_half_ppr": actual_path})

    paths, _ = load_saved_canonical_adp_paths(metadata_path=metadata_path)

    assert paths["1qb_half_ppr"] == actual_path


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
