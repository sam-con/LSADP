"""ADP data loading, live FantasyCalc caching, and player matching."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.config import (
    ADP_CACHE_DIR,
    ADP_CACHE_METADATA_FILE,
    ADP_CACHE_TTL_HOURS,
    CANONICAL_ADP_CACHE_FILES,
    CANONICAL_ENVIRONMENTS,
    FANTASYCALC_API_URL,
    FANTASYCALC_INCLUDE_ADP,
    FANTASYCALC_IS_DYNASTY,
    FANTASYCALC_SOURCE_NAME,
    REQUEST_TIMEOUT_SECONDS,
)
from src.models import ConfigError
from src.utils import ensure_columns, normalize_player_name, rank_players_within_position


def _coerce_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _frame_checksum(frame: pd.DataFrame) -> str:
    serialized = frame.sort_values(["adp", "player_name"], ascending=[True, True]).to_csv(index=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class CSVADPProvider:
    """CSV-backed ADP provider retained for tests, fixtures, and debugging."""

    required_columns = ("player_name", "position", "team", "adp")

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            raise ConfigError(f"ADP file not found at {self.path}")
        frame = pd.read_csv(self.path)
        return self._normalize_frame(frame)

    def load_with_metadata(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        frame = self.load()
        return frame, {
            "source": "csv",
            "path": str(self.path),
            "player_count": int(len(frame)),
            "checksum": _frame_checksum(frame),
        }

    @classmethod
    def _normalize_frame(cls, frame: pd.DataFrame) -> pd.DataFrame:
        ensure_columns(frame, cls.required_columns)
        if frame.empty:
            raise ConfigError("ADP file is empty")
        normalized = frame.copy()
        if "player_id" not in normalized.columns:
            normalized["player_id"] = pd.NA
        if "sleeper_id" not in normalized.columns:
            normalized["sleeper_id"] = normalized["player_id"]
        normalized["player_id"] = normalized["player_id"].astype("string")
        normalized["sleeper_id"] = normalized["sleeper_id"].astype("string")
        normalized["player_name"] = normalized["player_name"].astype(str)
        normalized["position"] = normalized["position"].astype(str).str.upper()
        normalized["team"] = normalized["team"].astype(str)
        normalized["adp"] = normalized["adp"].astype(float)
        if "source" not in normalized.columns:
            normalized["source"] = "csv"
        if "retrieved_at" not in normalized.columns:
            normalized["retrieved_at"] = pd.NA
        if "canonical_format" not in normalized.columns:
            normalized["canonical_format"] = pd.NA
        if "adp_source_field" not in normalized.columns:
            normalized["adp_source_field"] = "adp"
        normalized["normalized_name"] = normalized["player_name"].map(normalize_player_name)
        return rank_players_within_position(normalized)


ADPDataProvider = CSVADPProvider


class FantasyCalcADPProvider:
    """Live FantasyCalc ADP provider with persistent per-format caching."""

    def __init__(
        self,
        session: requests.Session | None = None,
        cache_dir: Path = ADP_CACHE_DIR,
        metadata_path: Path = ADP_CACHE_METADATA_FILE,
        base_url: str = FANTASYCALC_API_URL,
        request_timeout_seconds: int = REQUEST_TIMEOUT_SECONDS,
        cache_ttl_hours: int = ADP_CACHE_TTL_HOURS,
    ) -> None:
        self.session = session or requests.Session()
        self.cache_dir = cache_dir
        self.metadata_path = metadata_path
        self.base_url = base_url
        self.request_timeout_seconds = request_timeout_seconds
        self.cache_ttl_hours = cache_ttl_hours

    @staticmethod
    def parameters_for_environment(environment_key: str, num_teams: int) -> dict[str, Any]:
        mapping = {
            "1qb_half_ppr": {"numQbs": 1, "ppr": 0.5},
            "1qb_ppr": {"numQbs": 1, "ppr": 1.0},
            "sf_half_ppr": {"numQbs": 2, "ppr": 0.5},
            "sf_ppr": {"numQbs": 2, "ppr": 1.0},
        }
        if environment_key not in mapping:
            raise ConfigError(f"Unsupported canonical environment `{environment_key}` for FantasyCalc ADP.")
        return {
            "isDynasty": str(FANTASYCALC_IS_DYNASTY).lower(),
            "numQbs": mapping[environment_key]["numQbs"],
            "numTeams": int(num_teams),
            "ppr": mapping[environment_key]["ppr"],
            "includeAdp": str(FANTASYCALC_INCLUDE_ADP).lower(),
        }

    def load_environment(
        self,
        environment_key: str,
        *,
        num_teams: int,
        force_refresh: bool = False,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        params = self.parameters_for_environment(environment_key, num_teams=num_teams)
        metadata = self._read_metadata()
        entry = metadata["formats"].get(environment_key, {})
        cache_path = self.cache_dir / CANONICAL_ADP_CACHE_FILES[environment_key]
        fresh_cache = self._cache_is_fresh(entry, cache_path, params=params, num_teams=num_teams)

        if fresh_cache and not force_refresh:
            frame = self._load_cached_frame(cache_path)
            return frame, self._runtime_metadata(entry, status="cache_hit")

        try:
            frame, diagnostics = self._fetch_environment(environment_key, params=params)
        except Exception as exc:  # noqa: BLE001
            if cache_path.exists():
                frame = self._load_cached_frame(cache_path)
                fallback_status = "stale_cache_fallback" if not fresh_cache else "cache_hit"
                fallback_entry = entry.copy()
                fallback_entry["warning"] = str(exc)
                return frame, self._runtime_metadata(fallback_entry, status=fallback_status)
            raise ConfigError(
                f"FantasyCalc ADP refresh failed for {environment_key} and no cached copy is available: {exc}"
            ) from exc

        entry = {
            "source": FANTASYCALC_SOURCE_NAME,
            "status": "ok",
            "cache_file": CANONICAL_ADP_CACHE_FILES[environment_key],
            "retrieved_at": datetime.now(UTC).isoformat(),
            "endpoint": self.base_url,
            "parameters": params,
            "team_count": int(num_teams),
            "player_count": int(len(frame)),
            "checksum": _frame_checksum(frame),
            "payload_shape": diagnostics["payload_shape"],
            "missing_adp_count": diagnostics["missing_adp_count"],
            "fallback_adp_count": diagnostics["fallback_adp_count"],
            "missing_position_count": diagnostics["missing_position_count"],
            "duplicate_player_ids": diagnostics["duplicate_player_ids"],
            "ambiguous_name_keys": diagnostics["ambiguous_name_keys"],
            "request_note": diagnostics["request_note"],
        }
        self._persist_cache(environment_key, frame, entry, metadata)
        status = "force_refreshed" if force_refresh else "refreshed"
        return frame, self._runtime_metadata(entry, status=status)

    def load_canonical_markets(
        self,
        team_counts_by_environment: dict[str, int],
        *,
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        missing = [environment_key for environment_key in CANONICAL_ENVIRONMENTS if environment_key not in team_counts_by_environment]
        if missing:
            raise ConfigError(f"Missing canonical team counts for: {', '.join(missing)}")

        frames: dict[str, pd.DataFrame] = {}
        formats: dict[str, dict[str, Any]] = {}
        for environment_key in CANONICAL_ENVIRONMENTS:
            frame, entry = self.load_environment(
                environment_key,
                num_teams=int(team_counts_by_environment[environment_key]),
                force_refresh=force_refresh,
            )
            frames[environment_key] = frame
            formats[environment_key] = entry

        retrieved_at_values = [entry.get("retrieved_at") for entry in formats.values() if entry.get("retrieved_at")]
        statuses = [str(entry.get("status", "")) for entry in formats.values()]
        overall_status = "Healthy"
        if any(status == "stale_cache_fallback" for status in statuses):
            overall_status = "Degraded"

        return {
            "source": FANTASYCALC_SOURCE_NAME,
            "endpoint": self.base_url,
            "status": overall_status,
            "last_refresh": max(retrieved_at_values) if retrieved_at_values else None,
            "canonical_team_count": sorted(set(team_counts_by_environment.values())),
            "formats": formats,
            "frames": frames,
        }

    def _fetch_environment(self, environment_key: str, params: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
        response = self.session.get(self.base_url, params=params, timeout=self.request_timeout_seconds)
        response.raise_for_status()
        payload = response.json()
        payload_shape = "list"
        if isinstance(payload, dict):
            values = payload.get("value")
            if isinstance(values, list):
                payload = values
                payload_shape = "dict.value"
            else:
                raise ConfigError(f"Unexpected FantasyCalc payload shape for {environment_key}: expected a list.")
        if not isinstance(payload, list):
            raise ConfigError(f"Unexpected FantasyCalc payload type for {environment_key}: {type(payload).__name__}")
        frame, diagnostics = self._normalize_payload(payload, environment_key)
        diagnostics["payload_shape"] = payload_shape
        return frame, diagnostics

    def _normalize_payload(self, payload: list[Any], environment_key: str) -> tuple[pd.DataFrame, dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        missing_adp_count = 0
        fallback_adp_count = 0
        missing_position_count = 0
        request_note = ""

        for item in payload:
            if not isinstance(item, dict):
                continue
            player = item.get("player")
            if not isinstance(player, dict):
                player = {}

            player_name = _coerce_string(player.get("name") or item.get("name"))
            position = _coerce_string(player.get("position") or item.get("position"))
            if not player_name or not position:
                if not position:
                    missing_position_count += 1
                continue

            position = position.upper()
            if position not in {"QB", "RB", "WR", "TE"}:
                continue

            maybe_adp = _coerce_float(item.get("maybeAdp"))
            adp_source_field = "maybeAdp"
            if maybe_adp is None:
                maybe_adp = _coerce_float(item.get("overallRank"))
                adp_source_field = "overallRank"
                if maybe_adp is not None:
                    fallback_adp_count += 1
            if maybe_adp is None:
                missing_adp_count += 1
                continue

            team = _coerce_string(player.get("maybeTeam") or player.get("team") or item.get("team")) or ""
            sleeper_id = _coerce_string(player.get("sleeperId") or item.get("sleeperId"))
            rows.append(
                {
                    "player_id": sleeper_id,
                    "sleeper_id": sleeper_id,
                    "player_name": player_name,
                    "position": position,
                    "team": team,
                    "adp": float(maybe_adp),
                    "overall_rank": _coerce_float(item.get("overallRank")),
                    "position_rank_market": _coerce_float(item.get("positionRank")),
                    "canonical_format": environment_key,
                    "source": FANTASYCALC_SOURCE_NAME,
                    "adp_source_field": adp_source_field,
                    "retrieved_at": datetime.now(UTC).isoformat(),
                }
            )

        if not rows:
            raise ConfigError(f"FantasyCalc returned no usable ADP rows for {environment_key}.")

        frame = pd.DataFrame(rows).sort_values(["adp", "player_name"], ascending=[True, True]).reset_index(drop=True)
        frame["normalized_name"] = frame["player_name"].map(normalize_player_name)
        duplicate_ids = sorted(
            frame.loc[frame["player_id"].notna() & frame["player_id"].duplicated(keep=False), "player_id"].astype(str).unique().tolist()
        )
        with_ids = frame[frame["player_id"].notna()].drop_duplicates("player_id", keep="first")
        without_ids = frame[frame["player_id"].isna()].copy()
        ambiguous_keys_frame = without_ids[
            without_ids.duplicated(["normalized_name", "position"], keep=False)
        ][["normalized_name", "position"]].drop_duplicates()
        ambiguous_keys = [
            f"{row.normalized_name}|{row.position}"
            for row in ambiguous_keys_frame.itertuples(index=False)
        ]
        without_ids = without_ids.drop_duplicates(["normalized_name", "position"], keep="first")
        frame = pd.concat([with_ids, without_ids], ignore_index=True)
        frame = self._normalize_cached_frame(frame)
        if fallback_adp_count:
            request_note = (
                "FantasyCalc did not populate maybeAdp for some players, so overallRank was used as a "
                "FantasyCalc-native fallback for those rows."
            )
        return frame, {
            "missing_adp_count": missing_adp_count,
            "fallback_adp_count": fallback_adp_count,
            "missing_position_count": missing_position_count,
            "duplicate_player_ids": duplicate_ids,
            "ambiguous_name_keys": ambiguous_keys,
            "request_note": request_note,
        }

    def _persist_cache(
        self,
        environment_key: str,
        frame: pd.DataFrame,
        entry: dict[str, Any],
        metadata: dict[str, Any],
    ) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self.cache_dir / CANONICAL_ADP_CACHE_FILES[environment_key]
        frame.to_csv(cache_path, index=False)
        metadata.setdefault("source", FANTASYCALC_SOURCE_NAME)
        metadata.setdefault("endpoint", self.base_url)
        metadata.setdefault("cache_ttl_hours", self.cache_ttl_hours)
        metadata.setdefault("formats", {})
        metadata["formats"][environment_key] = entry
        refresh_values = [
            format_entry.get("retrieved_at")
            for format_entry in metadata["formats"].values()
            if format_entry.get("retrieved_at")
        ]
        metadata["last_refresh"] = max(refresh_values) if refresh_values else None
        self.metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    def _read_metadata(self) -> dict[str, Any]:
        if not self.metadata_path.exists():
            return {
                "source": FANTASYCALC_SOURCE_NAME,
                "endpoint": self.base_url,
                "cache_ttl_hours": self.cache_ttl_hours,
                "formats": {},
                "last_refresh": None,
            }
        try:
            return json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ConfigError(f"FantasyCalc cache metadata is malformed: {exc}") from exc

    def _runtime_metadata(self, entry: dict[str, Any], *, status: str) -> dict[str, Any]:
        runtime = entry.copy()
        runtime["status"] = status
        retrieved_at = _parse_timestamp(runtime.get("retrieved_at"))
        if retrieved_at is not None:
            age_hours = (datetime.now(UTC) - retrieved_at).total_seconds() / 3600.0
            runtime["cache_age_hours"] = round(age_hours, 2)
        else:
            runtime["cache_age_hours"] = None
        return runtime

    def _load_cached_frame(self, cache_path: Path) -> pd.DataFrame:
        if not cache_path.exists():
            raise ConfigError(f"FantasyCalc cache file is missing: {cache_path}")
        frame = pd.read_csv(cache_path)
        return self._normalize_cached_frame(frame)

    @staticmethod
    def _normalize_cached_frame(frame: pd.DataFrame) -> pd.DataFrame:
        normalized = frame.copy()
        ensure_columns(normalized, ("player_name", "position", "team", "adp"))
        if "player_id" not in normalized.columns:
            normalized["player_id"] = pd.NA
        if "sleeper_id" not in normalized.columns:
            normalized["sleeper_id"] = normalized["player_id"]
        if "canonical_format" not in normalized.columns:
            normalized["canonical_format"] = pd.NA
        if "source" not in normalized.columns:
            normalized["source"] = FANTASYCALC_SOURCE_NAME
        if "retrieved_at" not in normalized.columns:
            normalized["retrieved_at"] = pd.NA
        if "adp_source_field" not in normalized.columns:
            normalized["adp_source_field"] = "adp"
        normalized["player_id"] = normalized["player_id"].astype("string")
        normalized["sleeper_id"] = normalized["sleeper_id"].astype("string")
        normalized["player_name"] = normalized["player_name"].astype(str)
        normalized["position"] = normalized["position"].astype(str).str.upper()
        normalized["team"] = normalized["team"].fillna("").astype(str)
        normalized["adp"] = normalized["adp"].astype(float)
        normalized["normalized_name"] = normalized["player_name"].map(normalize_player_name)
        return rank_players_within_position(normalized)

    def _cache_is_fresh(
        self,
        entry: dict[str, Any],
        cache_path: Path,
        *,
        params: dict[str, Any],
        num_teams: int,
    ) -> bool:
        if not cache_path.exists():
            return False
        if entry.get("team_count") != int(num_teams):
            return False
        if entry.get("parameters") != params:
            return False
        retrieved_at = _parse_timestamp(entry.get("retrieved_at"))
        if retrieved_at is None:
            return False
        return datetime.now(UTC) - retrieved_at <= timedelta(hours=self.cache_ttl_hours)


def match_players_by_identity(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Match player tables using player_id first and normalized name second."""

    right_columns = [
        column
        for column in right.columns
        if column not in {"player_id", "normalized_name", "player_name", "team", "position"}
    ]
    if left["player_id"].notna().any() and right["player_id"].notna().any():
        merged = left.merge(right[["player_id", *right_columns]].drop_duplicates("player_id"), on="player_id", how="left")
    else:
        merged = left.merge(
            right[["normalized_name", "position", *right_columns]].drop_duplicates(["normalized_name", "position"]),
            on=["normalized_name", "position"],
            how="left",
        )
    return merged
