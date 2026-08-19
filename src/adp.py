"""ADP loading, BeatADP canonical caching, and player matching helpers."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.config import (
    ADP_CACHE_DIR,
    ADP_CACHE_METADATA_FILE,
    ADP_CACHE_TTL_HOURS,
    BEATADP_PARSER_VERSION,
    BEATADP_PLATFORM_URL,
    BEATADP_SOURCE_NAME,
    CANONICAL_ADP_CACHE_FILES,
    CANONICAL_ADP_METADATA_FILE,
    CANONICAL_ADP_PATHS,
    CANONICAL_ENVIRONMENTS,
    FANTASYCALC_API_URL,
    FANTASYCALC_INCLUDE_ADP,
    FANTASYCALC_IS_DYNASTY,
    FANTASYCALC_SOURCE_NAME,
    REQUEST_TIMEOUT_SECONDS,
)
from src.models import ConfigError
from src.utils import CORE_POSITIONS, ensure_columns, normalize_player_name, rank_players_within_position

FANTASYCALC_ADP_UNAVAILABLE_MESSAGE = "FantasyCalc ADP is currently unavailable for this configuration."
BEATADP_REQUIRED_SOURCE_PREFIX = "SLEEPER|"
BEATADP_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
)

_BEATADP_INTERNAL_TO_ENVIRONMENT = {
    ("PPR", "1QB"): "1qb_ppr",
    ("HALF_PPR", "1QB"): "1qb_half_ppr",
    ("HALF_PPR", "2QB"): "sf_half_ppr",
    ("PPR", "2QB"): "sf_ppr",
}

_BEATADP_ENVIRONMENT_TO_SOURCE_KEY = {
    environment_key: f"SLEEPER|{scoring_format}|REDRAFT|{qb_type}"
    for (scoring_format, qb_type), environment_key in _BEATADP_INTERNAL_TO_ENVIRONMENT.items()
}


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


def _ordered_environment_keys(environment_keys: list[str] | tuple[str, ...] | set[str]) -> tuple[str, ...]:
    requested = set(environment_keys)
    return tuple(environment_key for environment_key in CANONICAL_ENVIRONMENTS if environment_key in requested)


def _read_json_file(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"ADP metadata is malformed at {path}: {exc}") from exc


def load_saved_canonical_adp_paths(
    metadata_path: Path = CANONICAL_ADP_METADATA_FILE,
) -> tuple[dict[str, Path], dict[str, Any]]:
    """Return the saved canonical ADP paths and metadata for available environments."""

    if not metadata_path.exists():
        raise ConfigError(
            "Canonical ADP metadata is missing. Refresh BeatADP canonical ADPs from the Development page first."
        )
    metadata = _read_json_file(metadata_path)
    available = metadata.get("available_environments") or []
    if not available:
        raise ConfigError(
            "Canonical ADP metadata does not list any available environments. Refresh BeatADP canonical ADPs from the Development page."
        )

    paths: dict[str, Path] = {}
    for environment_key in _ordered_environment_keys(available):
        raw_path = metadata.get("formats", {}).get(environment_key, {}).get("path")
        path = Path(raw_path) if raw_path else CANONICAL_ADP_PATHS[environment_key]
        if not path.exists():
            raise ConfigError(
                f"Canonical ADP file is missing for {environment_key}: {path}. Refresh BeatADP canonical ADPs from the Development page."
            )
        paths[environment_key] = path
    return paths, metadata


class CSVADPProvider:
    """CSV-backed ADP provider retained for tests, fixtures, and cached snapshots."""

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
        normalized["team"] = normalized["team"].fillna("").astype(str)
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


class BeatADPProvider:
    """BeatADP-backed canonical ADP provider persisted for public runtime use."""

    def __init__(
        self,
        session: requests.Session | None = None,
        metadata_path: Path = CANONICAL_ADP_METADATA_FILE,
        canonical_paths: dict[str, Path] | None = None,
        source_url: str = BEATADP_PLATFORM_URL,
        request_timeout_seconds: int = REQUEST_TIMEOUT_SECONDS,
        max_attempts: int = 3,
    ) -> None:
        self.session = session or requests.Session()
        self.metadata_path = metadata_path
        self.canonical_paths = canonical_paths or CANONICAL_ADP_PATHS
        self.source_url = source_url
        self.request_timeout_seconds = request_timeout_seconds
        self.max_attempts = max_attempts

    def load_canonical_markets(
        self,
        *,
        force_refresh: bool = False,
        sleeper_players_payload: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Load saved canonical ADP snapshots, or refresh them from BeatADP when requested."""

        if force_refresh:
            return self.refresh_canonical_markets(sleeper_players_payload=sleeper_players_payload)
        return self.load_saved_canonical_markets()

    def load_saved_canonical_markets(self) -> dict[str, Any]:
        paths, metadata = load_saved_canonical_adp_paths(self.metadata_path)
        frames: dict[str, pd.DataFrame] = {}
        formats: dict[str, dict[str, Any]] = {}
        for environment_key, path in paths.items():
            frame, entry = ADPDataProvider(path).load_with_metadata()
            frame["canonical_format"] = environment_key
            frames[environment_key] = frame
            saved_entry = metadata.get("formats", {}).get(environment_key, {}).copy()
            saved_entry.setdefault("path", str(path))
            saved_entry.setdefault("player_count", int(len(frame)))
            saved_entry.setdefault("status", "saved")
            formats[environment_key] = saved_entry
        return {
            "source": metadata.get("source", BEATADP_SOURCE_NAME),
            "source_url": metadata.get("source_url", self.source_url),
            "status": metadata.get("status", "Saved"),
            "last_refresh": metadata.get("fetched_at"),
            "available_environments": _ordered_environment_keys(frames),
            "missing_environments": metadata.get("missing_environments", []),
            "parser_version": metadata.get("parser_version", BEATADP_PARSER_VERSION),
            "formats": formats,
            "frames": frames,
        }

    def refresh_canonical_markets(
        self,
        *,
        sleeper_players_payload: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Fetch BeatADP live, validate available Sleeper markets, and persist them to disk."""

        html = self._fetch_platform_html()
        payload = self.parse_platform_payload(html)
        discovered = self._discover_available_environments(payload)
        match_index = self._build_sleeper_match_index(sleeper_players_payload or {})
        frames: dict[str, pd.DataFrame] = {}
        format_entries: dict[str, dict[str, Any]] = {}

        fetched_at = datetime.now(UTC).isoformat()
        for environment_key, slice_entry in discovered.items():
            frame, entry = self._build_environment_frame(
                payload=payload,
                environment_key=environment_key,
                slice_entry=slice_entry,
                match_index=match_index,
                fetched_at=fetched_at,
            )
            save_path = self.canonical_paths[environment_key]
            save_path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(save_path, index=False)
            entry["path"] = str(save_path)
            frames[environment_key] = frame
            format_entries[environment_key] = entry

        available = _ordered_environment_keys(frames)
        missing = [environment_key for environment_key in CANONICAL_ENVIRONMENTS if environment_key not in available]
        metadata = {
            "source": BEATADP_SOURCE_NAME,
            "source_url": self.source_url,
            "fetched_at": fetched_at,
            "status": "Healthy" if available else "Unavailable",
            "available_environments": list(available),
            "missing_environments": missing,
            "parser_version": BEATADP_PARSER_VERSION,
            "formats": format_entries,
        }
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
        return {
            "source": metadata["source"],
            "source_url": metadata["source_url"],
            "status": metadata["status"],
            "last_refresh": metadata["fetched_at"],
            "available_environments": available,
            "missing_environments": missing,
            "parser_version": metadata["parser_version"],
            "formats": format_entries,
            "frames": frames,
        }

    def parse_platform_payload(self, html: str) -> dict[str, Any]:
        """Extract structured BeatADP payload fragments from the page HTML."""

        latest_recorded_at = self._extract_json_string_after_marker(html, '"latestRecordedAt":')
        slices = self._extract_json_array_after_marker(html, '"slices":')
        players = self._extract_json_array_after_marker(html, '"players":')
        if not isinstance(slices, list) or not isinstance(players, list):
            raise ConfigError("BeatADP page did not contain the expected structured slices/players payload.")
        latest_recorded_at = self._normalize_recorded_at_value(latest_recorded_at, slices)
        return {
            "latest_recorded_at": latest_recorded_at,
            "slices": slices,
            "players": players,
        }

    def _fetch_platform_html(self) -> str:
        headers = {"User-Agent": BEATADP_USER_AGENT, "Accept": "text/html,application/xhtml+xml"}
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self.session.get(self.source_url, headers=headers, timeout=self.request_timeout_seconds)
                response.raise_for_status()
                return response.text
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt == self.max_attempts:
                    break
                time.sleep(0.75 * attempt)
        raise ConfigError(f"BeatADP refresh failed for {self.source_url}: {last_error}") from last_error

    def _discover_available_environments(self, payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
        discovered: dict[str, dict[str, Any]] = {}
        for slice_entry in payload["slices"]:
            if not isinstance(slice_entry, dict):
                continue
            if slice_entry.get("platform") != "SLEEPER" or slice_entry.get("draftType") != "REDRAFT":
                continue
            scoring_format = _coerce_string(slice_entry.get("scoringFormat"))
            qb_type = _coerce_string(slice_entry.get("qbType"))
            if not scoring_format or not qb_type:
                continue
            environment_key = _BEATADP_INTERNAL_TO_ENVIRONMENT.get((scoring_format, qb_type))
            if environment_key is None:
                continue
            discovered[environment_key] = slice_entry
        return {environment_key: discovered[environment_key] for environment_key in _ordered_environment_keys(discovered)}

    def _build_sleeper_match_index(self, players_payload: dict[str, dict[str, Any]]) -> dict[str, dict[tuple[str, ...], list[dict[str, str]]]]:
        by_name_position: dict[tuple[str, str], list[dict[str, str]]] = {}
        by_name_team_position: dict[tuple[str, str, str], list[dict[str, str]]] = {}
        for player_id, payload in players_payload.items():
            position = _coerce_string(payload.get("position"))
            if position is None or position.upper() not in CORE_POSITIONS:
                continue
            position = position.upper()
            full_name = _coerce_string(payload.get("full_name"))
            if full_name is None:
                first_name = _coerce_string(payload.get("first_name")) or ""
                last_name = _coerce_string(payload.get("last_name")) or ""
                full_name = _coerce_string(f"{first_name} {last_name}")
            if full_name is None:
                continue
            team = (_coerce_string(payload.get("team")) or "").upper()
            candidate = {
                "player_id": str(player_id),
                "sleeper_id": str(player_id),
                "player_name": full_name,
                "position": position,
                "team": team,
            }
            normalized_name = normalize_player_name(full_name)
            by_name_position.setdefault((normalized_name, position), []).append(candidate)
            by_name_team_position.setdefault((normalized_name, team, position), []).append(candidate)
        return {
            "by_name_position": by_name_position,
            "by_name_team_position": by_name_team_position,
        }

    def _match_to_sleeper_player(
        self,
        player_name: str,
        position: str,
        team: str,
        match_index: dict[str, dict[tuple[str, ...], list[dict[str, str]]]],
    ) -> tuple[dict[str, str] | None, str]:
        if not match_index["by_name_position"]:
            return None, "unmatched"
        normalized_name = normalize_player_name(player_name)
        name_position_matches = match_index["by_name_position"].get((normalized_name, position), [])
        if len(name_position_matches) == 1:
            return name_position_matches[0], "matched"
        if len(name_position_matches) > 1:
            if team:
                team_matches = match_index["by_name_team_position"].get((normalized_name, team, position), [])
                if len(team_matches) == 1:
                    return team_matches[0], "matched"
                if len(team_matches) > 1:
                    return None, "ambiguous"
            return None, "ambiguous"
        if team:
            team_matches = match_index["by_name_team_position"].get((normalized_name, team, position), [])
            if len(team_matches) == 1:
                return team_matches[0], "matched"
            if len(team_matches) > 1:
                return None, "ambiguous"
        return None, "unmatched"

    def _build_environment_frame(
        self,
        *,
        payload: dict[str, Any],
        environment_key: str,
        slice_entry: dict[str, Any],
        match_index: dict[str, dict[tuple[str, ...], list[dict[str, str]]]],
        fetched_at: str,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        source_key = _BEATADP_ENVIRONMENT_TO_SOURCE_KEY[environment_key]
        raw_rows: list[dict[str, Any]] = []
        matched_rows = 0
        unmatched_rows = 0
        ambiguous_rows = 0
        missing_sleeper_adp_rows = 0

        for item in payload["players"]:
            if not isinstance(item, dict):
                continue
            full_name = _coerce_string(item.get("fullName"))
            position = _coerce_string(item.get("position"))
            if full_name is None or position is None:
                continue
            position = position.upper()
            if position not in CORE_POSITIONS:
                continue

            adps = item.get("adps")
            if not isinstance(adps, dict):
                continue
            adp_value = _coerce_float(adps.get(source_key))
            if adp_value is None:
                missing_sleeper_adp_rows += 1
                continue

            team = (_coerce_string(item.get("teamId")) or "").upper()
            matched_player, match_status = self._match_to_sleeper_player(full_name, position, team, match_index)
            if match_status == "matched" and matched_player is not None:
                matched_rows += 1
            elif match_status == "ambiguous":
                ambiguous_rows += 1
                continue
            else:
                unmatched_rows += 1
                continue

            raw_rows.append(
                {
                    "player_id": matched_player["player_id"],
                    "sleeper_id": matched_player["sleeper_id"],
                    "player_name": full_name,
                    "position": position,
                    "team": team,
                    "adp": float(adp_value),
                    "canonical_format": environment_key,
                    "source": BEATADP_SOURCE_NAME,
                    "adp_source_field": source_key,
                    "retrieved_at": fetched_at,
                    "beatadp_player_id": _coerce_string(item.get("id")),
                    "fantasycalc_rank": pd.NA,
                    "beatadp_recorded_at": _coerce_string(slice_entry.get("recordedAt")) or payload.get("latest_recorded_at"),
                }
            )

        if not raw_rows:
            raise ConfigError(
                f"BeatADP did not return any usable Sleeper ADP rows for {environment_key}. "
                f"Check {self.source_url} and the parser assumptions."
            )

        frame = pd.DataFrame(raw_rows).sort_values(["adp", "player_name"], ascending=[True, True]).reset_index(drop=True)
        duplicate_player_ids = sorted(
            frame.loc[frame["player_id"].duplicated(keep=False), "player_id"].astype(str).unique().tolist()
        )
        if duplicate_player_ids:
            frame = frame.drop_duplicates("player_id", keep="first").reset_index(drop=True)
        frame["normalized_name"] = frame["player_name"].map(normalize_player_name)
        frame = rank_players_within_position(frame)
        validation = self._validate_environment_frame(frame)
        entry = {
            "status": "ok",
            "label": environment_key,
            "request_url": self.source_url,
            "request_state": {
                "platform": "SLEEPER",
                "draftType": "REDRAFT",
                "scoringFormat": slice_entry.get("scoringFormat"),
                "qbType": slice_entry.get("qbType"),
            },
            "recorded_at": _coerce_string(slice_entry.get("recordedAt")) or payload.get("latest_recorded_at"),
            "slice_player_count": int(slice_entry.get("playerCount") or 0),
            "player_count": int(len(frame)),
            "raw_player_rows": int(len(raw_rows) + ambiguous_rows + unmatched_rows),
            "matched_rows": matched_rows,
            "unmatched_rows": unmatched_rows,
            "ambiguous_rows": ambiguous_rows,
            "duplicate_rows": len(duplicate_player_ids),
            "duplicate_player_ids": duplicate_player_ids,
            "missing_sleeper_adp_rows": missing_sleeper_adp_rows,
            "min_adp": float(frame["adp"].min()),
            "max_adp": float(frame["adp"].max()),
            "checksum": _frame_checksum(frame),
            "validation": validation,
            "sample_rows": frame[["player_name", "position", "team", "adp", "pos_rank"]].head(20).to_dict(orient="records"),
        }
        return frame, entry

    @staticmethod
    def _validate_environment_frame(frame: pd.DataFrame) -> dict[str, Any]:
        if frame.empty:
            raise ConfigError("BeatADP canonical ADP frame is empty after validation.")
        if frame["adp"].isna().any():
            raise ConfigError("BeatADP canonical ADP frame contains null ADP values.")
        if (frame["adp"] <= 0).any():
            raise ConfigError("BeatADP canonical ADP frame contains non-positive ADP values.")
        if frame["player_id"].duplicated().any():
            raise ConfigError("BeatADP canonical ADP frame contains duplicate matched player IDs.")
        coverage = frame["position"].value_counts().to_dict()
        return {
            "numeric_adp": True,
            "duplicate_player_rows": False,
            "position_coverage": coverage,
            "reasonable_max_adp": float(frame["adp"].max()) <= 400.0,
        }

    @staticmethod
    def _extract_json_array_after_marker(html: str, marker: str) -> list[Any]:
        start, matched_marker = BeatADPProvider._find_marker(html, marker)
        if start < 0:
            raise ConfigError(f"BeatADP parser could not find marker {marker!r}.")
        bracket_start = html.find("[", start)
        if bracket_start < 0:
            raise ConfigError(f"BeatADP parser could not find the JSON array after {marker!r}.")
        depth = 0
        in_string = False
        escape = False
        for index in range(bracket_start, len(html)):
            char = html[index]
            if escape:
                escape = False
                continue
            if char == "\\":
                escape = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == "[":
                depth += 1
            elif char == "]":
                depth -= 1
                if depth == 0:
                    try:
                        return BeatADPProvider._loads_json_fragment(
                            html[bracket_start : index + 1],
                            escaped=matched_marker != marker,
                        )
                    except json.JSONDecodeError as exc:
                        raise ConfigError(f"BeatADP parser could not decode structured data for {marker!r}: {exc}") from exc
        raise ConfigError(f"BeatADP parser could not locate the end of the JSON array for {marker!r}.")

    @staticmethod
    def _extract_json_string_after_marker(html: str, marker: str) -> str | None:
        start, matched_marker = BeatADPProvider._find_marker(html, marker)
        if start < 0:
            return None
        quote_start = html.find('"', start + len(matched_marker))
        if quote_start < 0:
            return None
        quote_end = quote_start + 1
        escape = False
        while quote_end < len(html):
            char = html[quote_end]
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                value = html[quote_start + 1 : quote_end]
                if matched_marker != marker:
                    return bytes(value, "utf-8").decode("unicode_escape")
                return value
            quote_end += 1
        return None

    @staticmethod
    def _find_marker(html: str, marker: str) -> tuple[int, str]:
        """Locate either the plain JSON marker or the escaped serialized-HTML variant."""

        candidates = [
            marker,
            marker.replace('"', '\\"'),
        ]
        for candidate in candidates:
            index = html.find(candidate)
            if index >= 0:
                return index, candidate
        return -1, marker

    @staticmethod
    def _loads_json_fragment(fragment: str, *, escaped: bool) -> Any:
        if not escaped:
            return json.loads(fragment)
        return json.loads(bytes(fragment, "utf-8").decode("unicode_escape"))

    @staticmethod
    def _normalize_recorded_at_value(value: str | None, slices: list[Any]) -> str | None:
        if value:
            cleaned = value.split('"', 1)[0].split("}", 1)[0].strip()
            if cleaned:
                return cleaned
        recorded_values = sorted(
            {
                str(item.get("recordedAt"))
                for item in slices
                if isinstance(item, dict) and item.get("recordedAt")
            }
        )
        return recorded_values[-1] if recorded_values else None


class FantasyCalcADPProvider:
    """Live FantasyCalc ADP provider retained only for diagnostics and legacy tests."""

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
            "non_null_maybe_adp_count": diagnostics["non_null_maybe_adp_count"],
            "non_null_maybe_adp_share": diagnostics["non_null_maybe_adp_share"],
            "missing_adp_count": diagnostics["missing_adp_count"],
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
        missing_position_count = 0
        request_note = "Requested FantasyCalc with includeAdp=true."

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
            if position not in CORE_POSITIONS:
                continue

            maybe_adp = _coerce_float(item.get("maybeAdp"))
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
                    "adp_source_field": "maybeAdp",
                    "retrieved_at": datetime.now(UTC).isoformat(),
                }
            )

        non_null_maybe_adp_count = len(rows)
        eligible_rows = non_null_maybe_adp_count + missing_adp_count
        non_null_maybe_adp_share = float(non_null_maybe_adp_count / eligible_rows) if eligible_rows else 0.0
        if missing_adp_count or not rows:
            raise ConfigError(
                f"{FANTASYCALC_ADP_UNAVAILABLE_MESSAGE} "
                f"FantasyCalc returned maybeAdp for {non_null_maybe_adp_count} of {eligible_rows} eligible rows in {environment_key}."
            )

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
        return frame, {
            "non_null_maybe_adp_count": non_null_maybe_adp_count,
            "non_null_maybe_adp_share": round(non_null_maybe_adp_share, 4),
            "missing_adp_count": missing_adp_count,
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
        normalized = self._normalize_cached_frame(frame)
        if not normalized["adp_source_field"].eq("maybeAdp").all():
            raise ConfigError(
                "FantasyCalc cache is invalid because it contains rows not backed by maybeAdp. "
                "Clear the cache and refresh the FantasyCalc source."
            )
        return normalized

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
