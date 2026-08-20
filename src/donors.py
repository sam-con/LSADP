"""Seed-league loading for the position-specific history library."""

from __future__ import annotations

import csv
import json
import re
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.errors import EmptyDataError

from src.config import HISTORICAL_DONOR_FILE
from src.models import ConfigError


DONOR_COLUMN_ALIASES = {
    "league_id": ("league_id", "donor_league_id", "sleeper_league_id", "league"),
    "season": ("season", "year"),
    "selected": ("selected", "use", "include", "enabled", "active", "is_selected"),
}


def _normalized_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _coerce_selected(value: Any) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return True
    token = _normalized_token(value)
    if token in {"", "1", "true", "t", "yes", "y", "selected", "include", "included", "use", "active"}:
        return True
    if token in {"0", "false", "f", "no", "n", "excluded", "exclude", "skip", "inactive"}:
        return False
    return True


def _first_matching_column(columns: pd.Index[str], aliases: tuple[str, ...]) -> str | None:
    lowered = {str(column).strip().lower(): str(column) for column in columns}
    for alias in aliases:
        if alias in lowered:
            return lowered[alias]
    return None


def _read_donor_source_file(donor_path: Path) -> pd.DataFrame:
    sample = donor_path.read_text(encoding="utf-8")
    preview = "\n".join(sample.splitlines()[:10])
    first_line = sample.splitlines()[0] if sample.splitlines() else ""
    if first_line.count("\t") > first_line.count(","):
        delimiter = "\t"
    else:
        try:
            dialect = csv.Sniffer().sniff(preview, delimiters=",\t;|")
            delimiter = dialect.delimiter
        except csv.Error:
            delimiter = ","
    return pd.read_csv(donor_path, sep=delimiter)


def _coerce_json_seed_entries(entries: Any) -> list[str]:
    if isinstance(entries, (str, int)):
        return [str(entries)]
    if isinstance(entries, dict):
        if "league_id" in entries:
            return [str(entries["league_id"])]
        if "league_ids" in entries and isinstance(entries["league_ids"], list):
            return [str(league_id) for league_id in entries["league_ids"]]
        raise ConfigError("Historical donor JSON entries must provide `league_id` or `league_ids`.")
    if isinstance(entries, list):
        flattened: list[str] = []
        for item in entries:
            flattened.extend(_coerce_json_seed_entries(item))
        return flattened
    raise ConfigError("Historical donor JSON cells must be a string, object, or list.")


def _read_year_keyed_donor_json(donor_path: Path) -> pd.DataFrame:
    try:
        payload = json.loads(donor_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Historical donor JSON is malformed: {exc}") from exc
    if not isinstance(payload, dict) or not payload:
        raise ConfigError("Historical donor JSON must be a non-empty object keyed by season.")

    rows: list[dict[str, Any]] = []
    for season_key, season_payload in payload.items():
        try:
            season = int(season_key)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"Historical donor JSON has an invalid season key: {season_key}") from exc
        if not isinstance(season_payload, dict):
            raise ConfigError(f"Historical donor JSON season `{season}` must map to a donor list or object map.")
        for entries in season_payload.values():
            for league_id in _coerce_json_seed_entries(entries):
                rows.append({"league_id": league_id, "season": season, "selected": True})
    return pd.DataFrame(rows)


def load_history_seed_leagues(
    donor_path: Path = HISTORICAL_DONOR_FILE,
    *,
    today: date | None = None,
    donor_configuration: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the curated seed leagues used to build the history library."""

    today = today or date.today()
    if donor_configuration is not None:
        raw = donor_configuration.copy()
        source = "provided_dataframe"
    else:
        if not donor_path.exists():
            raise ConfigError(f"Historical donor source is missing at {donor_path}.")
        if donor_path.stat().st_size == 0:
            raise ConfigError(f"Historical donor source `{donor_path}` is empty.")
        if donor_path.suffix.lower() == ".json":
            raw = _read_year_keyed_donor_json(donor_path)
        else:
            try:
                raw = _read_donor_source_file(donor_path)
            except EmptyDataError as exc:
                raise ConfigError(f"Historical donor source `{donor_path}` is empty.") from exc
        source = str(donor_path)

    if raw.empty:
        raise ConfigError("Historical donor source does not contain any seed-league rows.")

    column_map: dict[str, str] = {}
    for logical_name, aliases in DONOR_COLUMN_ALIASES.items():
        column = _first_matching_column(raw.columns, aliases)
        if column is not None:
            column_map[logical_name] = column
    if "league_id" not in column_map or "season" not in column_map:
        raise ConfigError(
            "Historical donor source must contain `league_id` and `season` columns for history-library seeding."
        )

    seeds = pd.DataFrame(
        {
            "league_id": raw[column_map["league_id"]].astype(str).str.strip(),
            "season": pd.to_numeric(raw[column_map["season"]], errors="coerce"),
            "selected": raw[column_map["selected"]].map(_coerce_selected) if "selected" in column_map else True,
        }
    )
    seeds = seeds.dropna(subset=["season"])
    seeds["season"] = seeds["season"].astype(int)
    seeds = seeds[seeds["selected"]].copy()
    max_completed_season = today.year - 1
    seeds = seeds[seeds["season"].between(2000, max_completed_season)].copy()
    seeds = seeds.drop_duplicates(subset=["league_id", "season"], keep="first").reset_index(drop=True)
    if seeds.empty:
        raise ConfigError("Historical donor source does not contain any selected completed seasons for seed loading.")

    metadata = {
        "source": source,
        "selected_rows": int(len(seeds)),
        "unique_leagues": int(seeds["league_id"].nunique()),
        "season_range": [int(seeds["season"].min()), int(seeds["season"].max())],
    }
    return seeds, metadata
