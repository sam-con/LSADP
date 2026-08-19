"""Historical donor-league loading, validation, optional discovery, and aggregation."""

from __future__ import annotations

import json
import csv
import re
from collections import deque
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.errors import EmptyDataError

from src.analysis import default_modeling_config
from src.canonical import detect_reception_format
from src.config import (
    HISTORICAL_DONOR_FILE,
    HISTORICAL_DONOR_METADATA_FILE,
    SCORING_FORMATS,
)
from src.historical_scores import (
    enrich_player_weeks_with_metadata,
    load_historical_player_weeks,
    summarize_historical_coverage,
    validate_historical_coverage,
)
from src.models import ConfigError, HistoricalLeagueSummary, HistoricalDataError, LeagueSettings, ModelingConfig
from src.sleeper import SleeperClient, as_historical_summary
from src.utils import normalize_scoring_settings, required_completed_seasons

OFFENSIVE_SCORING_KEYS = (
    "pass_yd",
    "pass_td",
    "pass_2pt",
    "pass_int",
    "rush_yd",
    "rush_td",
    "rush_2pt",
    "rec",
    "rec_yd",
    "rec_td",
    "rec_2pt",
    "fum_lost",
)

FLAGGED_SCORING_KEYS = (
    "rec_te",
    "bonus_rec_te",
    "first_down",
    "first_down_pass",
    "first_down_rush",
    "first_down_rec",
    "pass_cmp",
    "pass_icmp",
    "pass_att",
    "bonus_pass_yd_300",
    "bonus_pass_yd_400",
    "bonus_rush_yd_100",
    "bonus_rush_yd_200",
    "bonus_rec_yd_100",
    "bonus_rec_yd_200",
)

DONOR_RESULT_COLUMNS = [
    "league_id",
    "season",
    "scoring_format",
    "status",
    "reason",
    "quality_score",
    "weeks_loaded",
    "unique_player_weeks",
    "unique_players",
    "qb_coverage",
    "rb_coverage",
    "wr_coverage",
    "te_coverage",
    "team_count",
    "league_name",
    "scoring_signature",
    "selected",
    "source_row",
    "discovered_from_user_id",
]

DONOR_EXPORT_COLUMNS = [
    "league_id",
    "season",
    "scoring_format",
    "status",
    "reason",
    "quality_score",
    "weeks_loaded",
    "unique_player_weeks",
    "unique_players",
    "qb_coverage",
    "rb_coverage",
    "wr_coverage",
    "te_coverage",
    "team_count",
    "league_name",
    "scoring_signature",
    "selected",
    "discovered_from_user_id",
]

DONOR_COLUMN_ALIASES = {
    "league_id": ("league_id", "donor_league_id", "sleeper_league_id", "league"),
    "season": ("season", "year"),
    "scoring_format": ("scoring_format", "format", "scoring", "reception_format", "scoring_label"),
    "selected": ("selected", "use", "include", "enabled", "active", "is_selected"),
}

SCORING_FORMAT_ALIASES = {
    "standard": {
        "standard",
        "std",
        "nonppr",
        "non_ppr",
        "0",
        "0.0",
        "zero",
    },
    "half_ppr": {
        "half",
        "halfppr",
        "half_ppr",
        "half-ppr",
        "halfpointppr",
        "0.5",
        "0_5",
        "0-5",
    },
    "ppr": {
        "ppr",
        "fullppr",
        "full_ppr",
        "full-ppr",
        "1",
        "1.0",
    },
}


def canonical_scoring_signatures(canonical_leagues: dict[str, str], client: SleeperClient) -> dict[str, dict[str, float]]:
    """Build one explicit scoring signature per reception environment."""

    signatures: dict[str, dict[str, float]] = {}
    for league_id in canonical_leagues.values():
        league = client.get_league(league_id)
        scoring_format = detect_reception_format(league)
        if scoring_format in SCORING_FORMATS:
            signatures.setdefault(scoring_format, scoring_signature(league.scoring_settings))
    missing = [scoring_format for scoring_format in SCORING_FORMATS if scoring_format not in signatures]
    if missing:
        raise ConfigError(f"Canonical leagues do not provide signatures for: {', '.join(missing)}")
    return signatures


def scoring_signature(scoring_settings: dict[str, float]) -> dict[str, float]:
    """Normalize the offensive scoring rules that define a donor environment."""

    normalized = normalize_scoring_settings(scoring_settings)
    return {key: float(normalized.get(key, 0.0)) for key in OFFENSIVE_SCORING_KEYS}


def scoring_signature_text(signature: dict[str, float]) -> str:
    """Serialize a scoring signature deterministically for storage and UI."""

    return json.dumps(signature, sort_keys=True, separators=(",", ":"))


def donor_scoring_mismatches(
    league: LeagueSettings,
    expected_signature: dict[str, float],
) -> tuple[list[str], list[str]]:
    """Return exact signature mismatches plus unusual scoring flags."""

    league_signature = scoring_signature(league.scoring_settings)
    mismatches = [
        key
        for key in OFFENSIVE_SCORING_KEYS
        if abs(float(league_signature.get(key, 0.0)) - float(expected_signature.get(key, 0.0))) > 1e-9
    ]
    normalized = normalize_scoring_settings(league.scoring_settings)
    flagged = [key for key in FLAGGED_SCORING_KEYS if abs(float(normalized.get(key, 0.0))) > 1e-9]
    return mismatches, flagged


def donor_quality_score(coverage, flagged_keys: list[str]) -> float:
    """Rank accepted donor leagues by coverage depth and cleanliness."""

    depth_score = (
        0.2 * float(coverage.deepest_rank_by_position.get("QB", 0))
        + 0.3 * float(coverage.deepest_rank_by_position.get("RB", 0))
        + 0.3 * float(coverage.deepest_rank_by_position.get("WR", 0))
        + 0.2 * float(coverage.deepest_rank_by_position.get("TE", 0))
    )
    completeness_score = 0.05 * float(coverage.unique_player_weeks) + 0.1 * float(coverage.weeks_loaded)
    penalty = 25.0 * float(len(flagged_keys))
    return round(depth_score + completeness_score - penalty, 3)


def _normalized_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def normalize_donor_scoring_format(value: Any) -> str | None:
    """Map donor CSV labels onto the supported scoring format keys."""

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    token = _normalized_token(value)
    if not token:
        return None
    for scoring_format, aliases in SCORING_FORMAT_ALIASES.items():
        if token in aliases:
            return scoring_format
    return None


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
    """Read a donor source file while adapting to headered CSVs or headerless TSV exports."""

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
    first_tokens = [token.strip().strip('"') for token in first_line.split(delimiter)]
    header_aliases = {alias for aliases in DONOR_COLUMN_ALIASES.values() for alias in aliases}
    has_header = any(token.lower() in header_aliases for token in first_tokens[:4])

    if has_header:
        return pd.read_csv(donor_path, sep=delimiter)

    raw = pd.read_csv(donor_path, sep=delimiter, header=None)
    if raw.shape[1] < len(DONOR_EXPORT_COLUMNS):
        raise ConfigError(
            "Historical donor source is headerless but does not have enough columns to match the expected donor export schema."
        )
    trimmed = raw.iloc[:, : len(DONOR_EXPORT_COLUMNS)].copy()
    trimmed.columns = DONOR_EXPORT_COLUMNS
    return trimmed


def _coerce_json_donor_entries(entries: Any) -> list[dict[str, Any]]:
    if isinstance(entries, (str, int)):
        return [{"league_id": str(entries)}]
    if isinstance(entries, dict):
        if "league_id" in entries:
            return [entries]
        if "league_ids" in entries and isinstance(entries["league_ids"], list):
            return [{"league_id": str(league_id)} for league_id in entries["league_ids"]]
        raise ConfigError("Historical donor JSON entries must provide `league_id` or `league_ids`.")
    if isinstance(entries, list):
        normalized: list[dict[str, Any]] = []
        for item in entries:
            if isinstance(item, dict):
                if "league_id" not in item:
                    raise ConfigError("Historical donor JSON donor objects must include `league_id`.")
                normalized.append(item)
            else:
                normalized.append({"league_id": str(item)})
        return normalized
    raise ConfigError("Historical donor JSON cells must be a string, object, or list.")


def _read_year_keyed_donor_json(donor_path: Path) -> pd.DataFrame:
    """Read a compact donor configuration keyed by season and scoring format."""

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
            raise ConfigError(f"Historical donor JSON season `{season}` must map to a scoring-format object.")
        for scoring_key, entries in season_payload.items():
            scoring_format = normalize_donor_scoring_format(scoring_key)
            if scoring_format is None:
                raise ConfigError(f"Historical donor JSON has an unrecognized scoring format `{scoring_key}` in season {season}.")
            for entry in _coerce_json_donor_entries(entries):
                rows.append(
                    {
                        "league_id": str(entry["league_id"]),
                        "season": season,
                        "scoring_format": scoring_format,
                        "selected": bool(entry.get("selected", True)),
                    }
                )
    return pd.DataFrame(rows)


def _missing_cells_message(cells: list[tuple[int, str]]) -> str:
    formatted = ", ".join(f"{season} {scoring_format}" for season, scoring_format in cells)
    return f"Historical donor source is missing required cells: {formatted}"


def _normalize_inline_donor_configuration(
    donors: pd.DataFrame,
    *,
    today: date,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if donors.empty:
        raise ConfigError("Historical donor source does not contain any donor rows.")
    required_columns = {"league_id", "season", "scoring_format"}
    missing = sorted(required_columns - set(donors.columns))
    if missing:
        raise ConfigError(
            "Historical donor configuration is missing required columns: "
            f"{', '.join(missing)}"
        )

    normalized = donors.copy()
    normalized["league_id"] = normalized["league_id"].astype(str).str.strip()
    normalized["season"] = pd.to_numeric(normalized["season"], errors="coerce")
    normalized["scoring_format"] = normalized["scoring_format"].map(normalize_donor_scoring_format)
    normalized["selected"] = normalized["selected"].map(_coerce_selected) if "selected" in normalized.columns else True
    normalized["source_row"] = normalized["source_row"] if "source_row" in normalized.columns else pd.RangeIndex(start=2, stop=len(normalized) + 2)

    invalid_seasons = normalized[normalized["season"].isna()]
    if not invalid_seasons.empty:
        rows = ", ".join(str(int(value)) for value in invalid_seasons["source_row"].head(5))
        raise ConfigError(f"Historical donor configuration has invalid seasons on rows: {rows}")

    unknown_formats = normalized[normalized["scoring_format"].isna()]
    if not unknown_formats.empty:
        rows = ", ".join(str(int(value)) for value in unknown_formats["source_row"].head(5))
        raise ConfigError(f"Historical donor configuration has unrecognized scoring-format labels on rows: {rows}")

    normalized["season"] = normalized["season"].astype(int)
    required_seasons = required_completed_seasons(today=today, window=4)
    valid_max_season = today.year - 1
    season_mask = normalized["season"].between(2000, valid_max_season)
    if not bool(season_mask.all()):
        bad = normalized.loc[~season_mask, "source_row"].head(5).astype(int).tolist()
        raise ConfigError(f"Historical donor configuration has out-of-range seasons on rows: {', '.join(map(str, bad))}")

    selected = normalized[normalized["selected"]].copy()
    standard_rows = selected["scoring_format"] == "standard"
    out_of_window_rows = ~selected["season"].isin(required_seasons)
    active = selected[~standard_rows & ~out_of_window_rows].copy()
    active = active.drop_duplicates(subset=["league_id", "season", "scoring_format"], keep="first").reset_index(drop=True)
    if active.empty:
        raise ConfigError(
            "Historical donor configuration does not contain any selected Half-PPR or PPR donor rows "
            f"for the required seasons {required_seasons}."
        )

    missing_cells = [
        (season, scoring_format)
        for season in required_seasons
        for scoring_format in SCORING_FORMATS
        if active[(active["season"] == season) & (active["scoring_format"] == scoring_format)].empty
    ]
    if missing_cells:
        raise ConfigError(_missing_cells_message(missing_cells))

    metadata = {
        "source": "dataframe",
        "required_seasons": required_seasons,
        "ignored_standard_rows": int(standard_rows.sum()),
        "ignored_out_of_window_rows": int(out_of_window_rows.sum()),
        "selected_rows": int(len(selected)),
        "active_rows": int(len(active)),
        "duplicate_rows_removed": int(len(selected[~standard_rows & ~out_of_window_rows]) - len(active)),
    }
    return active, metadata


def load_saved_donor_configuration(
    donor_path: Path = HISTORICAL_DONOR_FILE,
    metadata_path: Path = HISTORICAL_DONOR_METADATA_FILE,
    *,
    today: date | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load and normalize the configured donor CSV from disk."""

    today = today or date.today()
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

    if raw.empty:
        raise ConfigError(f"Historical donor source `{donor_path}` has headers but no donor rows.")

    if {"league_id", "season", "scoring_format"}.issubset(raw.columns):
        normalized = raw.copy()
        if "selected" not in normalized.columns:
            normalized["selected"] = True
        normalized["source_row"] = pd.RangeIndex(start=2, stop=len(normalized) + 2)
        active, metadata = _normalize_inline_donor_configuration(normalized, today=today)
        metadata["source"] = str(donor_path)
        if metadata_path.exists():
            metadata["saved_metadata"] = json.loads(metadata_path.read_text(encoding="utf-8"))
        return active, metadata

    column_map: dict[str, str] = {}
    missing: list[str] = []
    for logical_name, aliases in DONOR_COLUMN_ALIASES.items():
        column = _first_matching_column(raw.columns, aliases)
        if column is None and logical_name != "selected":
            missing.append(logical_name)
        elif column is not None:
            column_map[logical_name] = column
    if missing:
        raise ConfigError(
            "Historical donor source is missing required fields: "
            f"{', '.join(missing)}. Available columns: {', '.join(map(str, raw.columns))}"
        )

    normalized = pd.DataFrame(
        {
            "league_id": raw[column_map["league_id"]],
            "season": raw[column_map["season"]],
            "scoring_format": raw[column_map["scoring_format"]],
        }
    )
    if "selected" in column_map:
        normalized["selected"] = raw[column_map["selected"]]
    normalized["source_row"] = pd.RangeIndex(start=2, stop=len(normalized) + 2)
    active, metadata = _normalize_inline_donor_configuration(normalized, today=today)
    metadata["source"] = str(donor_path)
    if metadata_path.exists():
        metadata["saved_metadata"] = json.loads(metadata_path.read_text(encoding="utf-8"))
    return active, metadata


def save_donor_configuration(
    donors: pd.DataFrame,
    *,
    donor_path: Path = HISTORICAL_DONOR_FILE,
    metadata_path: Path = HISTORICAL_DONOR_METADATA_FILE,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Persist a donor configuration, primarily for optional discovery workflows."""

    donor_path.parent.mkdir(parents=True, exist_ok=True)
    donors.to_csv(donor_path, index=False)
    metadata_payload = metadata or {}
    metadata_path.write_text(json.dumps(metadata_payload, indent=2, sort_keys=True), encoding="utf-8")


def donor_matrix_summary(
    donors: pd.DataFrame,
    required_seasons: list[int],
    required_donors_per_cell: int = 1,
) -> pd.DataFrame:
    """Build a season x scoring matrix summary for the donor UI."""

    rows: list[dict[str, Any]] = []
    for season in required_seasons:
        row: dict[str, Any] = {"season": season}
        for scoring_format in SCORING_FORMATS:
            count = 0
            if not donors.empty:
                count = int(
                    donors[
                        (donors["selected"])
                        & (donors["season"] == season)
                        & (donors["scoring_format"] == scoring_format)
                    ]["league_id"].nunique()
                )
            row[scoring_format] = f"{count} donors {'✓' if count >= required_donors_per_cell else '…'}"
        rows.append(row)
    return pd.DataFrame(rows)


def donors_complete(
    donors: pd.DataFrame,
    required_seasons: list[int],
    required_donors_per_cell: int = 1,
) -> bool:
    """Return True when every season/scoring cell has enough selected donors."""

    if donors.empty:
        return False
    for season in required_seasons:
        for scoring_format in SCORING_FORMATS:
            count = int(
                donors[
                    (donors["selected"])
                    & (donors["season"] == season)
                    & (donors["scoring_format"] == scoring_format)
                ]["league_id"].nunique()
            )
            if count < required_donors_per_cell:
                return False
    return True


def _prepare_donor_league(
    client: SleeperClient,
    league_id: str,
    season: int,
    scoring_format: str,
    expected_signature: dict[str, float],
    *,
    modeling_config: ModelingConfig,
    source_row: int | None = None,
    discovered_from_user_id: str | None = None,
) -> dict[str, Any]:
    league = client.get_league(league_id)
    base_result = {
        "league_id": league_id,
        "season": int(season),
        "scoring_format": scoring_format,
        "selected": True,
        "source_row": source_row,
        "discovered_from_user_id": discovered_from_user_id,
    }

    if int(league.season) != int(season):
        return {"result": {**base_result, "status": "rejected", "reason": "Wrong season"}}

    mismatches, flagged_keys = donor_scoring_mismatches(league, expected_signature)
    signature_text = scoring_signature_text(scoring_signature(league.scoring_settings))
    if mismatches:
        return {
            "result": {
                **base_result,
                "status": "rejected",
                "reason": f"Scoring mismatch: {', '.join(mismatches)}",
                "scoring_signature": signature_text,
            }
        }
    if flagged_keys:
        return {
            "result": {
                **base_result,
                "status": "rejected",
                "reason": f"Unusual scoring: {', '.join(flagged_keys)}",
                "scoring_signature": signature_text,
            }
        }

    historical_league = as_historical_summary(league)
    try:
        player_weeks, weeks_loaded = load_historical_player_weeks(client, historical_league)
        enriched = enrich_player_weeks_with_metadata(player_weeks, client.get_players("nfl"))
        coverage = summarize_historical_coverage(
            enriched,
            season=int(league.season),
            weeks_loaded=weeks_loaded,
            min_games=modeling_config.min_games,
        )
        validate_historical_coverage(
            [coverage],
            min_weeks=modeling_config.min_coverage_weeks,
            min_player_weeks_by_position=modeling_config.min_player_weeks_by_position,
            min_players_by_position=modeling_config.min_players_by_position,
        )
    except HistoricalDataError as exc:
        return {
            "result": {
                **base_result,
                "status": "rejected",
                "reason": str(exc),
                "scoring_signature": signature_text,
            }
        }

    quality = donor_quality_score(coverage, flagged_keys)
    result = {
        **base_result,
        "status": "accepted",
        "reason": "Good coverage",
        "quality_score": quality,
        "weeks_loaded": int(coverage.weeks_loaded),
        "unique_player_weeks": int(coverage.unique_player_weeks),
        "unique_players": int(coverage.unique_players),
        "qb_coverage": int(coverage.deepest_rank_by_position.get("QB", 0)),
        "rb_coverage": int(coverage.deepest_rank_by_position.get("RB", 0)),
        "wr_coverage": int(coverage.deepest_rank_by_position.get("WR", 0)),
        "te_coverage": int(coverage.deepest_rank_by_position.get("TE", 0)),
        "team_count": int(league.total_rosters),
        "league_name": league.name,
        "scoring_signature": signature_text,
    }
    return {"result": result, "league": league, "player_weeks": enriched, "coverage": coverage, "weeks_loaded": weeks_loaded}


def validate_donor_league(
    client: SleeperClient,
    league_id: str,
    season: int,
    scoring_format: str,
    expected_signature: dict[str, float],
    modeling_config: ModelingConfig | None = None,
) -> dict[str, Any]:
    """Validate one league as a donor candidate for a season/scoring cell."""

    prepared = _prepare_donor_league(
        client,
        league_id=league_id,
        season=season,
        scoring_format=scoring_format,
        expected_signature=expected_signature,
        modeling_config=modeling_config or default_modeling_config(),
    )
    return prepared["result"]


def validate_historical_donor_configuration(
    client: SleeperClient,
    canonical_leagues: dict[str, str],
    *,
    donor_configuration: pd.DataFrame | None = None,
    donor_path: Path = HISTORICAL_DONOR_FILE,
    today: date | None = None,
    modeling_config: ModelingConfig | None = None,
) -> dict[str, Any]:
    """Load, normalize, and validate the configured donor source against Sleeper."""

    today = today or date.today()
    modeling_config = modeling_config or default_modeling_config()
    required_seasons = required_completed_seasons(today=today, window=4)
    if donor_configuration is None:
        candidates, source_metadata = load_saved_donor_configuration(donor_path=donor_path, today=today)
    else:
        candidates, source_metadata = _normalize_inline_donor_configuration(donor_configuration, today=today)

    signatures = canonical_scoring_signatures(canonical_leagues, client)
    accepted_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    prepared_donors: list[dict[str, Any]] = []

    for donor in candidates.itertuples(index=False):
        prepared = _prepare_donor_league(
            client,
            league_id=str(donor.league_id),
            season=int(donor.season),
            scoring_format=str(donor.scoring_format),
            expected_signature=signatures[str(donor.scoring_format)],
            modeling_config=modeling_config,
            source_row=int(donor.source_row) if pd.notna(donor.source_row) else None,
        )
        result = prepared["result"]
        if result["status"] == "accepted":
            accepted_rows.append(result)
            prepared_donors.append(prepared)
        else:
            rejected_rows.append(result)

    accepted = pd.DataFrame(accepted_rows, columns=DONOR_RESULT_COLUMNS)
    rejected = pd.DataFrame(rejected_rows, columns=DONOR_RESULT_COLUMNS)
    if not accepted.empty:
        accepted["selected"] = True
    matrix = donor_matrix_summary(accepted, required_seasons, required_donors_per_cell=1)
    missing_cells = [
        (season, scoring_format)
        for season in required_seasons
        for scoring_format in SCORING_FORMATS
        if accepted[(accepted["season"] == season) & (accepted["scoring_format"] == scoring_format)].empty
    ]

    return {
        "source": donor_path if donor_configuration is None else None,
        "source_metadata": source_metadata,
        "required_seasons": required_seasons,
        "accepted": accepted,
        "rejected": rejected,
        "matrix": matrix,
        "missing_cells": missing_cells,
        "signatures": {key: scoring_signature_text(value) for key, value in signatures.items()},
        "prepared_donors": prepared_donors,
    }


def require_complete_validated_donors(validation_bundle: dict[str, Any]) -> None:
    """Raise when donor validation leaves a required season/scoring cell uncovered."""

    missing_cells = validation_bundle.get("missing_cells", [])
    if missing_cells:
        raise ConfigError(_missing_cells_message(missing_cells))


def build_historical_player_weeks_from_donor_validation(
    validation_bundle: dict[str, Any],
) -> dict[str, Any]:
    """Combine validated donor frames into per-format historical scoring datasets."""

    require_complete_validated_donors(validation_bundle)
    prepared_donors = validation_bundle.get("prepared_donors", [])
    if not prepared_donors:
        raise ConfigError("No validated historical donors are available.")

    combined_by_format: dict[str, pd.DataFrame] = {}
    coverage_by_format: dict[str, list[Any]] = {key: [] for key in SCORING_FORMATS}
    league_records: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []

    for scoring_format in SCORING_FORMATS:
        frames: list[pd.DataFrame] = []
        for prepared in prepared_donors:
            result = prepared["result"]
            if result["scoring_format"] != scoring_format:
                continue
            frames.append(prepared["player_weeks"])
            coverage_by_format[scoring_format].append(prepared["coverage"])
            league_records.append(
                {
                    "season": int(result["season"]),
                    "scoring_format": scoring_format,
                    "league_id": str(result["league_id"]),
                    "weeks_loaded": int(result["weeks_loaded"]),
                    "unique_players": int(result["unique_players"]),
                }
            )
        if not frames:
            continue
        combined = pd.concat(frames, ignore_index=True)
        duplicate_groups = (
            combined.groupby(["season", "week", "player_id"])["fantasy_points"]
            .nunique()
            .reset_index(name="value_count")
        )
        bad = duplicate_groups[duplicate_groups["value_count"] > 1]
        if not bad.empty:
            conflicts.extend(bad.to_dict(orient="records"))
            continue
        combined = combined.sort_values(["season", "week", "player_id", "league_id"]).drop_duplicates(
            subset=["season", "week", "player_id"],
            keep="first",
        )
        combined_by_format[scoring_format] = combined.reset_index(drop=True)

    if conflicts:
        raise ConfigError(
            "Historical donor leagues disagree on duplicated player-week fantasy points. "
            f"Conflicts: {conflicts[:5]}"
        )

    missing_formats = [scoring_format for scoring_format in SCORING_FORMATS if scoring_format not in combined_by_format]
    if missing_formats:
        raise ConfigError(f"Historical donors are incomplete for: {', '.join(missing_formats)}")

    return {
        "player_weeks_by_format": combined_by_format,
        "coverage_by_format": coverage_by_format,
        "league_records": pd.DataFrame(league_records),
    }


def build_historical_player_weeks_from_donors(
    client: SleeperClient,
    donors: pd.DataFrame,
    *,
    modeling_config: ModelingConfig | None = None,
) -> dict[str, Any]:
    """Backward-compatible wrapper for tests and direct donor-frame usage."""

    modeling_config = modeling_config or default_modeling_config()
    prepared_donors: list[dict[str, Any]] = []
    for donor in donors[donors["selected"]].itertuples(index=False):
        league = client.get_league(str(donor.league_id))
        historical_league = HistoricalLeagueSummary(
            league_id=str(league.league_id),
            season=int(donor.season),
            scoring_settings=league.scoring_settings,
            roster_positions=league.roster_positions,
            total_rosters=int(league.total_rosters),
            previous_league_id=league.previous_league_id,
            playoff_week_start=league.playoff_week_start,
        )
        player_weeks, weeks_loaded = load_historical_player_weeks(client, historical_league)
        enriched = enrich_player_weeks_with_metadata(player_weeks, client.get_players("nfl"))
        coverage = summarize_historical_coverage(
            enriched,
            season=int(donor.season),
            weeks_loaded=weeks_loaded,
            min_games=modeling_config.min_games,
        )
        prepared_donors.append(
            {
                "result": {
                    "league_id": str(donor.league_id),
                    "season": int(donor.season),
                    "scoring_format": str(donor.scoring_format),
                    "weeks_loaded": int(coverage.weeks_loaded),
                    "unique_players": int(coverage.unique_players),
                },
                "player_weeks": enriched,
                "coverage": coverage,
            }
        )
    validation_bundle = {
        "prepared_donors": prepared_donors,
        "missing_cells": [],
    }
    return build_historical_player_weeks_from_donor_validation(validation_bundle)


def discover_historical_donors(
    client: SleeperClient,
    canonical_leagues: dict[str, str],
    seed_user: str,
    *,
    max_users: int = 200,
    max_leagues: int = 1000,
    preferred_donors_per_cell: int = 3,
    max_depth: int = 3,
    today: date | None = None,
    modeling_config: ModelingConfig | None = None,
) -> dict[str, Any]:
    """Breadth-first crawl the Sleeper user graph and collect validated donors."""

    today = today or date.today()
    modeling_config = modeling_config or default_modeling_config()
    required_seasons = required_completed_seasons(today=today, window=4)
    signatures = canonical_scoring_signatures(canonical_leagues, client)

    seed_payload = client.get_user(seed_user)
    seed_user_id = str(seed_payload["user_id"])

    queue: deque[tuple[str, int]] = deque([(seed_user_id, 0)])
    seen_users: set[str] = set()
    seen_leagues: set[str] = set()
    accepted_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    progress: list[dict[str, Any]] = []
    user_leagues_cache: dict[tuple[str, int], list[LeagueSettings]] = {}
    league_users_cache: dict[str, list[dict[str, Any]]] = {}

    while queue and len(seen_users) < max_users and len(seen_leagues) < max_leagues:
        user_id, depth = queue.popleft()
        if user_id in seen_users or depth > max_depth:
            continue
        seen_users.add(user_id)
        progress.append({"event": "visit_user", "user_id": user_id, "depth": depth})

        for season in [today.year, *required_seasons]:
            cache_key = (user_id, int(season))
            if cache_key not in user_leagues_cache:
                user_leagues_cache[cache_key] = client.get_user_leagues(user_id=user_id, season=season)
            for league in user_leagues_cache[cache_key]:
                if league.league_id in seen_leagues:
                    continue
                if len(seen_leagues) >= max_leagues:
                    break
                seen_leagues.add(league.league_id)
                progress.append({"event": "visit_league", "league_id": league.league_id, "season": league.season})

                if int(league.season) in required_seasons:
                    scoring_format = detect_reception_format(league)
                    if scoring_format in signatures:
                        prepared = _prepare_donor_league(
                            client,
                            league_id=league.league_id,
                            season=int(league.season),
                            scoring_format=scoring_format,
                            expected_signature=signatures[scoring_format],
                            modeling_config=modeling_config,
                            discovered_from_user_id=user_id,
                        )
                        result = prepared["result"]
                        if result["status"] == "accepted":
                            result["selected"] = False
                            accepted_rows.append(result)
                        else:
                            rejected_rows.append(result)

                if depth >= max_depth:
                    continue
                if league.league_id not in league_users_cache:
                    league_users_cache[league.league_id] = client.get_league_users(league.league_id)
                for user_payload in league_users_cache[league.league_id]:
                    next_user_id = str(user_payload.get("user_id") or "")
                    if next_user_id and next_user_id not in seen_users:
                        queue.append((next_user_id, depth + 1))

        selected = select_top_donors(pd.DataFrame(accepted_rows), required_seasons, preferred_donors_per_cell)
        if donors_complete(selected, required_seasons, preferred_donors_per_cell):
            accepted_rows = selected.to_dict(orient="records")
            break

    accepted = pd.DataFrame(accepted_rows, columns=DONOR_RESULT_COLUMNS)
    selected = select_top_donors(accepted, required_seasons, preferred_donors_per_cell)
    rejected = pd.DataFrame(rejected_rows, columns=DONOR_RESULT_COLUMNS)
    matrix = donor_matrix_summary(selected, required_seasons, preferred_donors_per_cell)
    return {
        "seed_user_id": seed_user_id,
        "required_seasons": required_seasons,
        "preferred_donors_per_cell": preferred_donors_per_cell,
        "accepted": selected,
        "rejected": rejected,
        "matrix": matrix,
        "progress": pd.DataFrame(progress),
        "signatures": {key: scoring_signature_text(value) for key, value in signatures.items()},
        "crawl_stats": {
            "users_inspected": len(seen_users),
            "leagues_inspected": len(seen_leagues),
            "max_users": max_users,
            "max_leagues": max_leagues,
            "max_depth": max_depth,
        },
    }


def select_top_donors(
    donors: pd.DataFrame,
    required_seasons: list[int],
    preferred_donors_per_cell: int,
) -> pd.DataFrame:
    """Select the best donors per season/scoring cell by quality score."""

    if donors.empty:
        empty = donors.copy()
        if "selected" not in empty.columns:
            empty["selected"] = pd.Series(dtype=bool)
        return empty
    ordered = donors.sort_values(
        ["season", "scoring_format", "quality_score", "league_id"],
        ascending=[True, True, False, True],
    ).copy()
    ordered["selected"] = False
    for season in required_seasons:
        for scoring_format in SCORING_FORMATS:
            mask = (ordered["season"] == season) & (ordered["scoring_format"] == scoring_format)
            top_index = ordered[mask].head(preferred_donors_per_cell).index
            ordered.loc[top_index, "selected"] = True
    return ordered.reset_index(drop=True)
