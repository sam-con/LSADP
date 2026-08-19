"""Canonical environment selection and validation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.config import CANONICAL_ADP_PATHS, CANONICAL_ENVIRONMENTS, CANONICAL_LABELS, CANONICAL_LEAGUES
from src.models import ConfigError, LeagueSettings


def canonical_configuration(
    leagues: dict[str, str] | None = None,
    adp_paths: dict[str, Path] | None = None,
    adp_source: str | None = None,
    adp_details: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Return the configured canonical environment metadata."""

    leagues = leagues or CANONICAL_LEAGUES
    configuration: dict[str, dict[str, Any]] = {}
    for environment_key in CANONICAL_ENVIRONMENTS:
        configuration[environment_key] = {
            "label": CANONICAL_LABELS[environment_key],
            "league_id": leagues.get(environment_key, ""),
        }
        if adp_paths is not None:
            configuration[environment_key]["adp_path"] = str(adp_paths[environment_key])
        if adp_source is not None:
            configuration[environment_key]["adp_source"] = adp_source
        if adp_details is not None and environment_key in adp_details:
            configuration[environment_key]["adp_details"] = adp_details[environment_key]
    return configuration


def validate_canonical_configuration(leagues: dict[str, str], adp_paths: dict[str, Path] | None = None) -> None:
    """Ensure all canonical environments are configured explicitly."""

    missing_leagues = [environment_key for environment_key in CANONICAL_ENVIRONMENTS if not leagues.get(environment_key)]
    missing_paths = []
    if adp_paths is not None:
        missing_paths = [environment_key for environment_key in CANONICAL_ENVIRONMENTS if environment_key not in adp_paths]
    if missing_leagues or missing_paths:
        parts = []
        if missing_leagues:
            parts.append(f"missing league IDs for {', '.join(missing_leagues)}")
        if missing_paths:
            parts.append(f"missing ADP paths for {', '.join(missing_paths)}")
        raise ConfigError(f"Canonical configuration is incomplete: {'; '.join(parts)}")


def detect_qb_format(league: LeagueSettings) -> str:
    """Classify a league as 1QB or Superflex for canonical anchor purposes."""

    return "sf" if league.superflex_slots() > 0 else "1qb"


def detect_reception_value(league: LeagueSettings) -> float:
    """Return the base reception value used for canonical selection."""

    return float(league.scoring_settings.get("rec", 0.0))


def detect_reception_format(league: LeagueSettings) -> str:
    """Classify reception scoring into standard, half_ppr, or ppr."""

    value = detect_reception_value(league)
    nearest = min(
        [("standard", 0.0), ("half_ppr", 0.5), ("ppr", 1.0)],
        key=lambda candidate: abs(value - candidate[1]),
    )
    return nearest[0]


def canonical_environment_key_for_league(league: LeagueSettings) -> str:
    """Select the closest active canonical environment key for a target league."""

    qb_format = detect_qb_format(league)
    reception_value = detect_reception_value(league)
    candidates = [environment_key for environment_key in CANONICAL_ENVIRONMENTS if environment_key.startswith(f"{qb_format}_")]
    if not candidates:
        raise ConfigError(f"No canonical environments are configured for qb format `{qb_format}`.")
    return min(
        candidates,
        key=lambda environment_key: abs(
            reception_value
            - {
                "half_ppr": 0.5,
                "ppr": 1.0,
                "standard": 0.0,
            }[environment_key.split("_", 1)[1]]
        ),
    )


def classify_transformation_type(source_key: str, target_key: str) -> str:
    """Classify a directed canonical transformation."""

    source_qb, source_scoring = source_key.split("_", 1)
    target_qb, target_scoring = target_key.split("_", 1)
    same_qb = source_qb == target_qb
    same_scoring = source_scoring == target_scoring
    if same_qb and not same_scoring:
        return "Scoring-only"
    if same_scoring and not same_qb:
        return "Scarcity-only"
    return "Combined"


def directed_transform_pairs(environment_keys: tuple[str, ...] = CANONICAL_ENVIRONMENTS) -> list[tuple[str, str]]:
    """Generate all directed source -> target pairs, excluding self-transforms."""

    return [(source_key, target_key) for source_key in environment_keys for target_key in environment_keys if source_key != target_key]
