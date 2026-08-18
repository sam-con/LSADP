"""Shared utilities."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from src.config import CORE_POSITIONS


def normalize_scoring_settings(settings: dict[str, Any]) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for key, value in sorted(settings.items()):
        if value is None:
            continue
        try:
            normalized[key] = float(value)
        except (TypeError, ValueError):
            continue
    return normalized


def settings_signature(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def current_nfl_season(today: date | None = None) -> int:
    today = today or date.today()
    return today.year


def required_completed_seasons(today: date | None = None, window: int = 4) -> list[int]:
    season = current_nfl_season(today=today)
    return list(range(season - window, season))


def normalize_player_name(name: str) -> str:
    collapsed = re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()
    return re.sub(r"\s+", " ", collapsed)


def ensure_columns(frame: pd.DataFrame, required_columns: Iterable[str]) -> None:
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def inverse_adp_utility(utility: pd.Series | np.ndarray, transform: str = "neg_log") -> pd.Series:
    values = np.asarray(utility, dtype=float)
    if transform == "inverse_adp":
        inverted = np.where(np.abs(values) > 1e-12, 1.0 / values, np.inf)
        return pd.Series(inverted)
    restored = np.exp(-values)
    return pd.Series(restored)


def adp_utility(adp: pd.Series | np.ndarray, transform: str = "neg_log") -> pd.Series:
    clipped = np.clip(np.asarray(adp, dtype=float), 1e-6, None)
    if transform == "inverse_adp":
        return pd.Series(1.0 / clipped)
    return pd.Series(-np.log(clipped))


def adp_weights(adp: pd.Series | np.ndarray, power: float = 1.0) -> pd.Series:
    clipped = np.clip(np.asarray(adp, dtype=float), 1.0, None)
    return pd.Series(1.0 / np.power(clipped, power))


def material_scoring_differences(
    current_settings: dict[str, float],
    historical_settings: dict[str, float],
) -> list[tuple[str, float | None, float | None]]:
    keys = sorted(set(current_settings) | set(historical_settings))
    differences: list[tuple[str, float | None, float | None]] = []
    for key in keys:
        current_value = current_settings.get(key)
        historical_value = historical_settings.get(key)
        if current_value != historical_value:
            differences.append((key, current_value, historical_value))
    return differences


def position_sort_key(position: str) -> tuple[int, str]:
    try:
        return (CORE_POSITIONS.index(position), position)
    except ValueError:
        return (len(CORE_POSITIONS), position)


def rank_players_within_position(frame: pd.DataFrame, adp_column: str = "adp") -> pd.DataFrame:
    ordered = frame.sort_values([adp_column, "player_name"], ascending=[True, True]).copy()
    ordered["pos_rank"] = ordered.groupby("position").cumcount() + 1
    return ordered


def stable_rank_from_metric(frame: pd.DataFrame, column: str, ascending: bool, rank_column: str) -> pd.DataFrame:
    ordered = frame.sort_values([column, "player_name"], ascending=[ascending, True]).copy()
    ordered[rank_column] = np.arange(1, len(ordered) + 1)
    return ordered


def clamp_series(series: pd.Series, minimum: float | None = None, maximum: float | None = None) -> pd.Series:
    values = series.astype(float)
    if minimum is not None:
        values = values.clip(lower=minimum)
    if maximum is not None:
        values = values.clip(upper=maximum)
    return values


def coerce_position(position: Any) -> str | None:
    if position is None:
        return None
    normalized = str(position).upper()
    return normalized if normalized in CORE_POSITIONS else None


def correlation_safe(series_a: pd.Series, series_b: pd.Series, method: str) -> float:
    if len(series_a) < 2 or len(series_b) < 2:
        return math.nan
    return float(series_a.corr(series_b, method=method))
