"""Domain models used across ingestion, modeling, and UI layers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd


class LSADPError(Exception):
    """Base application error."""


class ConfigError(LSADPError):
    """Raised when local configuration or artifacts are missing."""


class SleeperAPIError(LSADPError):
    """Raised for invalid or unavailable Sleeper API responses."""


class HistoricalDataError(LSADPError):
    """Raised when historical data is incomplete or malformed."""


class ScoringConsistencyError(HistoricalDataError):
    """Raised when league scoring changes across the required window."""


class CoverageError(HistoricalDataError):
    """Raised when historical coverage is too sparse to fit reliable curves."""


class CurveFitError(LSADPError):
    """Raised when curve fitting fails."""


@dataclass(slots=True)
class LeagueSettings:
    league_id: str
    name: str
    season: int
    total_rosters: int
    scoring_settings: dict[str, float]
    roster_positions: list[str]
    previous_league_id: str | None = None
    playoff_week_start: int | None = None

    def mandatory_starter_counts(self) -> dict[str, int]:
        counts = {position: 0 for position in ("QB", "RB", "WR", "TE")}
        for slot in self.roster_positions:
            if slot in counts:
                counts[slot] += 1
        return counts

    def flex_slots(self) -> int:
        return sum(1 for slot in self.roster_positions if slot == "FLEX")

    def superflex_slots(self) -> int:
        return sum(1 for slot in self.roster_positions if slot == "SUPER_FLEX")

    def bench_size(self) -> int:
        ignored = {"BN", "IR", "TAXI"}
        return sum(1 for slot in self.roster_positions if slot in ignored)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class HistoricalLeagueSummary:
    league_id: str
    season: int
    scoring_settings: dict[str, float]
    roster_positions: list[str]
    total_rosters: int
    previous_league_id: str | None = None
    playoff_week_start: int | None = None


@dataclass(slots=True)
class ScoringDifference:
    season: int
    key: str
    current_value: float | None
    historical_value: float | None


@dataclass(slots=True)
class HistoricalCoverage:
    season: int
    weeks_loaded: int
    unique_player_weeks: int
    unique_players: int
    unique_players_by_position: dict[str, int]
    deepest_rank_by_position: dict[str, int]


@dataclass(slots=True)
class CurveFitResult:
    position: str
    model_name: str
    a: float
    c: float
    k: float
    rmse: float
    mae: float
    r2: float
    aic: float
    bic: float
    cv_rmse: float
    historical_window: str
    replacement_rank: int | None = None
    replacement_ppg: float | None = None

    def evaluate(self, rank: float) -> float:
        return float(self.a * np.exp(-self.c * (rank - 1)) + self.k)


@dataclass(slots=True)
class ReplacementLevel:
    position: str
    method: str
    replacement_rank: int
    replacement_ppg: float


@dataclass(slots=True)
class CalibrationResult:
    position: str
    metric_name: str
    intercept: float
    coefficient: float
    r2: float
    weighted_rmse: float


@dataclass(slots=True)
class BaselineArtifacts:
    curves: pd.DataFrame
    replacement: pd.DataFrame
    model: pd.DataFrame
    metadata: dict[str, Any]


@dataclass(slots=True)
class CanonicalArtifacts:
    curves: pd.DataFrame
    replacement: pd.DataFrame
    market_calibration: pd.DataFrame
    model_parameters: pd.DataFrame
    validation: pd.DataFrame
    canonical_config: dict[str, Any]
    metadata: dict[str, Any]
    history_position_environments: pd.DataFrame = field(default_factory=pd.DataFrame)
    history_environment_seasons: pd.DataFrame = field(default_factory=pd.DataFrame)
    history_curve_models: pd.DataFrame = field(default_factory=pd.DataFrame)
    history_curves: pd.DataFrame = field(default_factory=pd.DataFrame)
    history_library_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ValidationMetrics:
    model_name: str
    spearman: float
    pearson: float
    mae: float
    median_absolute_error: float
    weighted_mae: float
    rmse: float
    pairwise_accuracy: float
    top_12_overlap: float
    top_24_overlap: float
    top_50_overlap: float
    top_100_overlap: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ModelingConfig:
    min_games: int = 4
    aggregation: str = "median"
    recency_weights: dict[int, float] = field(default_factory=dict)
    min_coverage_weeks: int = 10
    min_player_weeks_by_position: dict[str, int] = field(default_factory=dict)
    min_players_by_position: dict[str, int] = field(default_factory=dict)
    curve_selection_relative_improvement: float = 0.02
