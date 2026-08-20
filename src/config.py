"""Application configuration."""

from __future__ import annotations

from pathlib import Path

SHOW_DEVELOPMENT_PAGE = True

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
BASELINE_DIR = DATA_DIR / "baseline"
PRODUCTION_MODEL_DIR = BASELINE_DIR / "production"
CANDIDATE_MODEL_DIR = BASELINE_DIR / "candidate"
HISTORICAL_DONOR_FILE = BASE_DIR / "donor_leagues.csv"
CANONICAL_ADP_METADATA_FILE = BASELINE_DIR / "adp_metadata.json"

CANONICAL_ENVIRONMENTS = (
    "1qb_half_ppr",
    "1qb_ppr",
    "sf_half_ppr",
    "sf_ppr",
)

MINIMUM_VIABLE_CANONICAL_ENVIRONMENTS = (
    "1qb_half_ppr",
    "1qb_ppr",
    "sf_half_ppr",
)

CANONICAL_LABELS = {
    "1qb_half_ppr": "1QB Half-PPR",
    "1qb_ppr": "1QB PPR",
    "sf_half_ppr": "Superflex Half-PPR",
    "sf_ppr": "Superflex PPR",
}

CANONICAL_ADP_CACHE_FILES = {
    "1qb_half_ppr": "1qb_half_ppr.csv",
    "1qb_ppr": "1qb_ppr.csv",
    "sf_half_ppr": "sf_half_ppr.csv",
    "sf_ppr": "sf_ppr.csv",
}

CANONICAL_LEAGUES = {
    "1qb_half_ppr": "1395457853489176576",
    "1qb_ppr": "1395452196744622080",
    "sf_half_ppr": "1395457206857515008",
    "sf_ppr": "1395452598621851648",
}

CANONICAL_ADP_PATHS = {
    "1qb_half_ppr": BASELINE_DIR / "adp_1qb_half_ppr.csv",
    "1qb_ppr": BASELINE_DIR / "adp_1qb_ppr.csv",
    "sf_half_ppr": BASELINE_DIR / "adp_sf_half_ppr.csv",
    "sf_ppr": BASELINE_DIR / "adp_sf_ppr.csv",
}

APP_VERSION = "0.1.0"
REQUEST_TIMEOUT_SECONDS = 20
BEATADP_PLATFORM_URL = "https://www.beatadp.com/platform-adp"
BEATADP_SOURCE_NAME = "BeatADP Sleeper ADP"
BEATADP_PARSER_VERSION = "beatadp-platform-adp-v1"
REGULAR_SEASON_WEEKS = tuple(range(1, 19))
CORE_POSITIONS = ("QB", "RB", "WR", "TE")
FLEX_POSITIONS = {"RB", "WR", "TE"}
SUPERFLEX_POSITIONS = {"QB", "RB", "WR", "TE"}

DEFAULT_MIN_GAMES = 4
DEFAULT_MIN_PLAYER_WEEKS_BY_POSITION = {
    "QB": 24,
    "RB": 48,
    "WR": 48,
    "TE": 24,
}
DEFAULT_MIN_PLAYERS_BY_POSITION = {
    "QB": 8,
    "RB": 12,
    "WR": 12,
    "TE": 8,
}
DEFAULT_COVERAGE_MIN_WEEKS = 10
CURVE_SELECTION_RELATIVE_IMPROVEMENT = 0.02

PRODUCTION_CURVES_FILE = PRODUCTION_MODEL_DIR / "canonical_curves.csv"
PRODUCTION_REPLACEMENT_FILE = PRODUCTION_MODEL_DIR / "canonical_replacement.csv"
PRODUCTION_MARKET_CALIBRATION_FILE = PRODUCTION_MODEL_DIR / "canonical_market_calibration.csv"
PRODUCTION_MODEL_PARAMETERS_FILE = PRODUCTION_MODEL_DIR / "model_parameters.csv"
PRODUCTION_MODEL_VALIDATION_FILE = PRODUCTION_MODEL_DIR / "model_validation.csv"
PRODUCTION_CANONICAL_LEAGUES_FILE = PRODUCTION_MODEL_DIR / "canonical_leagues.json"
PRODUCTION_METADATA_FILE = PRODUCTION_MODEL_DIR / "baseline_metadata.json"
PRODUCTION_HISTORY_ENVIRONMENTS_FILE = PRODUCTION_MODEL_DIR / "position_scoring_environments.csv"
PRODUCTION_HISTORY_ENVIRONMENT_SEASONS_FILE = PRODUCTION_MODEL_DIR / "position_environment_seasons.csv"
PRODUCTION_HISTORY_CURVE_MODELS_FILE = PRODUCTION_MODEL_DIR / "position_curve_models.csv"
PRODUCTION_HISTORY_CURVES_FILE = PRODUCTION_MODEL_DIR / "fitted_position_curves.csv"
PRODUCTION_HISTORY_LIBRARY_METADATA_FILE = PRODUCTION_MODEL_DIR / "history_library_metadata.json"

CANDIDATE_CURVES_FILE = CANDIDATE_MODEL_DIR / "canonical_curves.csv"
CANDIDATE_REPLACEMENT_FILE = CANDIDATE_MODEL_DIR / "canonical_replacement.csv"
CANDIDATE_MARKET_CALIBRATION_FILE = CANDIDATE_MODEL_DIR / "canonical_market_calibration.csv"
CANDIDATE_MODEL_PARAMETERS_FILE = CANDIDATE_MODEL_DIR / "model_parameters.csv"
CANDIDATE_MODEL_VALIDATION_FILE = CANDIDATE_MODEL_DIR / "model_validation.csv"
CANDIDATE_CANONICAL_LEAGUES_FILE = CANDIDATE_MODEL_DIR / "canonical_leagues.json"
CANDIDATE_METADATA_FILE = CANDIDATE_MODEL_DIR / "baseline_metadata.json"
CANDIDATE_HISTORY_ENVIRONMENTS_FILE = CANDIDATE_MODEL_DIR / "position_scoring_environments.csv"
CANDIDATE_HISTORY_ENVIRONMENT_SEASONS_FILE = CANDIDATE_MODEL_DIR / "position_environment_seasons.csv"
CANDIDATE_HISTORY_CURVE_MODELS_FILE = CANDIDATE_MODEL_DIR / "position_curve_models.csv"
CANDIDATE_HISTORY_CURVES_FILE = CANDIDATE_MODEL_DIR / "fitted_position_curves.csv"
CANDIDATE_HISTORY_LIBRARY_METADATA_FILE = CANDIDATE_MODEL_DIR / "history_library_metadata.json"
