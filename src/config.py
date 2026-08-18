"""Application configuration."""

from __future__ import annotations

from pathlib import Path

SHOW_DEVELOPMENT_PAGE = True

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
BASELINE_DIR = DATA_DIR / "baseline"
PRODUCTION_MODEL_DIR = BASELINE_DIR / "production"
CANDIDATE_MODEL_DIR = BASELINE_DIR / "candidate"
OUTPUTS_DIR = BASE_DIR / "outputs" / "validation"

ADP_1QB_PATH = DATA_DIR / "adp_1qb.csv"
ADP_SUPERFLEX_PATH = DATA_DIR / "adp_superflex.csv"

BASELINE_1QB_LEAGUE_ID = ""
BASELINE_SF_LEAGUE_ID = ""

CANONICAL_ENVIRONMENTS = (
    "1qb_standard",
    "1qb_half_ppr",
    "1qb_ppr",
    "sf_standard",
    "sf_half_ppr",
    "sf_ppr",
)

CANONICAL_LABELS = {
    "1qb_standard": "1QB Standard",
    "1qb_half_ppr": "1QB Half-PPR",
    "1qb_ppr": "1QB PPR",
    "sf_standard": "Superflex Standard",
    "sf_half_ppr": "Superflex Half-PPR",
    "sf_ppr": "Superflex PPR",
}

CANONICAL_LEAGUES = {
    "1qb_standard": "1395458028047720448",
    "1qb_half_ppr": "1395457853489176576",
    "1qb_ppr": "1395452196744622080",
    "sf_standard": "1395457482700128256",
    "sf_half_ppr": "1395457206857515008",
    "sf_ppr": "1395452598621851648",
}

CANONICAL_ADP_PATHS = {
    "1qb_standard": DATA_DIR / "adp_1qb_standard.csv",
    "1qb_half_ppr": DATA_DIR / "adp_1qb_half_ppr.csv",
    "1qb_ppr": DATA_DIR / "adp_1qb_ppr.csv",
    "sf_standard": DATA_DIR / "adp_sf_standard.csv",
    "sf_half_ppr": DATA_DIR / "adp_sf_half_ppr.csv",
    "sf_ppr": DATA_DIR / "adp_sf_ppr.csv",
}

APP_VERSION = "0.1.0"
REQUEST_TIMEOUT_SECONDS = 20
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

BASELINE_CURVES_FILE = BASELINE_DIR / "baseline_curves.csv"
BASELINE_REPLACEMENT_FILE = BASELINE_DIR / "baseline_replacement.csv"
BASELINE_MODEL_FILE = BASELINE_DIR / "baseline_model.csv"
BASELINE_METADATA_FILE = BASELINE_DIR / "baseline_metadata.json"

PRODUCTION_CURVES_FILE = PRODUCTION_MODEL_DIR / "canonical_curves.csv"
PRODUCTION_REPLACEMENT_FILE = PRODUCTION_MODEL_DIR / "canonical_replacement.csv"
PRODUCTION_MARKET_CALIBRATION_FILE = PRODUCTION_MODEL_DIR / "canonical_market_calibration.csv"
PRODUCTION_MODEL_PARAMETERS_FILE = PRODUCTION_MODEL_DIR / "model_parameters.csv"
PRODUCTION_MODEL_VALIDATION_FILE = PRODUCTION_MODEL_DIR / "model_validation.csv"
PRODUCTION_CANONICAL_LEAGUES_FILE = PRODUCTION_MODEL_DIR / "canonical_leagues.json"
PRODUCTION_METADATA_FILE = PRODUCTION_MODEL_DIR / "baseline_metadata.json"

CANDIDATE_CURVES_FILE = CANDIDATE_MODEL_DIR / "canonical_curves.csv"
CANDIDATE_REPLACEMENT_FILE = CANDIDATE_MODEL_DIR / "canonical_replacement.csv"
CANDIDATE_MARKET_CALIBRATION_FILE = CANDIDATE_MODEL_DIR / "canonical_market_calibration.csv"
CANDIDATE_MODEL_PARAMETERS_FILE = CANDIDATE_MODEL_DIR / "model_parameters.csv"
CANDIDATE_MODEL_VALIDATION_FILE = CANDIDATE_MODEL_DIR / "model_validation.csv"
CANDIDATE_CANONICAL_LEAGUES_FILE = CANDIDATE_MODEL_DIR / "canonical_leagues.json"
CANDIDATE_METADATA_FILE = CANDIDATE_MODEL_DIR / "baseline_metadata.json"
