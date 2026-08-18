# League-Specific ADP

League-Specific ADP is a Streamlit app and research toolkit for turning current fantasy football ADP into a Sleeper league-specific draft board.

The model starts from market ADP instead of trying to predict player quality from scratch. It uses historical Sleeper `players_points` data to estimate positional production curves, derives replacement-level scarcity from league roster settings, calibrates how VORP maps to current market draft value, and then adjusts only for the value shift created by the target league's scoring and roster rules.

## Core Idea

Current ADP already contains market beliefs about talent, role, upside, risk, age, injury concerns, and sentiment. The model preserves that information.

For each player:

1. Use current ADP to determine positional rank.
2. Map that rank onto a baseline expected production curve.
3. Convert baseline curve value into baseline VORP.
4. Recompute curve shape and replacement level for the target Sleeper league.
5. Measure the player's change in VORP.
6. Apply only that VORP-driven value shift to the player's current market utility.

The key formulas are:

```text
Baseline expected PPG:
ExpectedPPG_i = C_base(position_i, positional_rank_i)

Baseline VORP:
VORP_i^base = C_base(position_i, rank_i) - C_base(position_i, replacement_rank_i)

Target VORP:
VORP_i^target = C_target(position_i, rank_i) - C_target(position_i, replacement_rank_i)

Delta:
DeltaVORP_i = VORP_i^target - VORP_i^base

Latent market utility:
U_i = -log(ADP_i)

Per-position calibration:
U_i = alpha_position + beta_position * VORP_i^base + epsilon_i

League-adjusted utility:
U_i^target = U_i^base + beta_position * DeltaVORP_i
```

## V1 Constraints

The current implementation intentionally enforces these rules:

- Historical fantasy scoring comes only from Sleeper matchup `players_points`.
- The model uses exactly the previous four completed NFL seasons.
- With the current date of August 18, 2026, that historical window is `2022, 2023, 2024, 2025`.
- If any required historical season is missing, if `players_points` coverage is insufficient, or if scoring settings changed during that four-season window, the app stops with a clear error.
- There is no raw-stat scoring fallback and no fantasy-point reconstruction engine in V1.

## Project Structure

```text
app.py

src/
    adp.py
    analysis.py
    baseline_artifacts.py
    calibration.py
    charts.py
    config.py
    curves.py
    historical_scores.py
    models.py
    replacement.py
    sleeper.py
    sleeper_history.py
    transform.py
    utils.py
    validation.py
    vorp.py

scripts/
    validate_model.py

tests/
    conftest.py
    test_baseline_artifacts.py
    test_calibration.py
    test_curves.py
    test_historical_scores.py
    test_replacement.py
    test_sleeper.py
    test_sleeper_history.py
    test_transform.py
    test_validation.py
    test_vorp.py

data/
    adp_1qb.csv
    adp_superflex.csv
    baseline/
        baseline_curves.csv
        baseline_model.csv
        baseline_replacement.csv
        baseline_metadata.json

outputs/
    validation/

.streamlit/
    config.toml
```

## Installation

```bash
python -m pip install -r requirements.txt
```

## Required Local Configuration

Edit `src/config.py` and set:

```python
BASELINE_1QB_LEAGUE_ID = ""
BASELINE_SF_LEAGUE_ID = ""
SHOW_DEVELOPMENT_PAGE = False
```

Use:

- `BASELINE_1QB_LEAGUE_ID` for your representative standard 1QB Sleeper league.
- `BASELINE_SF_LEAGUE_ID` for your representative Superflex Sleeper league.
- `SHOW_DEVELOPMENT_PAGE = True` when you want to expose the hidden developer workflow inside Streamlit.

## ADP File Requirements

Place your current ADP files at:

- `data/adp_1qb.csv`
- `data/adp_superflex.csv`

Expected schema:

```text
player_id,player_name,position,team,adp
```

`player_id` is optional but strongly preferred. If it is absent, the app falls back to normalized player-name matching.

The public app does not fabricate ADP data. If the files are missing or empty, it shows a configuration error.

## Baseline Artifacts

The public app is designed to load saved baseline artifacts immediately at startup and not rebuild the baseline model every time the app launches.

Artifact files:

- `data/baseline/baseline_curves.csv`
- `data/baseline/baseline_model.csv`
- `data/baseline/baseline_replacement.csv`
- `data/baseline/baseline_metadata.json`

What they contain:

- `baseline_curves.csv`: fitted and empirical positional curves by rank
- `baseline_model.csv`: per-position market calibration coefficients
- `baseline_replacement.csv`: replacement ranks and replacement PPG values
- `baseline_metadata.json`: baseline IDs, historical seasons, model version, timestamps, and related context

## Building or Refreshing Baseline Artifacts

1. Set `SHOW_DEVELOPMENT_PAGE = True` in `src/config.py`.
2. Configure `BASELINE_1QB_LEAGUE_ID`.
3. Add `data/adp_1qb.csv`.
4. Run Streamlit.
5. Open the hidden `Development` page.
6. Click `Build / Refresh Baseline Model`.

That workflow:

- loads the configured baseline 1QB Sleeper league
- traverses the previous four completed seasons
- validates scoring consistency
- loads historical `players_points`
- measures historical coverage
- fits candidate curves
- selects per-position curve models
- derives starter-demand replacement levels
- calibrates market VORP exchange rates
- saves baseline artifacts under `data/baseline/`

The public page then uses those saved files directly on startup.

## Running the App

```bash
streamlit run app.py
```

Normal startup behavior is:

```text
Load current ADP
+ Load saved baseline artifacts
-> Wait for Sleeper league ID
-> Fetch target league
-> Validate four-season history and scoring consistency
-> Build target environment only
-> Display adjusted ADP
```

The app should not re-download baseline history or refit baseline curves on every launch.

## Validation Workflow

The main empirical benchmark is:

```text
1QB ADP -> predicted Superflex ADP -> compare to actual Superflex ADP
```

Run it from the CLI with:

```bash
python scripts/validate_model.py
```

Outputs are written to:

```text
outputs/validation/
```

The validation currently compares:

- `No Adjustment`
- `Curve Only`
- `Curve + Starter VORP`
- `Curve + Roster VORP`

Metrics include:

- Spearman correlation
- Pearson correlation
- MAE
- median absolute error
- weighted MAE
- RMSE
- pairwise ranking accuracy
- top-12, top-24, top-50, and top-100 overlap
- positional error tables for QB, RB, WR, and TE

## Testing

Run the automated suite with:

```bash
python -m pytest -q
```

The tests are fully mocked and do not require a live network connection.

Covered areas include:

- Sleeper league parsing
- historical league-chain traversal
- scoring-consistency validation
- `players_points` extraction and failure handling
- curve fitting and deterministic model selection
- replacement-level behavior for 1QB and Superflex
- VORP calculations
- transformation stability
- validation metrics
- baseline artifact save/load behavior

## Streamlit Page Behavior

The app has two conceptual pages:

- `League ADP`: public, user-facing page
- `Development`: hidden developer workflow

The development page is hidden when:

```python
SHOW_DEVELOPMENT_PAGE = False
```

## Historical Data Failure Behavior

The app fails clearly instead of guessing when:

- the Sleeper league ID is invalid
- the previous-league chain does not provide the required four completed seasons
- scoring settings changed across the required window
- Sleeper historical `players_points` are missing or unusable
- player coverage is too thin to fit reliable curves
- baseline artifacts are missing or malformed
- ADP files are missing or empty

## Limitations

- V1 uses only Sleeper historical `players_points`.
- The current public flow assumes your target league has four completed Sleeper seasons with unchanged scoring.
- Bench size is not used to explode scarcity values.
- Name-based ADP matching is a fallback and can still leave unmatched players if IDs are not supplied.
- Real validation depends on your actual 1QB and Superflex ADP files plus configured baseline league IDs.

## Current Repo State

This repository ships with the pipeline, app, tests, and validation tooling ready to go, but it does not include fabricated production ADP files or prebuilt baseline artifacts.

To finish setup:

1. Insert your standard 1QB league ID into `src/config.py`.
2. Insert your standard Superflex league ID into `src/config.py`.
3. Place your current 1QB ADP file at `data/adp_1qb.csv`.
4. Place your current Superflex ADP file at `data/adp_superflex.csv`.
5. Enable the Development page if you want to build baseline artifacts from the UI.
6. Build and save the baseline artifacts once.
7. Switch `SHOW_DEVELOPMENT_PAGE` back to `False` for normal public use.
