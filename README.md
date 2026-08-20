# League-Specific ADP

League-Specific ADP is a Streamlit app and model-building toolkit for turning saved canonical BeatADP Sleeper markets into league-specific draft boards.

## Current Architecture

```text
Seed leagues from donor_leagues.csv
        ->
Position-specific Sleeper scoring history library
        ->
Precomputed fitted positional curves
        ->
Replacement levels and VORP
        ->
Saved BeatADP canonical Sleeper ADP markets
        ->
Calibrated market transformation
        ->
League-specific ADP
```

The public app does not rebuild league history on demand. It uses the saved production history library plus the saved canonical BeatADP snapshot.

## Canonical ADP Source

Canonical ADP comes from BeatADP's Sleeper platform ADP page:

```text
https://www.beatadp.com/platform-adp
```

Only BeatADP's Sleeper ADP values are used. The runtime does not substitute consensus ranks or other third-party columns.

## Canonical Markets

The model supports these canonical environments:

- `1qb_half_ppr`
- `1qb_ppr`
- `sf_half_ppr`
- `sf_ppr`

The minimum viable saved market set is:

- `1qb_half_ppr`
- `1qb_ppr`
- `sf_half_ppr`

If BeatADP does not expose `sf_ppr`, the model synthesizes that corner from the other three saved markets during calibration/runtime loading.

## Saved Files

Saved canonical ADP snapshots live under `data/baseline/`:

```text
data/baseline/adp_1qb_half_ppr.csv
data/baseline/adp_1qb_ppr.csv
data/baseline/adp_sf_half_ppr.csv
data/baseline/adp_sf_ppr.csv
data/baseline/adp_metadata.json
```

Candidate and production model artifacts live under:

```text
data/baseline/candidate/
data/baseline/production/
```

Each promoted model is expected to include:

- canonical fitted curves
- replacement tables
- market calibration outputs
- validation summaries
- position history library artifacts

If the production history library artifacts are missing, the public app fails clearly and asks for a rebuild/promotion instead of falling back to legacy behavior.

## Development Workflow

Enable the hidden Development page in [src/config.py](/C:/Users/conle_tqane1n/Git/LSADP/src/config.py):

```python
SHOW_DEVELOPMENT_PAGE = True
```

From the Development page you can:

- refresh saved BeatADP canonical ADPs
- inspect canonical market distinctness
- inspect the configured seed-league file
- build a candidate model
- review candidate diagnostics
- promote the candidate to production

The seed-league source file is:

```text
donor_leagues.csv
```

## Testing

Run the suite with:

```bash
python -m pytest -q
```

The tests are mocked and cover:

- BeatADP parsing and persistence
- canonical market loading and synthetic `sf_ppr`
- position-history-library candidate builds
- public runtime transformations from saved production artifacts

## Sources

- BeatADP platform ADP page: https://www.beatadp.com/platform-adp
- Sleeper API: https://docs.sleeper.com/
