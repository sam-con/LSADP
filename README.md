# League-Specific ADP

League-Specific ADP is a Streamlit app and research toolkit for turning a saved canonical ADP snapshot into a Sleeper league-specific draft board.

## Current Architecture

```text
Historical side

donor_leagues / historical_donors_by_year
        ->
Sleeper historical players_points
        ->
Half-PPR / PPR production curves


Market side

BeatADP platform-adp
        ->
Sleeper ADP column only
        ->
Saved canonical ADP CSVs


Model

Historical production curves
        +
Canonical roster settings
        ->
Canonical VORP environments
        ->
Compare against saved BeatADP Sleeper ADPs
        ->
Select / validate transformation
        ->
Save production model
```

## Canonical ADP Source

Canonical ADP now comes from BeatADP's Sleeper platform ADP page:

```text
https://www.beatadp.com/platform-adp
```

Only BeatADP's Sleeper ADP values are used. The app does not substitute:

- BeatADP consensus ADP
- ESPN ADP
- FantasyPros ADP
- table rank
- positional rank
- FantasyCalc rankings

FantasyCalc `overallRank` remains a diagnostic-only field in legacy code and is never treated as ADP.

## Canonical Markets

The system supports up to four canonical markets:

- `1qb_half_ppr`
- `1qb_ppr`
- `sf_half_ppr`
- `sf_ppr`

V1 calibration requires this minimum viable set:

- `1qb_half_ppr`
- `1qb_ppr`
- `sf_half_ppr`

If BeatADP does not expose `sf_ppr`, calibration still proceeds with those three markets. That yields:

- 6 directed validations with 3 markets
- 12 directed validations with 4 markets

## Saved Canonical ADP Files

BeatADP refresh writes canonical ADP snapshots to:

```text
data/baseline/adp_1qb_half_ppr.csv
data/baseline/adp_1qb_ppr.csv
data/baseline/adp_sf_half_ppr.csv
data/baseline/adp_sf_ppr.csv
data/baseline/adp_metadata.json
```

`adp_sf_ppr.csv` exists only when BeatADP exposes that market.

The metadata file records:

- source URL
- fetch timestamp
- available environments
- missing environments
- player counts
- matching diagnostics
- parser version
- validation details

The public app reads these saved files only. It does not scrape BeatADP live.

## Public Runtime

Normal runtime is:

```text
Load production structural model
        +
Load saved BeatADP canonical ADP snapshot
        +
User enters Sleeper league ID
        ->
Choose nearest available canonical anchor
        ->
Load four completed Sleeper seasons
        ->
Recompute target production curves and scarcity
        ->
Transform canonical ADP into league-specific ADP
```

If a public target league is Superflex PPR but `sf_ppr` is unavailable, the runtime falls back explicitly to `sf_half_ppr` rather than crossing to a 1QB anchor.

## Development Workflow

Enable the hidden Development page in `src/config.py`:

```python
SHOW_DEVELOPMENT_PAGE = True
```

From the Development page you can:

- refresh BeatADP canonical ADPs
- inspect available canonical markets
- validate historical donor leagues
- build a candidate model
- validate the candidate
- promote the candidate to production

Candidate and production artifacts are saved under:

```text
data/baseline/candidate/
data/baseline/production/
```

## Historical Data Constraints

The historical side of the model is unchanged:

- historical scoring comes from Sleeper `players_points`
- the app uses the previous four completed NFL seasons
- with the app date pinned to August 19, 2026, that window is `2022-2025`
- scoring must remain consistent across that window
- sparse or malformed history fails clearly

## Testing

Run the suite with:

```bash
python -m pytest -q
```

The suite is mocked and does not require live BeatADP or live Sleeper access.

Current coverage includes:

- BeatADP parsing from saved HTML fixtures
- rejection of missing Sleeper ADP values
- no substitution from consensus or other columns
- three-market and four-market calibration
- dynamic validation counts
- explicit Superflex fallback behavior
- legacy guardrails that keep FantasyCalc rankings distinct from ADP

## Sources

- BeatADP platform ADP page: https://www.beatadp.com/platform-adp
- Sleeper API: https://docs.sleeper.com/
- FantasyCalc homepage: https://fantasycalc.com/
