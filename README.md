# League-Specific ADP

League-Specific ADP is a Streamlit app and research toolkit for turning today's market ADP into a Sleeper league-specific draft board.

The current production architecture is:

```text
Today's FantasyCalc ADP
+ Saved calibrated structural model
+ Target Sleeper league history/settings
= League-specific ADP
```

FantasyCalc supplies the live market anchor. Sleeper supplies league settings plus four completed seasons of historical `players_points`. The saved model translates between them.

## Core Idea

The model does not try to re-rank players from scratch.

It preserves what the market already knows about talent, role, risk, and sentiment, then adjusts only for how the target league changes positional production and scarcity.

At a high level:

1. Start from the closest canonical FantasyCalc market.
2. Map ADP rank to positional production curves.
3. Derive replacement levels and VORP from Sleeper history.
4. Recompute the target league's positional scarcity.
5. Apply only the model-derived value shift created by scoring, roster construction, and league size.

## Production ADP Source

Production now uses FantasyCalc instead of manually maintained ADP CSV files.

The live endpoint used by the app is:

```text
https://api.fantasycalc.com/values/current
```

The app queries it with:

```text
isDynasty=false
includeAdp=true
numQbs=1 or 2
numTeams=<canonical team count>
ppr=0 / 0.5 / 1
```

As implemented on August 18, 2026, the provider was built against the current live response shape actually returned by the endpoint: a JSON array of player-ranking objects with nested `player` metadata, plus fields such as `overallRank`, `positionRank`, and `maybeAdp`.

The normalized internal ADP frame contains:

```text
player_id
sleeper_id
player_name
position
team
adp
canonical_format
source
retrieved_at
```

FantasyCalc Sleeper IDs are preferred over name matching.

## Six Canonical Markets

The calibration process uses six canonical markets:

- `1qb_standard`
- `1qb_half_ppr`
- `1qb_ppr`
- `sf_standard`
- `sf_half_ppr`
- `sf_ppr`

Parameter mapping:

```text
1QB Standard   -> numQbs=1, ppr=0
1QB Half-PPR   -> numQbs=1, ppr=0.5
1QB PPR        -> numQbs=1, ppr=1
SF Standard    -> numQbs=2, ppr=0
SF Half-PPR    -> numQbs=2, ppr=0.5
SF PPR         -> numQbs=2, ppr=1
```

`numQbs=2` is treated as the canonical Superflex / 2QB market. The app does not apply an extra arbitrary Superflex correction on top of that feed.

Canonical team count comes from the configured six Sleeper canonical leagues. The app validates that those six leagues share one calibration team count before using FantasyCalc live ADP for model building.

## Live ADP vs Structural Artifacts

These are intentionally separate:

### Live ADP

- Comes from FantasyCalc
- Refreshes approximately daily
- Is not baked permanently into the saved structural model

### Structural Model Artifacts

- Positional production curves
- Replacement methodology
- VORP calibration
- Selected transformation spec
- Validation outputs

These artifacts are rebuilt only from the hidden Development page when explicitly requested.

## Daily FantasyCalc Cache

FantasyCalc requests are persisted under:

```text
data/adp_cache/
    1qb_standard.csv
    1qb_half_ppr.csv
    1qb_ppr.csv
    sf_standard.csv
    sf_half_ppr.csv
    sf_ppr.csv
    metadata.json
```

`metadata.json` stores source details such as:

- last refresh time
- endpoint used
- parameters used
- team count
- player counts
- checksum
- request diagnostics

Cache behavior:

1. If the cached copy is younger than 24 hours, use it.
2. If it is stale, attempt one refresh.
3. If refresh succeeds, persist the new copy.
4. If refresh fails but a cached copy exists, fall back to cache and surface diagnostics.
5. If refresh fails and no cache exists, fail clearly.

The public UI never asks users to manage this cache. The hidden Development page exposes a manual `Refresh FantasyCalc ADP` action that bypasses the daily cache.

## Missing ADP and Diagnostics

The provider records:

- duplicate Sleeper IDs
- ambiguous name-only rows
- missing positions
- missing ADP values
- per-market fallback usage

During live implementation, some FantasyCalc redraft rows returned `maybeAdp` as null even with `includeAdp=true`. For those rows, the provider uses FantasyCalc `overallRank` as an explicit FantasyCalc-native fallback and exposes that in diagnostics rather than silently substituting another provider.

The Development page also:

- shows last refresh and cache age
- shows player counts for all six markets
- compares the six canonical feeds for suspicious similarity
- lets you inspect one player across all six markets

If supposedly different canonical markets appear effectively identical, candidate model calibration is blocked.

## Public Runtime

Normal runtime behavior is:

```text
Load saved production canonical model
+ Load today's cached FantasyCalc canonical ADPs
+ User enters Sleeper league ID
+ Select nearest canonical anchor
+ Load four completed Sleeper seasons
+ Recompute target production curves and scarcity
+ Transform today's FantasyCalc ADP
= League-specific ADP
```

Users do not need to:

- upload ADP CSVs
- pick an ADP provider
- manually refresh ADP
- configure FantasyCalc

## Development Workflow

Enable the hidden Development page by setting:

```python
SHOW_DEVELOPMENT_PAGE = True
```

From the Development page you can:

- inspect FantasyCalc ADP status and freshness
- force-refresh the FantasyCalc cache
- build a candidate six-market model
- validate the candidate
- promote a validated candidate to production

Candidate and production artifacts are saved under:

```text
data/baseline/candidate/
data/baseline/production/
```

## CSV Support

CSV ADP support still exists, but only as a debugging and test abstraction.

`CSVADPProvider` remains useful for:

- automated tests
- reproducible local fixtures
- controlled experiments

Production no longer requires manually maintained ADP CSV files.

## Historical Data Constraints

The modeling layer still enforces the original Sleeper-history rules:

- historical fantasy scoring comes from Sleeper `players_points`
- the model uses the previous four completed NFL seasons
- with the app date pinned to August 18, 2026, that window is `2022-2025`
- scoring must remain consistent across that window
- sparse or malformed history fails clearly instead of guessing

## Testing

Run the suite with:

```bash
python -m pytest -q
```

The tests are fully mocked and do not require live FantasyCalc or live Sleeper access.

Current coverage includes:

- FantasyCalc parsing and parameter mapping
- persistent cache behavior
- canonical market distinctness checks
- six-market model calibration
- runtime use of updated FantasyCalc ADP without rebuilding structural artifacts
- Sleeper history, curve fitting, replacement logic, transforms, and validation metrics

## Sources

- FantasyCalc homepage: https://fantasycalc.com/
- FantasyCalc terms of usage: https://fantasycalc.com/terms-of-usage
- FantasyCalc live endpoint used by the app: https://api.fantasycalc.com/values/current
