# League-Specific Fantasy ADP

A Streamlit V1 that estimates how a Sleeper league's scoring and roster requirements should alter a current fantasy-football ADP board. It is deliberately an interpretable scarcity model, not a trained prediction system.

## Run

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest
streamlit run app.py
```

Enter a Sleeper **draft ID**. The app fetches the draft, its associated league, the league's scoring settings and roster slots, and current season Sleeper projections. The projection feed also supplies the reference market ADP.

## Deploy to Streamlit Community Cloud

This repository is ready to deploy without secrets or external system packages.

1. Push the repository to GitHub.
2. In Streamlit Community Cloud, create an app from the repository and choose **`app.py`** as the entrypoint.
3. In **Advanced settings**, select the same supported Python version used for local development (this project was verified with Python 3.13).
4. Deploy. Community Cloud installs the root `requirements.txt` automatically.

The app only makes HTTPS requests to Sleeper's public API at runtime. It has no credentials, uploads, database, or filesystem persistence requirements. `requirements-dev.txt` is deliberately for local testing only; Community Cloud uses the production-only `requirements.txt`.

## Sleeper data inspection (August 2026)

The live season endpoint is:

`https://api.sleeper.app/projections/nfl/{season}?season_type=regular&order_by=pts_ppr`

Each record has season statistics in `stats`, player metadata (including `position`, `first_name`, `last_name`, and `team`) in `player`, and market fields including `adp_ppr`, `adp_half_ppr`, `adp_std`, and `adp_2qb`. The `v1` endpoints are used for `/draft/{draft_id}` and `/league/{league_id}`.

Observed projection statistics include direct counting fields such as `pass_yd`, `pass_td`, `pass_int`, `rush_yd`, `rush_td`, `rec`, `rec_yd`, `rec_td`, `rec_2pt`, `fum_lost`, first-down fields, and selected kicker/IDP fields. V1 focuses its ADP model on QB/RB/WR/TE, but the scoring mapper supports every direct projection key listed in `src/scoring.py`.

A live league response was also inspected before implementation. Its `scoring_settings` used the core offensive keys directly (`pass_yd`, `pass_td`, `pass_int`, `rush_yd`, `rush_td`, `rec`, `rec_yd`, `rec_td`, `fum_lost`), along with rules such as `bonus_pass_yd_300`, defensive scoring, and special-teams events. Direct keys map one-to-one; threshold/event rules without a corresponding projected counting stat are reported as unsupported.

Sleeper scoring rules whose effect needs event-level distributions—notably yardage-threshold bonuses and other rules with no matching projected counting field—are **not guessed**. The app reports them in a warning and excludes them from the calculation.

Sleeper can retain legacy ADP for inactive or retired players. V1 treats value above replacement as a floor at zero: a player below replacement—including a zero-projection legacy record—has no negative scarcity value that could turn into an artificial gain when a league changes its replacement benchmark. These records are retained and flagged in the table, but rank after every player with a usable projected point total; historical ADP cannot outweigh an absence of projected production.

## Reference league

`src/models.py` defines the reference profiles and selects the closest market automatically:

- a 1QB league selects Sleeper `adp_std`, `adp_half_ppr`, or `adp_ppr` from its `rec` scoring setting
- a league with `SUPER_FLEX`, `SUPERFLEX`, `OP`, or two direct QB slots selects Sleeper `adp_2qb`
- 1QB profiles use a typical 12-team 1QB roster; the 2QB/Superflex profile uses the documented 12-team Superflex roster
- all profiles use 4-point passing TDs, -2 interceptions, and standard 0.1 rushing/receiving yards

Sleeper's current public projection payload offers only `adp_2qb` for the Superflex/2QB market. A Superflex league with standard or half-PPR receptions therefore uses that closest available market, but the custom scoring calculation still reflects its exact rules. For unusual reception values (such as 0.25 PPR or TE premium), V1 selects the nearest conventional market and models the rest as a scarcity adjustment. Change these profiles in one place to test another baseline.

## V1 method

1. Re-score every player from projected statistical categories and the actual league's `scoring_settings`. Reference points are independently scored from the reference configuration.
2. For QB/RB/WR/TE, sort projected points and construct positional curves. The model records rank, percentile, local next-player slope, value above a dynamic replacement benchmark, and elite/replacement separation.
3. Direct starter slots come from `roster_positions`. FLEX slots are allocated RB 40% / WR 45% / TE 15%; SUPER_FLEX slots QB 45% / RB 20% / WR 25% / TE 10%. The replacement benchmark is 1.5 times estimated league-wide starter demand. These allocation assumptions are explicit and intentionally easy to replace.
4. `scarcity_value` is value above replacement divided by the median elite-to-replacement gap across positions, with a modest starter-demand weight. The pooled scale is important: a league-wide point-scale increase does not manufacture value.
5. The model uses only **the change** from reference scarcity value to league scarcity value. It adds a robustly scaled, damped version of that change to the log of the original market ADP.
6. All players are re-ranked together. Their new ranks receive the sorted original ADP curve (with tiny tie-breaking increments), preserving its practical draft shape while yielding a coherent, unique adjusted board.

Consequences by design:

- A uniform additive point increase to one position cancels out of value above replacement.
- A broad equivalent change in total fantasy-point scale is normalized away.
- A change that increases elite-to-replacement separation or starts more players at a position can alter its relative draft value.
- The original ADP remains the market prior; projections only provide an interpretable league-specific adjustment.

## Tests

The test suite checks direct scoring plus core model invariants: identity when environments match, no artificial boost from an equal additive positional increase, elite separation, starter demand, and globally unique re-ranked ADPs.

## Known V1 limitations

- Projections are a current snapshot and can change during the season.
- FLEX/SUPERFLEX allocation is a transparent heuristic, not an optimizer.
- The scarcity-to-market translation is intentionally untrained. Calibration against completed drafts and league outcomes is the next step.
- IDP, kickers, defensive units, best ball, keepers, and custom event/threshold bonuses are outside the V1 ADP model.
