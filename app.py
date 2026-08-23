"""Streamlit entry point for the League-Specific Fantasy ADP estimator."""

from __future__ import annotations

from datetime import date

import altair as alt
import pandas as pd
import streamlit as st

from src.adp_model import estimate_adjusted_adp
from src.models import DEFAULT_REFERENCE, select_reference_league
from src.projections import projection_records_to_frame
from src.scoring import score_projection, unsupported_scoring_rules
from src.sleeper import SleeperAPIError, fetch_draft_league, fetch_projections, validate_draft_id

st.set_page_config(page_title="League ADP", page_icon="🏈", layout="wide")


def _league_teams(league: dict) -> int:
    settings = league.get("settings") or {}
    return int(settings.get("num_teams") or league.get("total_rosters") or DEFAULT_REFERENCE.teams)


def _score_frame(players: pd.DataFrame, scoring: dict, column: str) -> pd.DataFrame:
    result = players.copy()
    result[column] = result["stats"].map(lambda stats: score_projection(stats, scoring).points)
    return result


def _curve_chart(results: pd.DataFrame, position: str):
    rows = []
    subset = results[results.position == position]
    for environment, rank_col, points_col in (("Reference", "reference_pos_rank", "reference_points"), ("League", "league_pos_rank", "league_points")):
        rows.extend({"Environment": environment, "Positional rank": row[rank_col], "Projected points": row[points_col], "Player": row.player} for _, row in subset.iterrows())
    return alt.Chart(pd.DataFrame(rows)).mark_line(point=True).encode(
        x=alt.X("Positional rank:Q", sort="ascending"), y=alt.Y("Projected points:Q"), color="Environment:N",
        tooltip=["Player:N", "Environment:N", "Positional rank:Q", alt.Tooltip("Projected points:Q", format=".1f")],
    ).properties(height=270, title=f"{position} scoring curve")


def _movement_chart(results: pd.DataFrame):
    chart_data = results.copy()
    return alt.Chart(chart_data).mark_circle(size=55).encode(
        x=alt.X("current_adp:Q", title="Current Sleeper ADP", scale=alt.Scale(zero=True)),
        y=alt.Y("league_adjusted_adp:Q", title="League-adjusted ADP", scale=alt.Scale(zero=True)),
        color="position:N", tooltip=["player:N", "position:N", alt.Tooltip("current_adp:Q", format=".1f"), alt.Tooltip("league_adjusted_adp:Q", format=".1f"), alt.Tooltip("adp_change:Q", format="+.1f")],
    ).properties(height=370, title="Market ADP vs league-adjusted ADP")


def _run(draft_id: str):
    draft, league = fetch_draft_league(draft_id)
    scoring = league.get("scoring_settings") or {}
    roster = league.get("roster_positions") or []
    if not scoring or not roster:
        raise SleeperAPIError("This league is missing scoring settings or a roster configuration.")
    reference = select_reference_league(scoring, roster)
    records = fetch_projections(date.today().year)
    players = projection_records_to_frame(records, reference.adp_field)
    if players.empty:
        raise SleeperAPIError("No usable QB/RB/WR/TE projections were returned for the current season.")
    players = _score_frame(players, dict(reference.scoring_settings), "reference_points")
    players = _score_frame(players, scoring, "league_points")
    results, summaries = estimate_adjusted_adp(players, "reference_points", "league_points", list(reference.roster_positions), roster, reference.teams, _league_teams(league))
    return draft, league, results, summaries, unsupported_scoring_rules(scoring), reference


st.title("League-specific fantasy ADP")
st.caption("An interpretable V1 that recalculates Sleeper projections in your scoring environment, compares positional scarcity curves, then re-ranks the market as one board.")

with st.form("draft-form"):
    draft_id = st.text_input("Sleeper Draft ID", placeholder="e.g. 123456789012345678")
    submitted = st.form_submit_button("Estimate league ADP", type="primary")

if not submitted:
    st.info("Enter a draft ID to select the closest Sleeper market automatically: 1QB standard, half-PPR, or PPR; or Superflex/2QB PPR-market ADP.")
    st.stop()

try:
    draft_id = validate_draft_id(draft_id)
    with st.spinner("Loading Sleeper league settings, projections, and market ADP…"):
        draft, league, results, summaries, unsupported, reference = _run(draft_id)
except (ValueError, SleeperAPIError) as exc:
    st.error(str(exc))
    st.stop()
except Exception as exc:  # Last-resort UI boundary; details remain available in server logs.
    st.error(f"Could not calculate ADP: {exc}")
    st.stop()

league_name = league.get("name") or "Sleeper league"
teams = _league_teams(league)
st.subheader(league_name)
st.caption(f"{teams} teams · Draft {draft.get('draft_id', draft_id)} · {len(results)} projected QB/RB/WR/TE players")
st.info(f"Reference market selected: **{reference.name}** (Sleeper `{reference.adp_field}`). Your exact league scoring is still used for league projections and scarcity adjustments.")
if unsupported:
    st.warning("These non-zero Sleeper scoring rules are not modeled because season projections do not provide a direct matching counting stat: " + ", ".join(f"`{rule}`" for rule in unsupported) + ". Their effect is excluded from this V1 estimate.")

impact = results.groupby("position", as_index=False).agg(mean_scarcity_change=("scarcity_delta", "mean"), mean_adp_change=("adp_change", "mean"), players=("player_id", "count"))
left, right = st.columns((1, 2))
with left:
    st.markdown("#### Position impact")
    st.dataframe(impact.style.format({"mean_scarcity_change": "+.3f", "mean_adp_change": "+.1f"}), use_container_width=True, hide_index=True)
with right:
    st.altair_chart(_movement_chart(results), use_container_width=True)

st.markdown("#### Positional scoring curves")
tabs = st.tabs(["QB", "RB", "WR", "TE"])
for tab, position in zip(tabs, ("QB", "RB", "WR", "TE")):
    with tab:
        st.altair_chart(_curve_chart(results, position), use_container_width=True)
        info = summaries["league"].get(position, {})
        st.caption(f"League replacement benchmark: {position}{info.get('replacement_rank', '—')} ({info.get('replacement_points', 0):.1f} projected points).")

st.markdown("#### Player board")
positions = st.multiselect("Positions", ["QB", "RB", "WR", "TE"], default=["QB", "RB", "WR", "TE"])
view = results[results.position.isin(positions)].copy()
# Calculate this in the presentation layer as well as the model layer so an
# older cached result cannot make the table fail during a Cloud redeploy.
view["has_usable_projection"] = (view["reference_points"] > 0) | (view["league_points"] > 0)
columns = ["league_adjusted_rank", "player", "team", "position", "current_adp", "league_adjusted_adp", "adp_change", "reference_points", "league_points", "reference_pos_rank", "league_pos_rank", "league_scarcity_value", "scarcity_delta", "has_usable_projection", "market_adp_available"]
view = view[columns].rename(columns={"league_adjusted_rank": "Adjusted rank", "player": "Player", "team": "Team", "position": "Position", "current_adp": "Current ADP", "league_adjusted_adp": "League-adjusted ADP", "adp_change": "ADP change", "reference_points": "Reference points", "league_points": "League points", "reference_pos_rank": "Reference pos rank", "league_pos_rank": "League pos rank", "league_scarcity_value": "League scarcity value", "scarcity_delta": "Scarcity change", "has_usable_projection": "Usable projection", "market_adp_available": "Sleeper ADP available"})
st.dataframe(view, use_container_width=True, hide_index=True, column_config={"Current ADP": st.column_config.NumberColumn(format="%.1f"), "League-adjusted ADP": st.column_config.NumberColumn(format="%.1f"), "ADP change": st.column_config.NumberColumn(format="%+.1f"), "Reference points": st.column_config.NumberColumn(format="%.1f"), "League points": st.column_config.NumberColumn(format="%.1f"), "League scarcity value": st.column_config.NumberColumn(format="%.3f"), "Scarcity change": st.column_config.NumberColumn(format="%+.3f")})
