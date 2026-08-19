"""Streamlit app for league-specific fantasy football ADP."""

from __future__ import annotations

import json
from datetime import date
from typing import Any

import pandas as pd
import streamlit as st

from src.adp import BeatADPProvider
from src.baseline_artifacts import CanonicalArtifactManager
from src.canonical import CANONICAL_ENVIRONMENTS, CANONICAL_LABELS
from src.charts import (
    build_adp_movement_chart,
    build_current_vs_adjusted_scatter,
    build_curve_chart,
    build_environment_curve_chart,
    build_error_by_bucket_chart,
    build_positional_impact_chart,
    build_validation_heatmap,
    build_validation_scatter,
)
from src.config import APP_VERSION, CANONICAL_LEAGUES, HISTORICAL_DONOR_FILE, SHOW_DEVELOPMENT_PAGE
from src.donors import donor_matrix_summary, validate_historical_donor_configuration
from src.model_builder import (
    build_aggregated_positional_errors,
    build_candidate_model,
    build_error_by_adp_bucket,
    promote_candidate_model,
    run_public_canonical_analysis,
    save_candidate_model,
    summarize_canonical_market_distinctness,
)
from src.models import ConfigError, LSADPError, LeagueSettings
from src.sleeper import SleeperClient
from src.utils import material_scoring_differences

st.set_page_config(page_title="League-Specific ADP", page_icon="🏈", layout="wide")

TODAY = date(2026, 8, 19)

CUSTOM_CSS = """
<style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(220, 252, 231, 0.7), transparent 35%),
            radial-gradient(circle at top right, rgba(254, 240, 138, 0.55), transparent 32%),
            linear-gradient(180deg, #fffdf7 0%, #f7f4ea 100%);
        color: #1f2937;
    }
    .hero {
        padding: 1.25rem 1.5rem;
        border-radius: 22px;
        background: linear-gradient(135deg, #123524 0%, #1f5137 55%, #2f6c48 100%);
        color: #f8fafc;
        box-shadow: 0 18px 48px rgba(18, 53, 36, 0.18);
        margin-bottom: 1rem;
    }
    .hero h1 {
        font-size: 2.4rem;
        margin: 0 0 0.5rem 0;
        letter-spacing: -0.03em;
    }
    .hero p {
        margin: 0;
        font-size: 1rem;
        max-width: 56rem;
        line-height: 1.5;
        color: rgba(248, 250, 252, 0.9);
    }
    .summary-card {
        background: rgba(255, 255, 255, 0.82);
        border: 1px solid rgba(31, 41, 55, 0.08);
        border-radius: 18px;
        padding: 1rem 1.1rem;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
        min-height: 102px;
    }
    .summary-label {
        text-transform: uppercase;
        font-size: 0.76rem;
        letter-spacing: 0.08em;
        color: #6b7280;
        margin-bottom: 0.35rem;
    }
    .summary-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #111827;
    }
</style>
"""


def important_league_settings(league: LeagueSettings) -> dict[str, Any]:
    starters = league.mandatory_starter_counts()
    scoring = league.scoring_settings
    return {
        "League": league.name,
        "Teams": league.total_rosters,
        "QB / RB / WR / TE": f"{starters['QB']} / {starters['RB']} / {starters['WR']} / {starters['TE']}",
        "FLEX": league.flex_slots(),
        "SUPER_FLEX": league.superflex_slots(),
        "Bench": league.bench_size(),
        "PPR": scoring.get("rec", 0.0),
        "TE Premium": scoring.get("bonus_rec_te", 0.0) + scoring.get("rec_te", 0.0),
        "Pass TD": scoring.get("pass_td", 4.0),
    }


def canonical_difference_frame(target: LeagueSettings, canonical_config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    canonical_scoring = canonical_config.get("scoring_settings", {})
    for key, canonical_value, target_value in material_scoring_differences(canonical_scoring, target.scoring_settings):
        rows.append({"type": "Scoring", "setting": key, "canonical": canonical_value, "target": target_value})

    canonical_positions = canonical_config.get("roster_positions", [])
    tracked_slots = ["QB", "RB", "WR", "TE", "FLEX", "SUPER_FLEX", "BN"]
    for slot in tracked_slots:
        canonical_count = canonical_positions.count(slot)
        target_count = target.roster_positions.count(slot)
        if canonical_count != target_count:
            rows.append({"type": "Roster", "setting": slot, "canonical": canonical_count, "target": target_count})
    canonical_team_count = canonical_config.get("team_count")
    if canonical_team_count is not None and canonical_team_count != target.total_rosters:
        rows.append({"type": "League Size", "setting": "Teams", "canonical": canonical_team_count, "target": target.total_rosters})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def cached_run_public_analysis(target_league_id: str) -> dict[str, Any]:
    return run_public_canonical_analysis(
        client=SleeperClient(),
        production_manager=CanonicalArtifactManager.production(),
        target_league_id=target_league_id,
        today=TODAY,
    )


@st.cache_data(show_spinner=False)
def cached_build_candidate_model(canonical_leagues_json: str) -> dict[str, Any]:
    return build_candidate_model(
        client=SleeperClient(),
        canonical_leagues=json.loads(canonical_leagues_json),
        today=TODAY,
    )


@st.cache_data(show_spinner=False)
def cached_validate_historical_donors(canonical_leagues_json: str) -> dict[str, Any]:
    return validate_historical_donor_configuration(
        client=SleeperClient(),
        canonical_leagues=json.loads(canonical_leagues_json),
        today=TODAY,
    )


def validate_historical_donors_live(canonical_leagues_json: str) -> dict[str, Any]:
    return validate_historical_donor_configuration(
        client=SleeperClient(),
        canonical_leagues=json.loads(canonical_leagues_json),
        today=TODAY,
    )


def load_canonical_adp_status(canonical_leagues_json: str, *, force_refresh: bool = False) -> dict[str, Any]:
    _ = canonical_leagues_json
    provider = BeatADPProvider()
    if force_refresh:
        bundle = provider.load_canonical_markets(
            force_refresh=True,
            sleeper_players_payload=SleeperClient().get_players("nfl"),
        )
    else:
        bundle = provider.load_canonical_markets(force_refresh=False)
    bundle["market_distinctness"] = (
        summarize_canonical_market_distinctness(bundle["frames"])
        if len(bundle["frames"]) >= 2
        else pd.DataFrame()
    )
    return bundle


@st.cache_data(show_spinner=False)
def cached_canonical_adp_status(canonical_leagues_json: str) -> dict[str, Any]:
    return load_canonical_adp_status(canonical_leagues_json, force_refresh=False)


def render_hero() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown(
        """
        <section class="hero">
            <h1>League-Specific ADP</h1>
            <p>
                Start from the closest real canonical ADP market, preserve what the market already knows about
                each player, and adjust only for how your Sleeper league changes positional production and scarcity.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_setting_cards(league: LeagueSettings) -> None:
    settings = important_league_settings(league)
    columns = st.columns(4)
    for index, (label, value) in enumerate(settings.items()):
        columns[index % 4].markdown(
            f"""
            <div class="summary-card">
                <div class="summary-label">{label}</div>
                <div class="summary-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def format_timestamp(value: str | None) -> str:
    if not value:
        return "N/A"
    return value.replace("T", " ").replace("+00:00", " UTC")


def render_public_results(analysis: dict[str, Any]) -> None:
    results = analysis["results"].copy()
    target_environment = analysis["target_environment"]
    artifacts = analysis["artifacts"]
    adp_source_metadata = analysis.get("adp_source_metadata", {})
    selected_key = analysis["selected_canonical_key"]
    selected_label = analysis["selected_canonical_label"]
    requested_label = analysis.get("requested_canonical_label", selected_label)
    fallback_used = bool(analysis.get("selected_canonical_fallback"))
    canonical_config = artifacts.metadata["canonical_environments"][selected_key]
    target_league = target_environment["league"]

    st.subheader("League Settings")
    render_setting_cards(target_league)
    st.caption(f"Selected canonical anchor: {selected_label}")
    if fallback_used:
        st.info(
            f"Closest exact market was {requested_label}, so the app anchored from {selected_label} because that is the nearest saved Superflex market available."
        )

    differences = canonical_difference_frame(target_league, canonical_config)
    if not differences.empty:
        st.markdown("**How This League Differs From Its Canonical Anchor**")
        st.dataframe(differences, use_container_width=True, hide_index=True)

    metrics = st.columns(4)
    metrics[0].metric("Players Modeled", len(results))
    metrics[1].metric("Canonical Anchor", selected_label)
    metrics[2].metric("Production Model", artifacts.metadata["selected_model_name"])
    metrics[3].metric(
        "Canonical ADP Snapshot",
        format_timestamp(adp_source_metadata.get("retrieved_at") or adp_source_metadata.get("recorded_at")),
    )

    st.subheader("Adjusted Rankings")
    search = st.text_input("Search players", placeholder="Filter by player name or position")
    filtered = results.copy()
    if search:
        needle = search.lower()
        filtered = filtered[
            filtered["player_name"].str.lower().str.contains(needle)
            | filtered["position"].str.lower().str.contains(needle)
        ]

    display_columns = [
        "adjusted_rank",
        "player_name",
        "position",
        "adp",
        "pos_rank",
        "league_adjusted_adp",
        "adp_change",
        "canonical_expected_ppg",
        "league_expected_ppg",
        "canonical_vorp",
        "league_vorp",
        "delta_metric",
        "short_explanation",
    ]
    rename_columns = {
        "adjusted_rank": "Adjusted Rank",
        "player_name": "Player",
        "position": "Position",
        "adp": "Current Canonical ADP",
        "pos_rank": "Current Positional Rank",
        "league_adjusted_adp": "League-Adjusted ADP",
        "adp_change": "ADP Change",
        "canonical_expected_ppg": "Canonical Expected PPG",
        "league_expected_ppg": "League Expected PPG",
        "canonical_vorp": "Canonical VORP",
        "league_vorp": "League VORP",
        "delta_metric": "Delta Metric",
        "short_explanation": "Short Explanation",
    }
    columns_to_show = [column for column in display_columns if column in filtered.columns]
    st.dataframe(filtered[columns_to_show].rename(columns=rename_columns), use_container_width=True, hide_index=True)
    st.download_button(
        "Download CSV",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name=f"league_adjusted_adp_{target_league.league_id}.csv",
        mime="text/csv",
    )

    st.subheader("Player Explanation")
    selected_player_name = st.selectbox("Select a player", options=results["player_name"].tolist(), index=0)
    selected_player = results.loc[results["player_name"] == selected_player_name].iloc[0]
    st.markdown(
        f"""
        **{selected_player['player_name']}**

        Canonical ADP Environment: `{selected_label}`  
        Current ADP: `{selected_player['adp']:.1f}`  
        League ADP: `{selected_player['league_adjusted_adp']:.1f}`  
        Adjusted Movement: `{selected_player['adp_change']:+.1f}` picks  
        Current Positional Rank: `{selected_player['position']}{int(selected_player['pos_rank'])}`  
        Canonical Expected PPG: `{selected_player.get('canonical_expected_ppg', float('nan')):.2f}`  
        League Expected PPG: `{selected_player.get('league_expected_ppg', float('nan')):.2f}`  
        Canonical VORP: `{selected_player.get('canonical_vorp', float('nan')):.2f}`  
        League VORP: `{selected_player.get('league_vorp', float('nan')):.2f}`  
        Delta Metric: `{selected_player['delta_metric']:+.2f}`

        {selected_player['short_explanation']}
        """
    )

    st.subheader("Visuals")
    position_filter = st.segmented_control("Position Filter", options=["ALL", "QB", "RB", "WR", "TE"], default="ALL")
    left, right = st.columns(2)
    left.plotly_chart(build_adp_movement_chart(results, position_filter=position_filter), use_container_width=True)
    right.plotly_chart(build_current_vs_adjusted_scatter(results), use_container_width=True)

    impact_summary = (
        results.groupby("position", as_index=False)
        .agg(
            baseline_metric=("canonical_metric", "mean"),
            target_metric=("league_metric", "mean"),
            mean_adp_change=("adp_change", "mean"),
        )
    )
    impact_summary["impact_pct"] = impact_summary.apply(
        lambda row: ((row["target_metric"] - row["baseline_metric"]) / abs(row["baseline_metric"]) * 100.0)
        if abs(row["baseline_metric"]) > 1e-6
        else 0.0,
        axis=1,
    )
    st.plotly_chart(build_positional_impact_chart(impact_summary), use_container_width=True)

    curve_position = st.selectbox("Curve Position", options=["QB", "RB", "WR", "TE"], index=0)
    source_curves = artifacts.curves[(artifacts.curves["environment_key"] == selected_key) & (artifacts.curves["dataset"] == "fitted")]
    source_empirical = artifacts.curves[(artifacts.curves["environment_key"] == selected_key) & (artifacts.curves["dataset"] == "empirical")]
    source_replacement = artifacts.replacement[
        (artifacts.replacement["environment_key"] == selected_key)
        & (artifacts.replacement["replacement_method"] == artifacts.metadata["selected_replacement_method"])
    ].set_index("position")
    target_replacement = target_environment["replacement_variants"][artifacts.metadata["selected_replacement_method"]].set_index("position")
    st.plotly_chart(
        build_curve_chart(
            baseline_curve=source_curves,
            target_curve=target_environment["evaluated_curves"][target_environment["evaluated_curves"]["dataset"] == "fitted"],
            empirical_baseline=source_empirical,
            empirical_target=target_environment["evaluated_curves"][target_environment["evaluated_curves"]["dataset"] == "empirical"],
            replacement_baseline_rank=int(source_replacement.loc[curve_position, "replacement_rank"]),
            replacement_target_rank=int(target_replacement.loc[curve_position, "replacement_rank"]),
            position=curve_position,
        ),
        use_container_width=True,
    )

    with st.expander("Advanced / Methodology"):
        st.write(
            {
                "selected_canonical_key": selected_key,
                "selected_canonical_label": selected_label,
                "requested_canonical_label": requested_label,
                "fallback_used": fallback_used,
                "selected_model_name": artifacts.metadata["selected_model_name"],
                "selected_metric_mode": artifacts.metadata["selected_metric_mode"],
                "selected_replacement_method": artifacts.metadata["selected_replacement_method"],
                "selected_utility_transform": artifacts.metadata["selected_utility_transform"],
                "selected_weight_power": artifacts.metadata["selected_weight_power"],
                "canonical_adp_source": adp_source_metadata.get("source", "BeatADP Sleeper ADP"),
                "canonical_adp_status": adp_source_metadata.get("status"),
                "canonical_adp_recorded_at": adp_source_metadata.get("recorded_at"),
                "canonical_adp_retrieved_at": adp_source_metadata.get("retrieved_at"),
            }
        )
        st.caption("Public runtime reads the saved BeatADP Sleeper canonical snapshot from disk and does not scrape BeatADP live.")
        if target_environment.get("public_runtime_mode") == "no_history":
            st.info(
                "This public analysis used the canonical anchor's saved production curves and recalculated replacement levels from your league's current roster settings. Sleeper league history was not required."
            )
        else:
            coverage_rows = [
                {
                    "season": item.season,
                    "weeks_loaded": item.weeks_loaded,
                    "unique_player_weeks": item.unique_player_weeks,
                    "unique_players": item.unique_players,
                    "players_by_position": item.unique_players_by_position,
                    "deepest_rank_by_position": item.deepest_rank_by_position,
                }
                for item in target_environment["coverage"]
            ]
            st.markdown("**Historical Sleeper Coverage**")
            st.dataframe(pd.DataFrame(coverage_rows), use_container_width=True, hide_index=True)
            st.markdown("**Target Curve Fits**")
            st.dataframe(target_environment["candidate_curves"], use_container_width=True, hide_index=True)
        st.markdown("**Target Replacement Levels**")
        st.dataframe(target_environment["replacement_variants"][artifacts.metadata["selected_replacement_method"]], use_container_width=True, hide_index=True)


def render_public_page() -> None:
    render_hero()
    try:
        production_artifacts = CanonicalArtifactManager.production().load()
    except ConfigError as exc:
        st.error(str(exc))
        st.info(
            "Configure the canonical Sleeper league IDs in `src/config.py`, refresh BeatADP canonical ADPs from the "
            "Development page, build a candidate model, and promote it to production."
        )
        return

    st.caption(f"Model version {APP_VERSION} | Production canonical model loaded from disk")
    st.write(
        {
            "production_model": production_artifacts.metadata["selected_model_name"],
            "selected_metric_mode": production_artifacts.metadata["selected_metric_mode"],
            "selected_replacement_method": production_artifacts.metadata["selected_replacement_method"],
        }
    )
    analyzed_league_id = st.session_state.get("public_analysis_league_id")
    league_id = st.text_input("Sleeper League ID", placeholder="Enter your Sleeper league ID", key="public_league_id")
    analysis = None
    if st.button("Analyze League", type="primary", use_container_width=True):
        if not league_id.strip():
            st.warning("Enter a Sleeper league ID to analyze.")
        else:
            with st.spinner("Selecting the closest canonical anchor, applying current league settings, and transforming ADP..."):
                try:
                    analysis = cached_run_public_analysis(league_id.strip())
                except LSADPError as exc:
                    st.error(str(exc))
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Unexpected failure while analyzing the league: {exc}")
                else:
                    st.session_state["public_analysis_league_id"] = league_id.strip()
                    analyzed_league_id = league_id.strip()
    elif analyzed_league_id:
        try:
            analysis = cached_run_public_analysis(analyzed_league_id)
        except LSADPError as exc:
            st.error(str(exc))
            st.session_state.pop("public_analysis_league_id", None)
            analyzed_league_id = None
        except Exception as exc:  # noqa: BLE001
            st.error(f"Unexpected failure while reloading the league analysis: {exc}")
            st.session_state.pop("public_analysis_league_id", None)
            analyzed_league_id = None
    if analysis is not None:
        if analyzed_league_id:
            st.caption(f"Showing loaded analysis for league `{analyzed_league_id}`.")
        render_public_results(analysis)
    else:
        st.info("Enter a Sleeper league ID and the app will anchor from the nearest saved BeatADP Sleeper canonical market automatically.")


def canonical_inputs() -> dict[str, str]:
    league_values: dict[str, str] = {}
    st.markdown("## Canonical Configuration")
    for environment_key in CANONICAL_ENVIRONMENTS:
        league_values[environment_key] = st.text_input(
            f"{CANONICAL_LABELS[environment_key]} League ID",
            value=CANONICAL_LEAGUES.get(environment_key, ""),
            key=f"league_{environment_key}",
        )
    return league_values


def canonical_adp_status_rows(status_bundle: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for environment_key in CANONICAL_ENVIRONMENTS:
        entry = status_bundle["formats"].get(environment_key, {})
        rows.append(
            {
                "Format": CANONICAL_LABELS[environment_key],
                "Players": entry.get("player_count"),
                "Recorded": format_timestamp(entry.get("recorded_at") or entry.get("retrieved_at")),
                "Status": entry.get("status", "unavailable"),
                "Matched": entry.get("matched_rows", entry.get("player_count")),
                "Unmatched": entry.get("unmatched_rows", 0),
                "Ambiguous": entry.get("ambiguous_rows", 0),
            }
        )
    return pd.DataFrame(rows)


def player_market_lookup(status_bundle: dict[str, Any], player_name: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for environment_key, frame in status_bundle["frames"].items():
        match = frame.loc[frame["player_name"] == player_name]
        adp_value = float(match.iloc[0]["adp"]) if not match.empty else None
        rows.append({"Format": CANONICAL_LABELS[environment_key], "ADP": adp_value})
    return pd.DataFrame(rows)


def render_beatadp_section(canonical_leagues_json: str) -> None:
    st.markdown("## BeatADP Canonical ADP")

    status_bundle = st.session_state.get("beatadp_status")
    if status_bundle is None:
        try:
            status_bundle = cached_canonical_adp_status(canonical_leagues_json)
        except LSADPError as exc:
            st.error(str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            st.error(f"Unexpected BeatADP status failure: {exc}")
            return

    refresh_col, info_col = st.columns([1, 2])
    if refresh_col.button("Refresh BeatADP Canonical ADPs", use_container_width=True):
        with st.spinner("Fetching BeatADP, matching players, validating canonical markets, and saving ADP snapshots..."):
            try:
                status_bundle = load_canonical_adp_status(canonical_leagues_json, force_refresh=True)
                st.session_state["beatadp_status"] = status_bundle
                cached_canonical_adp_status.clear()
                cached_build_candidate_model.clear()
                cached_run_public_analysis.clear()
            except LSADPError as exc:
                st.error(str(exc))
            except Exception as exc:  # noqa: BLE001
                st.error(f"Unexpected BeatADP refresh failure: {exc}")
            else:
                st.success("BeatADP canonical ADP snapshot refreshed.")

    format_entries = list(status_bundle["formats"].values())
    total_players = sum(int(entry.get("player_count") or 0) for entry in format_entries)
    info_col.write(
        {
            "provider": status_bundle["source"],
            "status": status_bundle["status"],
            "last_refresh": format_timestamp(status_bundle.get("last_refresh")),
            "available_environments": status_bundle.get("available_environments"),
            "missing_environments": status_bundle.get("missing_environments"),
            "players_loaded": total_players,
        }
    )
    info_col.caption("The Development page is the only place that fetches BeatADP live. The public app reads these saved files only.")

    st.dataframe(canonical_adp_status_rows(status_bundle), use_container_width=True, hide_index=True)

    st.markdown("### Distinctness Check")
    if status_bundle["market_distinctness"].empty:
        st.info("Distinctness requires at least two saved canonical markets.")
    else:
        st.dataframe(status_bundle["market_distinctness"], use_container_width=True, hide_index=True)

    st.markdown("### Availability")
    for environment_key in CANONICAL_ENVIRONMENTS:
        label = CANONICAL_LABELS[environment_key]
        available = environment_key in status_bundle.get("available_environments", [])
        st.write(f"{label}: {'available' if available else 'unavailable'}")

    all_players = sorted({player_name for frame in status_bundle["frames"].values() for player_name in frame["player_name"].tolist()})
    if all_players:
        selected_player = st.selectbox("Inspect BeatADP player across saved markets", all_players, key="adp_player_lookup")
        st.dataframe(player_market_lookup(status_bundle, selected_player), use_container_width=True, hide_index=True)
        first_environment_key = next(iter(status_bundle["frames"]))
        st.markdown("### First 20 Rows")
        st.dataframe(
            status_bundle["frames"][first_environment_key][["player_name", "position", "team", "adp", "pos_rank"]].head(20),
            use_container_width=True,
            hide_index=True,
        )


def render_donor_validation_section(canonical_leagues_json: str) -> None:
    st.markdown("## Historical Donor Leagues")
    st.caption(f"Source: `{HISTORICAL_DONOR_FILE.relative_to(HISTORICAL_DONOR_FILE.parent.parent.parent)}`")

    validate_col, reload_col = st.columns(2)
    if validate_col.button("Load / Validate Donor Leagues", type="primary", use_container_width=True):
        with st.spinner("Loading the curated donor config, validating each donor against Sleeper, and measuring historical coverage..."):
            try:
                bundle = validate_historical_donors_live(canonical_leagues_json)
                st.session_state["validated_donors"] = bundle
                cached_validate_historical_donors.clear()
                cached_build_candidate_model.clear()
            except LSADPError as exc:
                st.error(str(exc))
            except Exception as exc:  # noqa: BLE001
                st.error(f"Unexpected donor validation failure: {exc}")

    if reload_col.button("Reload Cached Validation", use_container_width=True):
        try:
            bundle = cached_validate_historical_donors(canonical_leagues_json)
            st.session_state["validated_donors"] = bundle
            st.success("Historical donors reloaded from the configured source.")
        except LSADPError as exc:
            st.error(str(exc))
        except Exception as exc:  # noqa: BLE001
            st.error(f"Unexpected donor reload failure: {exc}")

    validated = st.session_state.get("validated_donors")
    if validated is None:
        try:
            validated = cached_validate_historical_donors(canonical_leagues_json)
            st.session_state["validated_donors"] = validated
        except ConfigError:
            validated = None

    if validated is None:
        return

    metadata = validated.get("source_metadata", {})
    st.write(
        {
            "source": metadata.get("source"),
            "required_seasons": validated.get("required_seasons"),
            "active_rows": metadata.get("active_rows"),
            "ignored_standard_rows": metadata.get("ignored_standard_rows"),
            "ignored_out_of_window_rows": metadata.get("ignored_out_of_window_rows"),
            "duplicate_rows_removed": metadata.get("duplicate_rows_removed"),
        }
    )

    if validated.get("missing_cells"):
        missing_text = ", ".join(f"{season} {scoring_format}" for season, scoring_format in validated["missing_cells"])
        st.error(f"Required donor cells are still missing after Sleeper validation: {missing_text}")

    st.markdown("### Donor Matrix")
    st.dataframe(
        donor_matrix_summary(validated["accepted"], validated["required_seasons"], required_donors_per_cell=1),
        use_container_width=True,
        hide_index=True,
    )
    st.markdown("### Accepted Donors")
    st.dataframe(validated["accepted"], use_container_width=True, hide_index=True)
    if not validated["rejected"].empty:
        st.markdown("### Rejected Donors")
        st.dataframe(validated["rejected"], use_container_width=True, hide_index=True)


def render_candidate_diagnostics(bundle: dict[str, Any]) -> None:
    selected_validation = bundle["selected_validation"].copy()
    selected_validation["source_label"] = selected_validation["source_environment"].map(CANONICAL_LABELS)
    selected_validation["target_label"] = selected_validation["target_environment"].map(CANONICAL_LABELS)

    st.markdown("## ADP Inputs")
    st.write(bundle["adp_source_summary"])
    st.dataframe(bundle["market_distinctness"], use_container_width=True, hide_index=True)

    st.markdown("## Validation")
    st.dataframe(selected_validation, use_container_width=True, hide_index=True)
    st.dataframe(bundle["validation_by_type"], use_container_width=True, hide_index=True)
    st.dataframe(bundle["leave_one_out"], use_container_width=True, hide_index=True)

    left, right = st.columns(2)
    left.plotly_chart(build_validation_heatmap(selected_validation), use_container_width=True)

    sample_pair = next(iter(bundle["predictions"]))
    sample_prediction = bundle["predictions"][sample_pair]
    sample_actual = bundle["source_adp_by_environment"][sample_pair[1]]
    right.plotly_chart(
        build_validation_scatter(
            sample_prediction,
            sample_actual,
            f"{CANONICAL_LABELS[sample_pair[0]]} -> {CANONICAL_LABELS[sample_pair[1]]}",
        ),
        use_container_width=True,
    )

    bucket_frame = build_error_by_adp_bucket(sample_prediction, sample_actual)
    positional_errors = build_aggregated_positional_errors(bundle["predictions"], bundle["source_adp_by_environment"])
    left, right = st.columns(2)
    left.plotly_chart(build_error_by_bucket_chart(bucket_frame), use_container_width=True)
    right.dataframe(positional_errors, use_container_width=True, hide_index=True)

    st.markdown("## Production Curves")
    selected_position = st.selectbox("Canonical Curve Position", options=["QB", "RB", "WR", "TE"], key="canonical_curve_position")
    st.plotly_chart(build_environment_curve_chart(bundle["curves"], selected_position), use_container_width=True)


def render_development_page() -> None:
    st.title("Development")
    st.caption("Hidden model builder for donor-backed history, BeatADP Sleeper canonical ADPs, candidate comparison, and promotion.")

    canonical_leagues = canonical_inputs()
    leagues_json = json.dumps(canonical_leagues, sort_keys=True)

    render_beatadp_section(leagues_json)
    render_donor_validation_section(leagues_json)

    st.markdown("## Model Builder")
    build_col, validate_col, promote_col = st.columns(3)

    if build_col.button("Build Candidate Model", type="primary", use_container_width=True):
        with st.spinner("Loading active canonical leagues, validating donor history, fitting curves, calibrating specs, and saving the candidate..."):
            try:
                bundle = cached_build_candidate_model(leagues_json)
                save_candidate_model(CanonicalArtifactManager.candidate(), bundle)
                st.session_state["candidate_bundle"] = bundle
            except LSADPError as exc:
                st.error(str(exc))
            except Exception as exc:  # noqa: BLE001
                st.error(f"Unexpected candidate build failure: {exc}")
            else:
                st.success("Candidate model saved under `data/baseline/candidate/`.")

    if validate_col.button("Validate Candidate", use_container_width=True):
        with st.spinner("Rebuilding the candidate diagnostics for inspection..."):
            try:
                bundle = cached_build_candidate_model(leagues_json)
                st.session_state["candidate_bundle"] = bundle
            except LSADPError as exc:
                st.error(str(exc))
            except Exception as exc:  # noqa: BLE001
                st.error(f"Unexpected candidate validation failure: {exc}")

    if promote_col.button("Promote to Production", use_container_width=True):
        try:
            promote_candidate_model(CanonicalArtifactManager.candidate(), CanonicalArtifactManager.production())
        except LSADPError as exc:
            st.error(str(exc))
        except Exception as exc:  # noqa: BLE001
            st.error(f"Unexpected promotion failure: {exc}")
        else:
            st.success("Candidate promoted to production.")

    st.markdown("## Candidate Model")
    try:
        candidate_artifacts = CanonicalArtifactManager.candidate().load()
        st.write(
            {
                "selected_model_name": candidate_artifacts.metadata["selected_model_name"],
                "selected_metric_mode": candidate_artifacts.metadata["selected_metric_mode"],
                "selected_replacement_method": candidate_artifacts.metadata["selected_replacement_method"],
                "selected_utility_transform": candidate_artifacts.metadata["selected_utility_transform"],
                "selected_weight_power": candidate_artifacts.metadata["selected_weight_power"],
                "selected_model_score": candidate_artifacts.metadata["selected_model_score"],
                "available_canonical_environments": candidate_artifacts.metadata.get("available_canonical_environments"),
            }
        )
        st.dataframe(candidate_artifacts.model_parameters, use_container_width=True, hide_index=True)
    except ConfigError as exc:
        st.info(str(exc))

    st.markdown("## Production Model")
    try:
        production_artifacts = CanonicalArtifactManager.production().load()
        st.write(
            {
                "selected_model_name": production_artifacts.metadata["selected_model_name"],
                "selected_metric_mode": production_artifacts.metadata["selected_metric_mode"],
                "selected_replacement_method": production_artifacts.metadata["selected_replacement_method"],
                "selected_utility_transform": production_artifacts.metadata["selected_utility_transform"],
                "selected_weight_power": production_artifacts.metadata["selected_weight_power"],
                "selected_model_score": production_artifacts.metadata["selected_model_score"],
                "available_canonical_environments": production_artifacts.metadata.get("available_canonical_environments"),
            }
        )
    except ConfigError as exc:
        st.info(str(exc))

    candidate_bundle = st.session_state.get("candidate_bundle")
    if candidate_bundle is not None:
        render_candidate_diagnostics(candidate_bundle)


def render_public_page_live() -> None:
    render_hero()
    try:
        production_artifacts = CanonicalArtifactManager.production().load()
    except ConfigError as exc:
        st.error(str(exc))
        st.info(
            "Configure the canonical Sleeper league IDs in `src/config.py`, refresh BeatADP canonical ADPs from the "
            "Development page, build a candidate model, and promote it to production."
        )
        return

    st.caption(f"Model version {APP_VERSION} | Production canonical model loaded from disk | BeatADP canonical ADPs cached separately")
    st.write(
        {
            "production_model": production_artifacts.metadata["selected_model_name"],
            "selected_metric_mode": production_artifacts.metadata["selected_metric_mode"],
            "selected_replacement_method": production_artifacts.metadata["selected_replacement_method"],
            "available_canonical_environments": production_artifacts.metadata.get("available_canonical_environments"),
        }
    )
    league_id = st.text_input("Sleeper League ID", placeholder="Enter your Sleeper league ID")
    if st.button("Analyze League", type="primary", use_container_width=True):
        if not league_id.strip():
            st.warning("Enter a Sleeper league ID to analyze.")
            return
        with st.spinner("Selecting the closest canonical anchor, validating four-season history, and transforming ADP..."):
            try:
                analysis = cached_run_public_analysis(league_id.strip())
            except LSADPError as exc:
                st.error(str(exc))
                return
            except Exception as exc:  # noqa: BLE001
                st.error(f"Unexpected failure while analyzing the league: {exc}")
                return
        render_public_results(analysis)
    else:
        st.info("Enter a Sleeper league ID and the app will anchor from the nearest saved BeatADP Sleeper canonical market automatically.")


def main() -> None:
    if SHOW_DEVELOPMENT_PAGE and hasattr(st, "navigation") and hasattr(st, "Page"):
        navigation = st.navigation(
            [
                st.Page(render_public_page_live, title="League ADP", default=True),
                st.Page(render_development_page, title="Development"),
            ]
        )
        navigation.run()
        return

    if SHOW_DEVELOPMENT_PAGE:
        page = st.sidebar.radio("Page", options=["League ADP", "Development"])
        if page == "Development":
            render_development_page()
            return

    render_public_page_live()


if __name__ == "__main__":
    main()
