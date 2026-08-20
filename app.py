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
    build_curve_chart,
    build_environment_curve_chart,
    build_error_by_bucket_chart,
    build_validation_heatmap,
    build_validation_scatter,
)
from src.config import APP_VERSION, CANONICAL_LEAGUES, HISTORICAL_DONOR_FILE, SHOW_DEVELOPMENT_PAGE
from src.donors import load_history_seed_leagues
from src.history_library import build_league_environment_from_library
from src.model_builder import (
    build_aggregated_positional_errors,
    build_candidate_model,
    build_error_by_adp_bucket,
    promote_candidate_model,
    run_public_canonical_analysis,
    save_candidate_model,
    summarize_canonical_market_distinctness,
)
from src.models import ConfigError, LSADPError, SleeperAPIError
from src.public_ui import (
    build_biggest_fallers_frame,
    build_biggest_risers_frame,
    build_historical_match_details,
    build_historical_reference_frame,
    build_player_advanced_frame,
    build_position_impact_frame,
    build_public_download_frame,
    build_public_rankings_frame,
    filter_results_for_display,
    league_format_label,
    missing_modeled_positions,
    modeled_positions_for_league,
    public_methodology_lines,
    public_player_explanation,
    scoring_detail_lines,
    scoring_primary_label,
    scoring_summary_text,
    starting_lineup_text,
    unsupported_roster_positions,
)
from src.sleeper import SleeperClient

st.set_page_config(page_title="League-Specific ADP", page_icon="football", layout="wide")

TODAY = date(2026, 8, 19)


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


def format_timestamp(value: str | None) -> str:
    if not value:
        return "N/A"
    return value.replace("T", " ").replace("+00:00", " UTC")


def ensure_widget_choice(key: str, options: list[str], default: str | None = None) -> str | None:
    if not options:
        st.session_state.pop(key, None)
        return None
    selected = st.session_state.get(key)
    if selected not in options:
        st.session_state[key] = default if default in options else options[0]
    return str(st.session_state[key])


def public_error_message(exc: Exception) -> str:
    message = str(exc)
    lowered = message.lower()
    if isinstance(exc, SleeperAPIError) or "sleeper" in lowered:
        return "We couldn't load that Sleeper league. Check the league ID and try again."
    if "position history library" in lowered or "historical scoring environment" in lowered:
        return "Historical scoring data is not available for this league configuration yet."
    if "canonical model artifacts" in lowered or "production canonical model" in lowered:
        return "The saved production model is not ready yet. Rebuild and promote it from the Development page."
    return message


def public_number_column(label: str, width: str = "small") -> st.column_config.NumberColumn:
    return st.column_config.NumberColumn(label, format="%.1f", width=width)


def render_public_header() -> None:
    st.title("League-Specific ADP")
    st.caption("Adjust current Sleeper ADP for your league's scoring, roster settings, and positional scarcity.")


def render_league_summary(league, results: pd.DataFrame) -> None:
    st.header("League Summary")
    st.subheader(league.name)
    summary_columns = st.columns(4)
    summary_columns[0].metric("Teams", league.total_rosters)
    summary_columns[1].metric("Format", league_format_label(league))
    summary_columns[2].metric("Scoring", scoring_primary_label(league))
    summary_columns[3].metric("Players Modeled", len(results))
    st.caption(f"Starting lineup: {starting_lineup_text(league)}")
    details = scoring_detail_lines(league)
    if details:
        st.caption("Scoring details: " + " | ".join(details))


def render_public_rankings(results: pd.DataFrame, positions: list[str], league_id: str) -> pd.DataFrame:
    st.header("Scoring-Adjusted ADP")
    filter_columns = st.columns([1, 2])
    selected_positions = filter_columns[0].multiselect(
        "Position",
        options=positions,
        default=positions,
        key="public_positions",
    )
    active_positions = selected_positions or positions
    search = filter_columns[1].text_input(
        "Search player",
        placeholder="Search by player, team, or position",
        key="public_player_search",
    )

    public_rankings = build_public_rankings_frame(results, positions=active_positions, search=search)
    st.caption("Positive change means draft earlier than current Sleeper market ADP.")
    st.dataframe(
        public_rankings,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Rank": st.column_config.NumberColumn("Rank", format="%d", width="small"),
            "Pos Rank": st.column_config.NumberColumn("Pos Rank", format="%d", width="small"),
            "Market ADP": public_number_column("Market ADP"),
            "League ADP": public_number_column("League ADP"),
            "Change": public_number_column("Change"),
        },
    )
    download_frame = build_public_download_frame(results[results["position"].isin(positions)].copy())
    st.download_button(
        "Download League ADP",
        data=download_frame.to_csv(index=False).encode("utf-8"),
        file_name=f"league_adjusted_adp_{league_id}.csv",
        mime="text/csv",
        key="public_download_csv",
    )
    return active_positions


def render_biggest_movers(results: pd.DataFrame, positions: list[str]) -> None:
    filtered = results[results["position"].isin(positions)].copy()
    st.header("Biggest Risers and Fallers")
    left, right = st.columns(2)
    left.subheader("Biggest Risers")
    left.dataframe(
        build_biggest_risers_frame(filtered),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Market ADP": public_number_column("Market ADP"),
            "League ADP": public_number_column("League ADP"),
            "Change": public_number_column("Change"),
        },
    )
    right.subheader("Biggest Fallers")
    right.dataframe(
        build_biggest_fallers_frame(filtered),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Market ADP": public_number_column("Market ADP"),
            "League ADP": public_number_column("League ADP"),
            "Change": public_number_column("Change"),
        },
    )


def render_position_context(
    analysis: dict[str, Any],
    results: pd.DataFrame,
    positions: list[str],
) -> None:
    target_environment = analysis["target_environment"]
    match_summary = target_environment.get("position_match_summary", pd.DataFrame())
    st.header("League Impact")
    left, right = st.columns(2)
    left.subheader("Position Impact")
    left.dataframe(
        build_position_impact_frame(results, positions),
        use_container_width=True,
        hide_index=True,
        column_config={"Avg Change": public_number_column("Avg Change")},
    )
    right.subheader("Historical References")
    right.dataframe(
        build_historical_reference_frame(match_summary, positions),
        use_container_width=True,
        hide_index=True,
    )

    match_details = build_historical_match_details(match_summary, positions)
    if match_details:
        with st.expander("Historical scoring match details"):
            for detail in match_details:
                st.markdown(f"**{detail['position']} historical reference: {detail['match_quality']}**")
                for line in detail["differences"]:
                    st.write(f"- {line}")


def render_player_detail(
    analysis: dict[str, Any],
    results: pd.DataFrame,
    positions: list[str],
) -> None:
    target_environment = analysis["target_environment"]
    match_summary = target_environment.get("position_match_summary", pd.DataFrame())
    options_frame = results[results["position"].isin(positions)].sort_values(["adjusted_rank", "player_name"])
    player_options = options_frame["player_name"].tolist()
    if not player_options:
        return

    st.header("Player Detail")
    ensure_widget_choice("public_selected_player_name", player_options, player_options[0])
    selected_player_name = st.selectbox("Select player", options=player_options, key="public_selected_player_name")
    selected_player = options_frame.loc[options_frame["player_name"] == selected_player_name].iloc[0]

    metric_columns = st.columns(4)
    metric_columns[0].metric("Market ADP", f"{float(selected_player['adp']):.1f}")
    metric_columns[1].metric("League ADP", f"{float(selected_player['league_adjusted_adp']):.1f}")
    metric_columns[2].metric("Change", f"{float(selected_player['adp_change']):+.1f}")
    metric_columns[3].metric("Pos Rank", f"{selected_player['position']}{int(selected_player['pos_rank'])}")
    st.write(public_player_explanation(selected_player))

    player_match = match_summary[match_summary["position"] == selected_player["position"]]
    if not player_match.empty:
        match_row = player_match.iloc[0]
        st.caption(f"{selected_player['position']} historical reference: {match_row['match_quality']}")

    with st.expander("Why did this player move?"):
        st.dataframe(
            build_player_advanced_frame(selected_player),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Canonical": st.column_config.NumberColumn("Canonical", format="%.2f"),
                "League": st.column_config.NumberColumn("League", format="%.2f"),
            },
        )


def render_curve_context(analysis: dict[str, Any], results: pd.DataFrame, positions: list[str]) -> None:
    if not positions:
        return
    artifacts = analysis["artifacts"]
    target_environment = analysis["target_environment"]
    selected_key = analysis["selected_canonical_key"]
    selected_replacement_method = artifacts.metadata["selected_replacement_method"]

    st.header("Position-Level Context")
    ensure_widget_choice("public_curve_position", positions, positions[0])
    curve_position = st.selectbox("Position", options=positions, key="public_curve_position")

    source_curves = artifacts.curves[
        (artifacts.curves["environment_key"] == selected_key)
        & (artifacts.curves["dataset"] == "fitted")
        & (artifacts.curves["position"] == curve_position)
    ]
    source_empirical = artifacts.curves[
        (artifacts.curves["environment_key"] == selected_key)
        & (artifacts.curves["dataset"] == "empirical")
        & (artifacts.curves["position"] == curve_position)
    ]
    target_curves = target_environment["evaluated_curves"][
        (target_environment["evaluated_curves"]["dataset"] == "fitted")
        & (target_environment["evaluated_curves"]["position"] == curve_position)
    ]
    target_empirical = target_environment["evaluated_curves"][
        (target_environment["evaluated_curves"]["dataset"] == "empirical")
        & (target_environment["evaluated_curves"]["position"] == curve_position)
    ]
    source_replacement = artifacts.replacement[
        (artifacts.replacement["environment_key"] == selected_key)
        & (artifacts.replacement["replacement_method"] == selected_replacement_method)
    ].set_index("position")
    target_replacement = target_environment["replacement_variants"][selected_replacement_method].set_index("position")
    if curve_position not in source_replacement.index or curve_position not in target_replacement.index:
        return

    st.plotly_chart(
        build_curve_chart(
            baseline_curve=source_curves,
            target_curve=target_curves,
            empirical_baseline=source_empirical,
            empirical_target=target_empirical,
            replacement_baseline_rank=int(source_replacement.loc[curve_position, "replacement_rank"]),
            replacement_target_rank=int(target_replacement.loc[curve_position, "replacement_rank"]),
            position=curve_position,
        ),
        use_container_width=True,
    )


def render_public_methodology(analysis: dict[str, Any]) -> None:
    adp_source_metadata = analysis.get("adp_source_metadata", {})
    with st.expander("How this works"):
        for line in public_methodology_lines():
            st.write(f"- {line}")
        st.caption(
            "Saved market snapshot: "
            + format_timestamp(adp_source_metadata.get("retrieved_at") or adp_source_metadata.get("recorded_at"))
        )


def render_public_results(analysis: dict[str, Any]) -> None:
    results = analysis["results"].copy()
    target_league = analysis["target_environment"]["league"]
    positions = modeled_positions_for_league(target_league, results)
    unsupported_positions = unsupported_roster_positions(target_league)
    unavailable_positions = missing_modeled_positions(target_league, results)

    render_league_summary(target_league, results)
    if unsupported_positions:
        st.warning(
            "This version adjusts QB, RB, WR, and TE only. "
            + ", ".join(unsupported_positions)
            + " are not currently modeled."
        )
    if unavailable_positions:
        st.warning(
            "No adjusted results were produced for: " + ", ".join(unavailable_positions) + "."
        )

    active_positions = render_public_rankings(results, positions, target_league.league_id)
    filtered_results = filter_results_for_display(results, positions=active_positions)
    if not filtered_results.empty:
        st.plotly_chart(
            build_adp_movement_chart(
                filtered_results,
                position_filter=active_positions[0] if len(active_positions) == 1 else "ALL",
            ),
            use_container_width=True,
        )
    render_biggest_movers(results, active_positions)
    render_position_context(analysis, results, active_positions)
    render_player_detail(analysis, results, active_positions)
    render_curve_context(analysis, results, active_positions)
    render_public_methodology(analysis)


def render_public_page() -> None:
    render_public_header()
    try:
        CanonicalArtifactManager.production().load()
    except ConfigError as exc:
        st.error(public_error_message(exc))
        st.info("Build and promote a production model from the Development page before analyzing leagues.")
        return

    analyzed_league_id = st.session_state.get("public_analysis_league_id")
    league_id = st.text_input("Sleeper League ID", placeholder="Enter your Sleeper league ID", key="public_league_id")
    analysis = None
    if st.button("Analyze League", type="primary", use_container_width=True):
        if not league_id.strip():
            st.warning("Enter a Sleeper league ID to analyze.")
        else:
            with st.spinner("Analyzing league..."):
                try:
                    analysis = cached_run_public_analysis(league_id.strip())
                except LSADPError as exc:
                    st.error(public_error_message(exc))
                except Exception:  # noqa: BLE001
                    st.error("The app ran into an unexpected problem while analyzing this league.")
                else:
                    st.session_state["public_analysis_league_id"] = league_id.strip()
                    analyzed_league_id = league_id.strip()
                    st.session_state["public_player_search"] = ""
                    if analysis is not None and not analysis["results"].empty:
                        st.session_state["public_selected_player_name"] = str(analysis["results"].iloc[0]["player_name"])
    elif analyzed_league_id:
        try:
            analysis = cached_run_public_analysis(analyzed_league_id)
        except LSADPError as exc:
            st.error(public_error_message(exc))
            st.session_state.pop("public_analysis_league_id", None)
        except Exception:  # noqa: BLE001
            st.error("The app ran into an unexpected problem while reloading the saved league analysis.")
            st.session_state.pop("public_analysis_league_id", None)

    if analysis is None:
        st.info("Enter a Sleeper league ID to build a league-specific draft board.")
        return
    render_public_results(analysis)


def canonical_inputs() -> dict[str, str]:
    league_values: dict[str, str] = {}
    st.subheader("Canonical League Configuration")
    input_columns = st.columns(2)
    for index, environment_key in enumerate(CANONICAL_ENVIRONMENTS):
        with input_columns[index % 2]:
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
    st.subheader("Canonical ADP")
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

    if st.button("Refresh BeatADP Canonical ADPs", use_container_width=True):
        with st.spinner("Refreshing canonical ADP..."):
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
    metric_columns = st.columns(4)
    metric_columns[0].metric("Provider", status_bundle["source"])
    metric_columns[1].metric("Status", status_bundle["status"])
    metric_columns[2].metric("Last Refresh", format_timestamp(status_bundle.get("last_refresh")))
    metric_columns[3].metric("Players Loaded", total_players)
    st.caption("The Development page fetches BeatADP live. The public page reads the saved files only.")
    st.dataframe(canonical_adp_status_rows(status_bundle), use_container_width=True, hide_index=True)

    with st.expander("Market Distinctness"):
        if status_bundle["market_distinctness"].empty:
            st.info("Distinctness requires at least two saved canonical markets.")
        else:
            st.dataframe(status_bundle["market_distinctness"], use_container_width=True, hide_index=True)

    all_players = sorted({player_name for frame in status_bundle["frames"].values() for player_name in frame["player_name"].tolist()})
    if all_players:
        selected_player = st.selectbox("Inspect saved market ADP for a player", all_players, key="adp_player_lookup")
        st.dataframe(
            player_market_lookup(status_bundle, selected_player),
            use_container_width=True,
            hide_index=True,
            column_config={"ADP": public_number_column("ADP")},
        )


def render_seed_league_section() -> None:
    st.subheader("Historical Library Seeds")
    st.caption(f"Source file: `{HISTORICAL_DONOR_FILE.name}`")

    if st.button("Reload Seed League File", use_container_width=True):
        st.session_state.pop("history_seed_bundle", None)

    seed_bundle = st.session_state.get("history_seed_bundle")
    if seed_bundle is None:
        try:
            seeds, metadata = load_history_seed_leagues(today=TODAY)
        except LSADPError as exc:
            st.error(str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            st.error(f"Unexpected seed-league load failure: {exc}")
            return
        seed_bundle = {"seeds": seeds, "metadata": metadata}
        st.session_state["history_seed_bundle"] = seed_bundle

    seeds = seed_bundle["seeds"]
    metadata = seed_bundle["metadata"]
    metric_columns = st.columns(3)
    metric_columns[0].metric("Selected Rows", metadata["selected_rows"])
    metric_columns[1].metric("Unique Leagues", metadata["unique_leagues"])
    metric_columns[2].metric("Season Range", f"{metadata['season_range'][0]}-{metadata['season_range'][1]}")
    summary = (
        seeds.groupby("season", as_index=False)
        .agg(seed_leagues=("league_id", "nunique"))
        .sort_values("season")
        .reset_index(drop=True)
    )
    st.dataframe(summary, use_container_width=True, hide_index=True)
    with st.expander("Seed League Rows"):
        st.dataframe(seeds.sort_values(["season", "league_id"]).reset_index(drop=True), use_container_width=True, hide_index=True)


def render_candidate_diagnostics(bundle: dict[str, Any]) -> None:
    history_environments = bundle.get("history_position_environments", pd.DataFrame())
    history_seasons = bundle.get("history_environment_seasons", pd.DataFrame())
    selected_validation = bundle["selected_validation"].copy()
    selected_validation["Source"] = selected_validation["source_environment"].map(CANONICAL_LABELS)
    selected_validation["Target"] = selected_validation["target_environment"].map(CANONICAL_LABELS)

    if not history_environments.empty:
        st.subheader("Position History Library")
        counts = (
            history_environments.groupby(["position", "status"], as_index=False)
            .size()
            .pivot(index="position", columns="status", values="size")
            .fillna(0)
            .reset_index()
        )
        st.dataframe(counts, use_container_width=True, hide_index=True)
        if not history_seasons.empty:
            with st.expander("Seasonal Coverage"):
                st.dataframe(history_seasons, use_container_width=True, hide_index=True)

    st.subheader("Validation")
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

    st.subheader("Canonical Curves")
    selected_position = st.selectbox("Canonical curve position", options=["QB", "RB", "WR", "TE"], key="canonical_curve_position")
    st.plotly_chart(build_environment_curve_chart(bundle["curves"], selected_position), use_container_width=True)


def render_artifact_summary(title: str, artifacts) -> None:
    st.subheader(title)
    metric_columns = st.columns(4)
    metric_columns[0].metric("Selected Model", artifacts.metadata["selected_model_name"])
    metric_columns[1].metric("Replacement", artifacts.metadata["selected_replacement_method"])
    metric_columns[2].metric("Score", f"{float(artifacts.metadata['selected_model_score']):.2f}")
    metric_columns[3].metric("Environments", len(artifacts.metadata.get("available_canonical_environments", [])))
    with st.expander(f"{title} model parameters"):
        st.dataframe(artifacts.model_parameters, use_container_width=True, hide_index=True)


def render_history_match_tester(candidate_bundle: dict[str, Any]) -> None:
    st.subheader("History Match Tester")
    inspect_league_id = st.text_input("Inspect Sleeper league ID", key="history_match_league_id")
    if st.button("Inspect History Match", use_container_width=True):
        if not inspect_league_id.strip():
            st.warning("Enter a Sleeper league ID to inspect.")
        else:
            try:
                league = SleeperClient().get_league(inspect_league_id.strip())
                inspected = build_league_environment_from_library(
                    league=league,
                    library_bundle={
                        "position_scoring_environments": candidate_bundle.get("history_position_environments", pd.DataFrame()),
                        "environment_seasons": candidate_bundle.get("history_environment_seasons", pd.DataFrame()),
                        "curve_models": candidate_bundle.get("history_curve_models", pd.DataFrame()),
                        "fitted_curves": candidate_bundle.get("history_curves", pd.DataFrame()),
                    },
                    replacement_method="starter_demand",
                )
                st.dataframe(inspected["position_match_summary"], use_container_width=True, hide_index=True)
            except LSADPError as exc:
                st.error(str(exc))
            except Exception as exc:  # noqa: BLE001
                st.error(f"Unexpected history-match inspection failure: {exc}")


def render_development_page() -> None:
    st.title("Development")
    st.caption(f"Native Streamlit diagnostics for the production pipeline. Version {APP_VERSION}.")

    canonical_leagues = canonical_inputs()
    leagues_json = json.dumps(canonical_leagues, sort_keys=True)

    build_col, validate_col, promote_col = st.columns(3)
    if build_col.button("Build Candidate Model", type="primary", use_container_width=True):
        with st.spinner("Building candidate model..."):
            try:
                bundle = cached_build_candidate_model(leagues_json)
                save_candidate_model(CanonicalArtifactManager.candidate(), bundle)
                st.session_state["candidate_bundle"] = bundle
            except LSADPError as exc:
                st.error(str(exc))
            except Exception as exc:  # noqa: BLE001
                st.error(f"Unexpected candidate build failure: {exc}")
            else:
                st.success("Candidate model saved.")

    if validate_col.button("Refresh Candidate Diagnostics", use_container_width=True):
        with st.spinner("Refreshing candidate diagnostics..."):
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

    tabs = st.tabs(["Canonical ADP", "Historical Library", "Diagnostics", "Artifacts"])
    with tabs[0]:
        render_beatadp_section(leagues_json)
    with tabs[1]:
        render_seed_league_section()
        candidate_bundle = st.session_state.get("candidate_bundle")
        if candidate_bundle is not None:
            render_history_match_tester(candidate_bundle)
    with tabs[2]:
        candidate_bundle = st.session_state.get("candidate_bundle")
        if candidate_bundle is None:
            st.info("Build or refresh a candidate model to inspect diagnostics.")
        else:
            render_candidate_diagnostics(candidate_bundle)
    with tabs[3]:
        try:
            render_artifact_summary("Candidate Artifacts", CanonicalArtifactManager.candidate().load())
        except ConfigError as exc:
            st.info(str(exc))
        try:
            render_artifact_summary("Production Artifacts", CanonicalArtifactManager.production().load())
        except ConfigError as exc:
            st.info(str(exc))


def render_public_page_live() -> None:
    render_public_page()


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
