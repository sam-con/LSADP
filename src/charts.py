"""Plotly chart builders for the Streamlit app."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def build_adp_movement_chart(results: pd.DataFrame, position_filter: str = "ALL") -> go.Figure:
    frame = results.copy()
    if position_filter != "ALL":
        frame = frame[frame["position"] == position_filter]
    frame = frame.reindex(frame["adp_change"].abs().sort_values(ascending=False).index).head(20)

    figure = go.Figure()
    for _, row in frame.iterrows():
        figure.add_trace(
            go.Scatter(
                x=[row["adp"], row["league_adjusted_adp"]],
                y=[row["player_name"], row["player_name"]],
                mode="lines+markers",
                line={"width": 3},
                marker={"size": 10},
                name=row["player_name"],
                showlegend=False,
                hovertemplate=(
                    f"{row['player_name']}<br>Current ADP: {row['adp']:.1f}<br>"
                    f"League ADP: {row['league_adjusted_adp']:.1f}<extra></extra>"
                ),
            )
        )
    figure.update_layout(
        title="Largest ADP Movers",
        xaxis_title="ADP",
        yaxis_title="Player",
        template="plotly_white",
        height=700,
    )
    figure.update_xaxes(autorange="reversed")
    return figure


def build_curve_chart(
    baseline_curve: pd.DataFrame,
    target_curve: pd.DataFrame,
    empirical_baseline: pd.DataFrame,
    empirical_target: pd.DataFrame,
    replacement_baseline_rank: int,
    replacement_target_rank: int,
    position: str,
) -> go.Figure:
    figure = go.Figure()
    for frame, name, dash in [
        (baseline_curve, "Baseline Curve", "solid"),
        (target_curve, "League Curve", "dash"),
    ]:
        position_frame = frame[frame["position"] == position]
        figure.add_trace(
            go.Scatter(
                x=position_frame["rank"],
                y=position_frame["expected_ppg"],
                mode="lines",
                name=name,
                line={"width": 3, "dash": dash},
            )
        )

    for frame, name, color in [
        (empirical_baseline, "Baseline Empirical", "#0f766e"),
        (empirical_target, "League Empirical", "#b45309"),
    ]:
        position_frame = frame[frame["position"] == position]
        figure.add_trace(
            go.Scatter(
                x=position_frame["rank"],
                y=position_frame["expected_ppg"],
                mode="markers",
                marker={"size": 7, "color": color},
                name=name,
            )
        )

    figure.add_vline(x=replacement_baseline_rank, line_dash="dot", line_color="#0f766e")
    figure.add_vline(x=replacement_target_rank, line_dash="dot", line_color="#b45309")
    figure.update_layout(
        title=f"{position} Production Curve",
        xaxis_title="Positional Rank",
        yaxis_title="Expected PPG",
        template="plotly_white",
    )
    return figure


def build_current_vs_adjusted_scatter(results: pd.DataFrame) -> go.Figure:
    figure = px.scatter(
        results,
        x="adp",
        y="league_adjusted_adp",
        color="position",
        hover_name="player_name",
        hover_data={"short_explanation": True, "adp_change": ":.2f"},
        title="Current ADP vs League-Adjusted ADP",
        template="plotly_white",
    )
    max_value = float(max(results["adp"].max(), results["league_adjusted_adp"].max()))
    figure.add_shape(type="line", x0=0, x1=max_value, y0=0, y1=max_value, line={"dash": "dash"})
    figure.update_xaxes(autorange="reversed")
    figure.update_yaxes(autorange="reversed")
    return figure


def build_positional_impact_chart(summary: pd.DataFrame) -> go.Figure:
    figure = px.bar(
        summary,
        x="position",
        y="impact_pct",
        color="position",
        text=summary["impact_pct"].map(lambda value: f"{value:.1f}%"),
        title="Positional Impact Summary",
        template="plotly_white",
    )
    figure.update_layout(showlegend=False)
    return figure


def build_validation_scatter(predicted: pd.DataFrame, actual: pd.DataFrame, title: str) -> go.Figure:
    actual_frame = actual[["player_name", "position", "adp"]].rename(columns={"adp": "actual_adp"})
    merged = predicted.merge(actual_frame, on=["player_name", "position"], how="inner")
    figure = px.scatter(
        merged,
        x="league_adjusted_adp",
        y="actual_adp",
        color="position",
        hover_name="player_name",
        title=title,
        template="plotly_white",
    )
    max_value = float(max(merged["league_adjusted_adp"].max(), merged["actual_adp"].max()))
    figure.add_shape(type="line", x0=0, x1=max_value, y0=0, y1=max_value, line={"dash": "dash"})
    figure.update_xaxes(autorange="reversed", title="Predicted ADP")
    figure.update_yaxes(autorange="reversed", title="Actual ADP")
    return figure


def build_validation_heatmap(validation_frame: pd.DataFrame, value_column: str = "weighted_mae") -> go.Figure:
    figure = px.density_heatmap(
        validation_frame,
        x="source_label",
        y="target_label",
        z=value_column,
        histfunc="avg",
        color_continuous_scale="YlGn",
        title=f"Validation Matrix: {value_column.replace('_', ' ').title()}",
    )
    figure.update_layout(template="plotly_white")
    return figure


def build_error_by_bucket_chart(bucket_frame: pd.DataFrame) -> go.Figure:
    figure = px.bar(
        bucket_frame,
        x="bucket",
        y="mae",
        text="mae",
        title="Error by ADP Range",
        template="plotly_white",
    )
    figure.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    return figure


def build_environment_curve_chart(curves: pd.DataFrame, position: str) -> go.Figure:
    frame = curves[(curves["position"] == position) & (curves["dataset"] == "fitted")].copy()
    figure = px.line(
        frame,
        x="rank",
        y="expected_ppg",
        color="environment_key",
        title=f"{position} Canonical Production Curves",
        template="plotly_white",
    )
    figure.update_layout(xaxis_title="Positional Rank", yaxis_title="Expected PPG")
    return figure
