"""Positional production-curve construction and fitting."""

from __future__ import annotations

from dataclasses import asdict
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from src.config import CORE_POSITIONS, CURVE_SELECTION_RELATIVE_IMPROVEMENT, DEFAULT_MIN_GAMES
from src.models import CurveFitError, CurveFitResult


def exponential_curve(rank: np.ndarray, a: float, c: float) -> np.ndarray:
    return a * np.exp(-c * (rank - 1))


def exponential_curve_with_floor(rank: np.ndarray, a: float, c: float, k: float) -> np.ndarray:
    return a * np.exp(-c * (rank - 1)) + k


def build_season_player_ppg(player_weeks: pd.DataFrame, min_games: int = DEFAULT_MIN_GAMES) -> pd.DataFrame:
    """Aggregate player-weeks into per-season PPG rows."""

    grouped = (
        player_weeks.groupby(["season", "player_id", "player_name", "position"], as_index=False)
        .agg(games=("week", "nunique"), ppg=("fantasy_points", "mean"))
    )
    filtered = grouped[grouped["games"] >= min_games].copy()
    filtered = filtered.sort_values(["season", "position", "ppg", "player_name"], ascending=[True, True, False, True])
    filtered["rank"] = filtered.groupby(["season", "position"]).cumcount() + 1
    return filtered


def aggregate_rank_curves(
    season_player_ppg: pd.DataFrame,
    aggregation: str = "median",
    recency_weights: dict[int, float] | None = None,
) -> pd.DataFrame:
    """Aggregate seasonal rank curves into one expected curve per position."""

    recency_weights = recency_weights or {}
    working = season_player_ppg.copy()
    working["season_weight"] = working["season"].map(recency_weights).fillna(1.0)

    if aggregation == "weighted_mean":
        rows: list[dict[str, float | int | str]] = []
        for (position, rank), group in working.groupby(["position", "rank"]):
            rows.append(
                {
                    "position": position,
                    "rank": rank,
                    "expected_ppg": float(np.average(group["ppg"], weights=group["season_weight"])),
                    "season_count": int(group["season"].nunique()),
                    "player_count": int(group["player_id"].nunique()),
                }
            )
        aggregated = pd.DataFrame(rows)
    else:
        aggregated = (
            working.groupby(["position", "rank"], as_index=False)
            .agg(
                expected_ppg=("ppg", "median"),
                season_count=("season", "nunique"),
                player_count=("player_id", "nunique"),
            )
        )

    return aggregated.sort_values(["position", "rank"]).reset_index(drop=True)


def _fit_single_model(
    rank_points: np.ndarray,
    ppg_points: np.ndarray,
    model_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    if model_name == "exp":
        func: Callable[..., np.ndarray] = exponential_curve
        bounds = ([0.0, 0.0], [100.0, 5.0])
        p0 = [float(np.nanmax(ppg_points)), 0.08]
    else:
        func = exponential_curve_with_floor
        bounds = ([0.0, 0.0, 0.0], [100.0, 5.0, 20.0])
        p0 = [float(np.nanmax(ppg_points)), 0.08, float(np.nanmin(ppg_points) * 0.5)]

    try:
        params, covariance = curve_fit(
            func,
            rank_points,
            ppg_points,
            p0=p0,
            bounds=bounds,
            maxfev=10000,
        )
    except Exception as exc:  # noqa: BLE001
        raise CurveFitError(f"Curve fit failed for model {model_name}") from exc
    return params, covariance


def _prediction_metrics(actual: np.ndarray, predicted: np.ndarray, parameter_count: int) -> dict[str, float]:
    residuals = actual - predicted
    sse = float(np.sum(np.square(residuals)))
    rmse = float(np.sqrt(np.mean(np.square(residuals))))
    mae = float(np.mean(np.abs(residuals)))
    total = float(np.sum(np.square(actual - np.mean(actual))))
    r2 = 1.0 - sse / total if total else 0.0
    n = max(len(actual), 1)
    aic = float(n * np.log(max(sse / n, 1e-12)) + 2 * parameter_count)
    bic = float(n * np.log(max(sse / n, 1e-12)) + parameter_count * np.log(n))
    return {"rmse": rmse, "mae": mae, "r2": r2, "aic": aic, "bic": bic}


def _cross_validated_rmse(season_player_ppg: pd.DataFrame, position: str, model_name: str) -> float:
    position_frame = season_player_ppg[season_player_ppg["position"] == position]
    seasons = sorted(position_frame["season"].unique())
    if len(seasons) < 2:
        return float("inf")

    errors: list[float] = []
    for holdout in seasons:
        train = position_frame[position_frame["season"] != holdout]
        test = position_frame[position_frame["season"] == holdout]
        if train.empty or test.empty:
            continue
        aggregated = aggregate_rank_curves(train)
        fit = fit_curve_for_position(aggregated[aggregated["position"] == position], position, model_name=model_name, historical_window="cv")
        predictions = np.array([fit.evaluate(rank) for rank in test["rank"].to_numpy(dtype=float)])
        errors.append(float(np.sqrt(np.mean(np.square(test["ppg"].to_numpy(dtype=float) - predictions)))))
    if not errors:
        return float("inf")
    return float(np.mean(errors))


def fit_curve_for_position(
    empirical_curve: pd.DataFrame,
    position: str,
    model_name: str,
    historical_window: str,
) -> CurveFitResult:
    """Fit one candidate curve for a single position."""

    position_curve = empirical_curve[empirical_curve["position"] == position].sort_values("rank")
    if position_curve.empty:
        raise CurveFitError(f"No empirical curve data available for {position}")

    rank_points = position_curve["rank"].to_numpy(dtype=float)
    ppg_points = position_curve["expected_ppg"].to_numpy(dtype=float)
    params, _ = _fit_single_model(rank_points=rank_points, ppg_points=ppg_points, model_name=model_name)

    if model_name == "exp":
        predicted = exponential_curve(rank_points, params[0], params[1])
        a, c = params
        k = 0.0
    else:
        predicted = exponential_curve_with_floor(rank_points, params[0], params[1], params[2])
        a, c, k = params

    metrics = _prediction_metrics(ppg_points, predicted, parameter_count=len(params))
    return CurveFitResult(
        position=position,
        model_name=model_name,
        a=float(a),
        c=float(c),
        k=float(k),
        rmse=metrics["rmse"],
        mae=metrics["mae"],
        r2=metrics["r2"],
        aic=metrics["aic"],
        bic=metrics["bic"],
        cv_rmse=float("inf"),
        historical_window=historical_window,
    )


def select_curve_models(
    season_player_ppg: pd.DataFrame,
    aggregated_curve: pd.DataFrame,
    historical_window: str,
    minimum_relative_improvement: float = CURVE_SELECTION_RELATIVE_IMPROVEMENT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit and select the preferred curve model for each position."""

    fit_records: list[dict[str, float | str]] = []
    selected_records: list[dict[str, float | str]] = []

    for position in CORE_POSITIONS:
        position_empirical = aggregated_curve[aggregated_curve["position"] == position]
        if position_empirical.empty:
            continue

        exp_fit = fit_curve_for_position(position_empirical, position=position, model_name="exp", historical_window=historical_window)
        exp_fit.cv_rmse = _cross_validated_rmse(season_player_ppg, position=position, model_name="exp")

        floor_fit = fit_curve_for_position(position_empirical, position=position, model_name="exp_k", historical_window=historical_window)
        floor_fit.cv_rmse = _cross_validated_rmse(season_player_ppg, position=position, model_name="exp_k")

        fit_records.extend([asdict(exp_fit), asdict(floor_fit)])

        selected = exp_fit
        if np.isfinite(exp_fit.cv_rmse) and np.isfinite(floor_fit.cv_rmse):
            improvement = (exp_fit.cv_rmse - floor_fit.cv_rmse) / exp_fit.cv_rmse if exp_fit.cv_rmse else 0.0
            if improvement >= minimum_relative_improvement:
                selected = floor_fit
        selected_records.append(asdict(selected))

    return pd.DataFrame(fit_records), pd.DataFrame(selected_records)


def evaluate_curve_series(curve_fit: CurveFitResult, max_rank: int) -> pd.DataFrame:
    """Evaluate a fitted curve across positional ranks."""

    ranks = np.arange(1, max_rank + 1, dtype=float)
    values = [curve_fit.evaluate(rank) for rank in ranks]
    return pd.DataFrame({"position": curve_fit.position, "rank": ranks.astype(int), "expected_ppg": values})
