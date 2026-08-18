"""Market calibration between current ADP and modeled metrics."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from src.models import CalibrationResult
from src.utils import adp_utility, adp_weights, correlation_safe, safe_divide


def fit_weighted_linear_regression(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float]:
    """Closed-form weighted linear regression with intercept."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    weights = np.asarray(weights, dtype=float)

    weight_sum = weights.sum()
    mean_x = safe_divide(float(np.sum(weights * x)), float(weight_sum))
    mean_y = safe_divide(float(np.sum(weights * y)), float(weight_sum))

    numerator = float(np.sum(weights * (x - mean_x) * (y - mean_y)))
    denominator = float(np.sum(weights * np.square(x - mean_x)))
    slope = safe_divide(numerator, denominator)
    intercept = mean_y - slope * mean_x
    return intercept, slope


def calibrate_market_values(
    adp_frame: pd.DataFrame,
    metric_column: str = "baseline_vorp",
    metric_name: str = "vorp",
    utility_transform: str = "neg_log",
    weight_power: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit per-position market exchange rates from the current ADP market."""

    working = adp_frame.copy()
    working["utility"] = adp_utility(working["adp"], transform=utility_transform)
    working["regression_weight"] = adp_weights(working["adp"], power=weight_power)

    calibration_rows: list[dict[str, float | str]] = []
    enriched_frames: list[pd.DataFrame] = []

    for position, position_frame in working.groupby("position"):
        x = position_frame[metric_column].fillna(0.0).to_numpy(dtype=float)
        y = position_frame["utility"].to_numpy(dtype=float)
        weights = position_frame["regression_weight"].to_numpy(dtype=float)
        intercept, slope = fit_weighted_linear_regression(x=x, y=y, weights=weights)
        predicted = intercept + slope * x
        residual = y - predicted
        weighted_rmse = float(np.sqrt(np.average(np.square(residual), weights=weights)))
        correlation = correlation_safe(pd.Series(y), pd.Series(predicted), method="pearson")
        position_r2 = float(correlation**2) if pd.notna(correlation) else 0.0

        enriched = position_frame.copy()
        enriched["market_metric_name"] = metric_name
        enriched["market_intercept"] = intercept
        enriched["market_coefficient"] = slope
        enriched["market_predicted_utility"] = predicted
        enriched["market_residual"] = residual
        enriched["utility_transform"] = utility_transform
        enriched["weight_power"] = weight_power
        enriched_frames.append(enriched)

        calibration_rows.append(
            asdict(
                CalibrationResult(
                position=str(position),
                metric_name=metric_name,
                intercept=float(intercept),
                coefficient=float(slope),
                r2=position_r2,
                weighted_rmse=weighted_rmse,
                )
            )
        )
        calibration_rows[-1]["utility_transform"] = utility_transform
        calibration_rows[-1]["weight_power"] = weight_power

    calibration_frame = pd.DataFrame(calibration_rows).sort_values("position").reset_index(drop=True)
    enriched_frame = pd.concat(enriched_frames, ignore_index=True)
    return calibration_frame, enriched_frame
