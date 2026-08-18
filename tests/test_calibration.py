from __future__ import annotations

from src.calibration import calibrate_market_values


def test_calibration_returns_positive_exchange_rates(adp_frame) -> None:
    frame = adp_frame.copy()
    frame["baseline_vorp"] = frame.groupby("position").cumcount(ascending=True).rsub(6).astype(float)
    calibration, enriched = calibrate_market_values(frame, metric_column="baseline_vorp", metric_name="vorp")
    assert (calibration["coefficient"] > 0).all()
    assert "market_residual" in enriched.columns

