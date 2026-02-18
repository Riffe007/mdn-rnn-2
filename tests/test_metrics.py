from __future__ import annotations

import numpy as np

from pre.eval.calibration import quantile_reliability_bins
from pre.eval.metrics import crps_gaussian, gaussian_nll, interval_coverage, mae, rmse
from pre.eval.reports import build_report, to_markdown


def test_point_metrics_deterministic() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.5, 2.5, 2.0])
    std = np.array([1.0, 1.0, 1.0])

    assert mae(y_true, y_pred) == 2.0 / 3.0
    assert np.isclose(rmse(y_true, y_pred), np.sqrt(0.5))
    assert np.isfinite(gaussian_nll(y_true, y_pred, std))
    assert np.isfinite(crps_gaussian(y_true, y_pred, std))


def test_gaussian_coverage_for_10_90_interval_is_near_80_percent() -> None:
    rng = np.random.default_rng(7)
    y_true = rng.normal(loc=0.0, scale=1.0, size=20_000)
    lower = np.full_like(y_true, -1.28155)
    upper = np.full_like(y_true, 1.28155)

    coverage = interval_coverage(y_true, lower, upper)
    assert 0.79 <= coverage <= 0.81


def test_reliability_bins_and_report_generation() -> None:
    y_true = np.array([0.0, 1.0, 2.0, 3.0])
    quantiles = {
        0.1: np.array([-1.0, 0.0, 1.0, 2.0]),
        0.5: np.array([0.0, 1.0, 2.0, 3.0]),
        0.9: np.array([1.0, 2.0, 3.0, 4.0]),
    }

    bins = quantile_reliability_bins(y_true=y_true, quantile_predictions=quantiles)
    assert len(bins) == 3
    assert bins[1].expected == 0.5

    report = build_report(
        y_true=y_true,
        y_pred_mean=quantiles[0.5],
        y_pred_std=np.ones_like(y_true),
        quantiles=quantiles,
    )
    markdown = to_markdown(report)

    assert "# Evaluation Report" in markdown
    assert "| expected | observed | count |" in markdown
