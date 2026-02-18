from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from pre.eval.calibration import picp, quantile_reliability_bins
from pre.eval.metrics import crps_gaussian, gaussian_nll, mae, rmse


def build_report(
    y_true: np.ndarray,
    y_pred_mean: np.ndarray,
    y_pred_std: np.ndarray,
    quantiles: dict[float, np.ndarray],
) -> dict[str, object]:
    lower = quantiles[min(quantiles)]
    upper = quantiles[max(quantiles)]
    reliability = quantile_reliability_bins(y_true=y_true, quantile_predictions=quantiles)

    return {
        "mae": mae(y_true, y_pred_mean),
        "rmse": rmse(y_true, y_pred_mean),
        "nll": gaussian_nll(y_true, y_pred_mean, y_pred_std),
        "crps": crps_gaussian(y_true, y_pred_mean, y_pred_std),
        "coverage": picp(y_true, lower, upper),
        "reliability_bins": [asdict(bin_value) for bin_value in reliability],
    }


def to_markdown(report: dict[str, Any]) -> str:
    lines = ["# Evaluation Report", "", "## Summary", ""]
    for key in ["mae", "rmse", "nll", "crps", "coverage"]:
        value = report[key]
        lines.append(f"- {key}: {value:.6f}")

    lines.extend(
        ["", "## Reliability Bins", "", "| expected | observed | count |", "|---|---|---|"]
    )
    for bin_value in report["reliability_bins"]:
        row = bin_value
        lines.append(f"| {row['expected']:.2f} | {row['observed']:.2f} | {row['count']} |")
    return "\n".join(lines)
