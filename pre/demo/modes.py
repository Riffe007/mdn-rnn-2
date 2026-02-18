from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DemoMode:
    slug: str
    title: str
    dataset: str
    model: str
    horizon: int
    context_length: int
    description: str


DEMO_MODES: dict[str, DemoMode] = {
    "telemetry": DemoMode(
        slug="telemetry",
        title="Operational Risk Mode",
        dataset="telemetry",
        model="lgbm_quantile",
        horizon=24,
        context_length=120,
        description=(
            "Failure probability, resource exhaustion risk, and "
            "regime-shift aware telemetry forecasting."
        ),
    ),
    "demand": DemoMode(
        slug="demand",
        title="Forecasting Lab Mode",
        dataset="nyc_taxi",
        model="lstm_gaussian",
        horizon=24,
        context_length=168,
        description=(
            "Demand uncertainty bands, peak event probability, and "
            "seasonal regime analysis."
        ),
    ),
    "project-risk": DemoMode(
        slug="project-risk",
        title="Project Risk Mode",
        dataset="project_sim",
        model="lgbm_quantile",
        horizon=12,
        context_length=72,
        description=(
            "Delay probability, cost overrun pressure, and "
            "critical-path volatility simulation."
        ),
    ),
    "event-sandbox": DemoMode(
        slug="event-sandbox",
        title="Event Forecasting Sandbox",
        dataset="energy_load",
        model="lstm_gaussian",
        horizon=18,
        context_length=120,
        description=(
            "Probability drift, confidence shift, and "
            "uncertainty widening in event-like trajectories."
        ),
    ),
    "finance": DemoMode(
        slug="finance",
        title="Financial Regime Mode",
        dataset="energy_load",
        model="lgbm_quantile",
        horizon=30,
        context_length=180,
        description=(
            "Volatility band expansion and regime-change likelihood "
            "without alpha claims."
        ),
    ),
}
