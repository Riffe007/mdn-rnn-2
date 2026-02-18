# Riffe Labs Probabilistic Risk Engine (PRE)

Contract-first foundation for a probabilistic forecasting and risk platform.

## Current Scope (PR1 + PR2 + PR3 + PR4 + PR5)
- Python package scaffold under `pre/`
- Unified probabilistic output contract (`PredictiveDistribution`)
- Evaluation harness (`MAE`, `RMSE`, Gaussian `NLL`, Gaussian `CRPS`, interval coverage)
- Quantile reliability bins + markdown report generation
- NYC Taxi dataset adapter with windowing, temporal train/val/test split, and rolling backtest folds
- `pre-train` dataset -> split -> shape-check execution path for `--model dummy`
- Fitted Gaussian forecaster backend
  (`lstm_gaussian`, numpy offline-safe placeholder)
- Quantile baseline backend (`lgbm_quantile`) integrated into training/evaluation flow
- Artifact bundle output per run:
  `config.json`, `scaler.npz`, `model.npz`, `report.json`, `report.md`
- Evaluation report includes `MAE`, `RMSE`, `NLL`, `CRPS`, and coverage
- Benchmark runner (`pre-benchmark`) with markdown/json leaderboard output
- Multi-demo runner (`pre-demo`) for:
  - Operational Risk Mode (`telemetry`)
  - Forecasting Lab Mode (`demand`)
  - Project Risk Mode (`project-risk`)
  - Event Forecasting Sandbox (`event-sandbox`)
  - Financial Regime Mode (`finance`)
- CLI commands (`pre-train`, `pre-eval`, `pre-predict`)
- `web/` Next.js multi-mode lab scaffold with dedicated pages for all demo modes
- CI for lint, type-check, and tests

## Quick Start
```bash
pip install -e .[dev]
pytest
ruff check .
mypy pre
```

## CLI Stubs
```bash
pre-train --dataset nyc_taxi --model dummy
pre-eval --run-id latest
pre-predict --run-id latest
```

`pre-train` also supports:
```bash
pre-train --dataset nyc_taxi --model dummy --horizon 12 --context-length 72 --stride 1
pre-train --dataset nyc_taxi --model lstm_gaussian \
  --horizon 12 --context-length 72 --artifact-root artifacts
pre-train --dataset telemetry --model lgbm_quantile \
  --horizon 24 --context-length 120 --artifact-root artifacts
pre-benchmark --dataset nyc_taxi --models lstm_gaussian,lgbm_quantile \
  --horizon 24 --context-length 168
pre-demo --list-modes
pre-demo --mode all
pre-api
```

## Web Lab
```bash
cd web
npm install
npm run dev
```

Set API source for live demo payloads:
```bash
export NEXT_PUBLIC_PRE_API_BASE=http://localhost:8000
```

## Next Milestone
FastAPI transport layer and Next.js UI consuming demo/benchmark payloads with production charts.
