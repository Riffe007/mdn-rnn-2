import type { DemoPayload } from '../lib/types';
import { CalibrationPlot } from './CalibrationPlot';
import { FanChart } from './FanChart';
import { ForecastBands } from './ForecastBands';
import { RegimeMarkers } from './RegimeMarkers';
import { TailRiskHeatmap } from './TailRiskHeatmap';

export function ModeDashboard({ payload }: { payload: DemoPayload }) {
  const { visuals } = payload;

  return (
    <main>
      <section className="panel grid">
        <h1>{payload.title}</h1>
        <p>{payload.description}</p>
        <div className="kpi">
          <span>Dataset: {payload.dataset}</span>
          <span>Model: {payload.model}</span>
          <span>Tail Risk Score: {payload.tail_risk_score.toFixed(4)}</span>
          <span>Regime Shift Score: {payload.regime_shift_score.toFixed(4)}</span>
          <span>Coverage: {payload.metrics.coverage.toFixed(4)}</span>
          <span>CRPS: {payload.metrics.crps.toFixed(4)}</span>
        </div>
      </section>
      <section className="grid" style={{ marginTop: '1rem' }}>
        <ForecastBands
          p10={visuals.bands.p10}
          p50={visuals.bands.p50}
          p90={visuals.bands.p90}
        />
        <FanChart
          p05={visuals.fan.p05}
          p25={visuals.fan.p25}
          p50={visuals.fan.p50}
          p75={visuals.fan.p75}
          p95={visuals.fan.p95}
        />
        <TailRiskHeatmap matrix={visuals.tail_risk_heatmap} />
        <RegimeMarkers p50={visuals.bands.p50} markers={visuals.regime_markers} />
        <CalibrationPlot bins={visuals.calibration_bins} />
      </section>
    </main>
  );
}
