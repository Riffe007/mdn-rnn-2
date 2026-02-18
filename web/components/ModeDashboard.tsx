import type { DemoPayload } from '../lib/types';
import { CalibrationPlot } from './CalibrationPlot';
import { FanChart } from './FanChart';
import { ForecastBands } from './ForecastBands';
import { RegimeMarkers } from './RegimeMarkers';
import { TailRiskHeatmap } from './TailRiskHeatmap';

export function ModeDashboard({ payload }: { payload: DemoPayload }) {
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
        <ForecastBands />
        <FanChart />
        <TailRiskHeatmap />
        <RegimeMarkers />
        <CalibrationPlot />
      </section>
    </main>
  );
}
