import { toBandPath, toPath } from './chartUtils';

export function FanChart({
  p05,
  p25,
  p50,
  p75,
  p95,
}: {
  p05: number[];
  p25: number[];
  p50: number[];
  p75: number[];
  p95: number[];
}) {
  const width = 760;
  const height = 170;
  const minValue = Math.min(...p05, ...p25, ...p50, ...p75, ...p95);
  const maxValue = Math.max(...p05, ...p25, ...p50, ...p75, ...p95);

  return (
    <section className="panel">
      <h3>Monte Carlo Fan Chart</h3>
      <p>Scenario spread for forward uncertainty rollouts.</p>
      <svg className="chart-svg" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Fan chart">
        <path d={toBandPath(p05, p95, width, height, minValue, maxValue)} fill="rgba(230,106,31,0.20)" />
        <path d={toBandPath(p25, p75, width, height, minValue, maxValue)} fill="rgba(230,106,31,0.35)" />
        <path
          d={toPath(p50, width, height, minValue, maxValue)}
          fill="none"
          stroke="#7a3409"
          strokeWidth="2.2"
        />
      </svg>
    </section>
  );
}
