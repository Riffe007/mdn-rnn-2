import { toBandPath, toPath } from './chartUtils';

export function ForecastBands({
  p10,
  p50,
  p90,
}: {
  p10: number[];
  p50: number[];
  p90: number[];
}) {
  const width = 760;
  const height = 170;
  const minValue = Math.min(...p10, ...p50, ...p90);
  const maxValue = Math.max(...p10, ...p50, ...p90);

  return (
    <section className="panel">
      <h3>Forecast Bands</h3>
      <p>P10 / P50 / P90 distribution envelope from PRE predictions.</p>
      <svg
        className="chart-svg"
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label="Forecast bands"
      >
        <path d={toBandPath(p10, p90, width, height, minValue, maxValue)} fill="rgba(0,149,168,0.28)" />
        <path
          d={toPath(p50, width, height, minValue, maxValue)}
          fill="none"
          stroke="#0a5f6b"
          strokeWidth="2.5"
        />
      </svg>
    </section>
  );
}
