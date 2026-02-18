import { toPath } from './chartUtils';

export function RegimeMarkers({
  p50,
  markers,
}: {
  p50: number[];
  markers: number[];
}) {
  const width = 760;
  const height = 170;
  const minValue = Math.min(...p50);
  const maxValue = Math.max(...p50);

  return (
    <section className="panel">
      <h3>Regime Shift Markers</h3>
      <p>Detected change points and transition confidence.</p>
      <svg
        className="chart-svg"
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label="Regime markers"
      >
        <path
          d={toPath(p50, width, height, minValue, maxValue)}
          fill="none"
          stroke="#0f1720"
          strokeWidth="2"
        />
        {markers.map((idx) => {
          const x = (idx / Math.max(p50.length - 1, 1)) * width;
          return (
            <line
              key={idx}
              x1={x}
              y1="0"
              x2={x}
              y2={height}
              stroke="#e66a1f"
              strokeWidth="2"
              strokeDasharray="5 4"
            />
          );
        })}
      </svg>
    </section>
  );
}
