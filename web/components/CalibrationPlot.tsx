import { toPath } from './chartUtils';

export function CalibrationPlot({
  bins,
}: {
  bins: Array<{ expected: number; observed: number; count: number }>;
}) {
  const width = 320;
  const height = 220;
  const expected = bins.map((item) => item.expected);
  const observed = bins.map((item) => item.observed);

  return (
    <section className="panel">
      <h3>Calibration Plot</h3>
      <p>Observed vs expected coverage reliability bins.</p>
      <svg className="chart-svg" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Calibration plot">
        <line x1="0" y1={height} x2={width} y2="0" stroke="rgba(15,23,32,0.25)" strokeDasharray="6 6" />
        <path d={toPath(observed, width, height, 0, 1)} fill="none" stroke="#0095a8" strokeWidth="2.3" />
        {expected.map((value, idx) => {
          const x = value * width;
          const y = height - observed[idx] * height;
          return <circle key={idx} cx={x} cy={y} r="4" fill="#e66a1f" />;
        })}
      </svg>
    </section>
  );
}
