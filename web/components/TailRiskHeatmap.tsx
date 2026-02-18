export function TailRiskHeatmap({ matrix }: { matrix: number[][] }) {
  const rows = matrix.length;
  const cols = rows > 0 ? matrix[0].length : 0;

  return (
    <section className="panel">
      <h3>Tail Risk Heatmap</h3>
      <p>Concentration of low-probability high-impact outcomes.</p>
      <div className="heatmap" role="img" aria-label="Tail risk heatmap">
        {matrix.map((row, r) =>
          row.map((value, c) => (
            <span
              key={`${r}-${c}`}
              style={{
                background: `rgba(230,106,31,${Math.max(0.08, Math.min(0.95, value))})`,
                gridRow: r + 1,
                gridColumn: c + 1,
              }}
              aria-label={`risk-${r}-${c}`}
            />
          )),
        )}
      </div>
      <p className="kpi">
        Grid: {rows} x {cols}
      </p>
    </section>
  );
}
