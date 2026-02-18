export function toPath(
  values: number[],
  width: number,
  height: number,
  minValue: number,
  maxValue: number,
): string {
  if (values.length === 0) {
    return '';
  }
  const span = Math.max(maxValue - minValue, 1e-6);
  return values
    .map((value, idx) => {
      const x = (idx / Math.max(values.length - 1, 1)) * width;
      const y = height - ((value - minValue) / span) * height;
      return `${idx === 0 ? 'M' : 'L'}${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(' ');
}

export function toBandPath(
  lower: number[],
  upper: number[],
  width: number,
  height: number,
  minValue: number,
  maxValue: number,
): string {
  if (lower.length === 0 || upper.length === 0) {
    return '';
  }
  const span = Math.max(maxValue - minValue, 1e-6);
  const top = upper.map((value, idx) => {
    const x = (idx / Math.max(upper.length - 1, 1)) * width;
    const y = height - ((value - minValue) / span) * height;
    return `${idx === 0 ? 'M' : 'L'}${x.toFixed(2)} ${y.toFixed(2)}`;
  });
  const bottom = lower
    .map((value, idx) => {
      const rev = lower.length - 1 - idx;
      const x = (rev / Math.max(lower.length - 1, 1)) * width;
      const y = height - ((value - minValue) / span) * height;
      return `L${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(' ');
  return `${top.join(' ')} ${bottom} Z`;
}
