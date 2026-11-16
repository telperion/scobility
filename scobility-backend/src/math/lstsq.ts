/**
 * Weighted least-squares slope through origin: b ≈ k * a
 * (matches numpy.linalg.lstsq on design matrix a * w, target b * w)
 */
export function lstsq(a: Float64Array, b: Float64Array, w: Float64Array): number[] {
  let num = 0;
  let den = 0;
  for (let i = 0; i < a.length; i++) {
    num += w[i] * a[i] * b[i];
    den += w[i] * a[i] * a[i];
  }
  if (den === 0) {
    return [0];
  }
  return [num / den];
}