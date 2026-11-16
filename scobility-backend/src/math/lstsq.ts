import { Matrix, solve } from "ml-matrix";

/**
 * Port of numpy.linalg.lstsq(A, b, rcond=None)[0] for the 1-row design
 * matrices used in Relationship.calc_relationship.
 */
export function lstsq(a: Float64Array, b: Float64Array, w: Float64Array): number[] {
  const n = a.length;
  const aw = new Float64Array(n);
  const bw = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    aw[i] = a[i] * w[i];
    bw[i] = b[i] * w[i];
  }
  const A = Matrix.from1DArray(n, 1, Array.from(aw));
  const B = Matrix.columnVector(Array.from(bw));
  const coef = solve(A, B);
  return coef.to1DArray();
}