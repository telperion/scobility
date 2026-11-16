import { describe, expect, it } from "vitest";
import { lstsq } from "../src/math/lstsq.js";

/**
 * Fixture: relationship fit where y ≈ 1.05 * x on filtered points.
 * Values chosen to mirror Relationship.calc_relationship weighting setup.
 */
describe("lstsq", () => {
  it("matches simple proportional fit within tolerance", () => {
    const x = [0.05, 0.08, 0.12, 0.15, 0.18, 0.22, 0.25, 0.28];
    const slope = 1.047;
    const y = x.map((v) => v * slope);
    const w = x.map((v) => 1 - 1 / Math.pow(v + 0.5, 2));
    const a = new Float64Array(x);
    const b = new Float64Array(y);
    const ww = new Float64Array(w);
    const coefs = lstsq(a, b, ww);
    expect(coefs[0]).toBeCloseTo(1.047, 12);
  });
});