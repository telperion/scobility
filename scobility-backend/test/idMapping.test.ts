import { describe, expect, it } from "vitest";
import {
  globalChartId,
  performanceId,
  scoreId,
} from "../src/catalog/idMapping.js";

describe("idMapping", () => {
  it("matches notebook catalog 6 pattern", () => {
    expect(globalChartId(6, 42)).toBe(6042);
    expect(performanceId(6, 7)).toBe(60007);
    expect(scoreId(6, 0)).toBe(6000000);
  });
});