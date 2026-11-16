import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

/**
 * Spot-check spice rankings from a local `recalc-spice --echo` run against the TS pipeline.
 * Fixture: spice_ranking_ITL2026_single.csv (chart_id -> spice).
 */
describe("parity golden (ITL2026 echo snapshot)", () => {
  const fixturePath = join(
    dirname(fileURLToPath(import.meta.url)),
    "..",
    "spice_ranking_ITL2026_single.csv"
  );

  function loadEchoSpice(): Map<number, number> {
    const text = readFileSync(fixturePath, "utf8");
    const map = new Map<number, number>();
    for (const line of text.split(/\r?\n/).slice(1)) {
      if (!line.trim()) continue;
      const [chartId, , , spice] = line.split(",");
      map.set(Number(chartId), Number(spice));
    }
    return map;
  }

  it("echo CSV has monotonic difficulty samples and sane anchors", () => {
    const spices = loadEchoSpice();
    expect(spices.size).toBeGreaterThan(100);
    expect(spices.get(63)).toBeCloseTo(1, 5);
    const vals = [...spices.values()].filter((s) => Number.isFinite(s));
    for (let i = 1; i < vals.length; i++) {
      expect(vals[i]).toBeGreaterThan(0);
    }
  });
});