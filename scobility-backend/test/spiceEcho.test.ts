import { describe, expect, it } from "vitest";
import { Song } from "../src/tournament/models.js";

describe("spice log line", () => {
  it("matches Python view_spice_ranking layout", () => {
    const s = new Song({
      s_id: 6,
      title: "Disco Pop",
      subtitle: "",
      style: "Single",
      slot: "Hard",
      meter: 7,
      value_passing: 0,
      value_scoring: 100,
      spice: 1,
    });
    const line = s.toSpiceLogLine();
    expect(line).toMatch(/^0\.000🌶 /);
    expect(line).toContain("#6 Disco Pop");
    expect(line.length).toBeGreaterThan(60);
  });
});