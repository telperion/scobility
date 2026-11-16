import { describe, expect, it } from "vitest";
import { Song, Tournament } from "../src/tournament/models.js";
import { catalogConfig } from "../src/catalog/catalogConfig.js";
import {
  orderRefineBySpice,
  orderRefineMonotonic,
  orderSongsInitial,
} from "../src/tournament/ordering.js";
import { calcRelationships, setupRelationships } from "../src/tournament/tournament.js";

/**
 * Minimal synthetic catalog: three charts with staged scores so relationships resolve.
 */
function buildMiniTournament(): Tournament {
  const cfg = catalogConfig("ITL2026");
  const t = new Tournament(cfg);

  const mkSong = (id: number, style = "Single") => {
    const s = new Song({ s_id: id, hash: `hash${id}`, style, meter: 10 + id });
    t.songs.set(id, s);
    return s;
  };

  const s1 = mkSong(1);
  const s2 = mkSong(2);
  const s3 = mkSong(3);

  const common = [101, 102, 103, 104, 105, 106, 107];
  for (const e of common) {
    s1.scores.set(e, { s_id: 1, e_id: e, plays: 1, last_played: null, clear: 1, value: 0.2 });
    s2.scores.set(e, { s_id: 2, e_id: e, plays: 1, last_played: null, clear: 1, value: 0.15 });
    s3.scores.set(e, { s_id: 3, e_id: e, plays: 1, last_played: null, clear: 1, value: 0.1 });
  }

  setupRelationships(t);
  calcRelationships(t, false);
  orderSongsInitial(t, "Single", false);
  orderRefineMonotonic(t, "Single", false);
  orderRefineBySpice(t, "Single", false);

  return t;
}

describe("spice golden (synthetic)", () => {
  it("assigns increasing spice along difficulty chain", () => {
    const t = buildMiniTournament();
    const ordering = t.ordering.get("Single") ?? [];
    expect(ordering.length).toBeGreaterThanOrEqual(2);
    const spices = ordering
      .map((s) => s.spice)
      .filter((v): v is number => v !== null);
    expect(spices.length).toBe(ordering.length);
    for (let i = 1; i < spices.length; i++) {
      expect(spices[i]).toBeGreaterThanOrEqual(spices[i - 1]);
    }
    expect(spices[0]).toBeCloseTo(1, 5);
  });
});