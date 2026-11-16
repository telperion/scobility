import type { ChartRow, PlayerRow, ScoreRow } from "../types/db.js";
import { catalogConfig } from "../catalog/catalogConfig.js";
import { Song, Tournament, type TournamentScore } from "../tournament/models.js";
export function hydrateTournamentFromDb(
  catalogName: string,
  charts: ChartRow[],
  players: PlayerRow[],
  scores: ScoreRow[]
): Tournament {
  const cfg = catalogConfig(catalogName);
  const t = new Tournament(cfg);

  const perfToEntrant = new Map<number, number>();
  for (const p of players) {
    perfToEntrant.set(p.performance_id, p.entrant_id);
  }

  const chartIdByGlobal = new Map<number, number>();
  for (const c of charts) {
    chartIdByGlobal.set(c.global_chart_id, c.chart_id);
    const song = new Song({
      s_id: c.chart_id,
      hash: c.hash,
      title: c.title ?? "",
      subtitle: c.subtitle ?? "",
      artist: c.artist ?? "",
      meter: c.meter ?? 0,
      style: c.style ?? "Single",
      slot: c.slot ?? "Challenge",
      value: c.value ?? 0,
      value_scoring: c.value_scoring ?? 0,
      value_passing: c.value_passing ?? 0,
      spice: c.spice,
    });
    t.songs.set(song.s_id, song);
  }

  for (const row of scores) {
    const chartId = chartIdByGlobal.get(row.global_chart_id);
    if (chartId === undefined) {
      continue;
    }
    const entrantId = perfToEntrant.get(row.performance_id);
    if (entrantId === undefined) {
      continue;
    }
    const song = t.songs.get(chartId);
    if (!song) {
      continue;
    }
    const dbScore = row.score ?? 0;
    const value = 1 - dbScore;
    const existing = song.scores.get(entrantId);
    if (existing && existing.value <= value) {
      continue;
    }
    const ts: TournamentScore = {
      s_id: chartId,
      e_id: entrantId,
      plays: row.plays ?? 1,
      last_played: row.last_played,
      clear: 1,
      value,
    };
    song.scores.set(entrantId, ts);
  }

  return t;
}