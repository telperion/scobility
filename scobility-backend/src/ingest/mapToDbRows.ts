import {
  globalChartId,
  performanceId,
  scoreId,
} from "../catalog/idMapping.js";
import type { ChartRow, PlayerRow, ScoreRow } from "../types/db.js";
import { clearName } from "../tournament/constants.js";
import { Song, type TournamentScore } from "../tournament/models.js";

export interface IngestEntities {
  songs: Song[];
  players: { e_id: number; name: string; g_id: number }[];
  scores: TournamentScore[];
}

export function mapIngestToDbRows(
  catalogId: number,
  _catalogName: string,
  data: IngestEntities
): { charts: ChartRow[]; players: PlayerRow[]; scores: ScoreRow[] } {
  const charts: ChartRow[] = data.songs.map((s) => ({
    global_chart_id: globalChartId(catalogId, s.s_id),
    catalog_id: catalogId,
    chart_id: s.s_id,
    hash: s.hash,
    title: s.title,
    subtitle: s.subtitle,
    artist: s.artist,
    meter: s.meter,
    slot: s.slot,
    style: s.style,
    value: s.value,
    value_scoring: s.value_scoring,
    value_passing: s.value_passing,
    spice: null,
    spice_calc_time: null,
  }));

  const players: PlayerRow[] = data.players.map((p) => ({
    performance_id: performanceId(catalogId, p.e_id),
    catalog_id: catalogId,
    entrant_id: p.e_id,
    groovestats_id: p.g_id,
    boogiestats_id: null,
    name: p.name,
    scobility: null,
    timing_power: null,
    comfort_zone: null,
    scobility_calc_time: null,
  }));

  const perfByEntrant = new Map(
    data.players.map((p) => [p.e_id, performanceId(catalogId, p.e_id)])
  );

  const scores: ScoreRow[] = data.scores.map((s, i) => ({
    score_id: scoreId(catalogId, i),
    catalog_id: catalogId,
    global_chart_id: globalChartId(catalogId, s.s_id),
    performance_id: perfByEntrant.get(s.e_id)!,
    plays: s.plays,
    last_played: s.last_played,
    clear: clearName(s.clear),
    score: 1 - s.value,
    prediction: 1,
    prediction_calc_time: new Date(),
  }));

  return { charts, players, scores };
}

export function loadScoreFromGroovestatsLeaderboard(
  chartId: number,
  entrantId: number,
  row: Record<string, unknown>,
  scoreScalar: number
): TournamentScore {
  let lastPlayed: Date | null = null;
  if (typeof row["lastUpdated"] === "string") {
    lastPlayed = new Date(row["lastUpdated"]);
  }
  const ex = (row["ex"] as number) ?? 0;
  return {
    s_id: chartId,
    e_id: entrantId,
    plays: (row["totalPasses"] as number) ?? 1,
    last_played: lastPlayed,
    clear: (row["clearType"] as number) ?? 1,
    value: 1 - ex * scoreScalar,
  };
}