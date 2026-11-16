import { readFile, readdir } from "node:fs/promises";
import path from "node:path";
import { catalogConfig } from "../catalog/catalogConfig.js";
import { loadScoreFromGroovestatsLeaderboard } from "../ingest/mapToDbRows.js";
import { Song, type TournamentScore } from "../tournament/models.js";

export async function loadScrapeDir(scrapeRoot: string, catalogName: string) {
  const chartsPath = path.join(scrapeRoot, "charts.json");
  const chartsRaw = JSON.parse(await readFile(chartsPath, "utf-8")) as Record<
    string,
    Record<string, unknown>
  >;

  const songs: Song[] = [];
  for (const c of Object.values(chartsRaw)) {
    songs.push(Song.fromGroovestatsChart(c));
  }

  const entrantDir = path.join(scrapeRoot, "entrant_info");
  const players: { e_id: number; name: string; g_id: number }[] = [];
  try {
    const entrantFiles = await readdir(entrantDir);
    for (const fn of entrantFiles) {
      if (!fn.endsWith(".json")) continue;
      const j = JSON.parse(
        await readFile(path.join(entrantDir, fn), "utf-8")
      ) as { entrant: Record<string, unknown> };
      const e = j.entrant;
      players.push({
        e_id: e["id"] as number,
        name: String(e["name"] ?? ""),
        g_id: (e["membersId"] as number) ?? 0,
      });
    }
  } catch {
    // entrant_info may be missing for score-only rescrapes
  }

  const scoreScalar = catalogConfig(catalogName).scoreScalar;
  const scores: TournamentScore[] = [];
  const scoreDir = path.join(scrapeRoot, "song_scores");
  const scoreFiles = await readdir(scoreDir);
  for (const fn of scoreFiles) {
    if (!fn.endsWith(".json")) continue;
    const j = JSON.parse(await readFile(path.join(scoreDir, fn), "utf-8")) as {
      scores: Record<string, unknown>[];
    };
    for (const row of j.scores) {
      const chartId = row["chartId"] as number;
      const entrantId = row["entrantId"] as number;
      scores.push(
        loadScoreFromGroovestatsLeaderboard(chartId, entrantId, row, scoreScalar)
      );
    }
  }

  return { songs, players, scores };
}