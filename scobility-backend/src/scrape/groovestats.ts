import { loadScoreFromGroovestatsLeaderboard } from "../ingest/mapToDbRows.js";
import { catalogConfig, tourneyHostForCatalog } from "../catalog/catalogConfig.js";
import { Song, type TournamentScore } from "../tournament/models.js";

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

async function getJson(url: string, init?: RequestInit): Promise<unknown> {
  const r = await fetch(url, init);
  return r.json();
}

export async function scrapeGroovestats(catalogName: string) {
  const host = tourneyHostForCatalog(catalogName);
  const base = `https://${host}.groovestats.com`;
  const scoreScalar = catalogConfig(catalogName).scoreScalar;

  const charts: Record<number, Record<string, unknown>> = {};
  const strikes: number[] = [];
  for (let i = 0; i < 10000; i++) {
    await sleep(500);
    let j: { success?: boolean; data?: Record<string, unknown>; message?: string };
    try {
      j = (await getJson(`${base}/api/chart/${i}`)) as typeof j;
    } catch (e) {
      j = { success: false, message: String(e) };
    }
    if (!j.success) {
      console.warn(`${i}: ${j.message ?? ""}`);
      strikes.push(i);
      if (strikes.length > 100) break;
    } else {
      strikes.length = 0;
      charts[i] = j.data ?? {};
      console.log(`${i}: ${charts[i]["artist"]} - "${charts[i]["title"]}"`);
    }
  }

  const players: { e_id: number; name: string; g_id: number }[] = [];
  strikes.length = 0;
  for (let i = 0; i < 10000; i++) {
    await sleep(500);
    let j: { success?: boolean; data?: { entrant: Record<string, unknown> }; message?: string };
    try {
      j = (await getJson(`${base}/api/entrant/${i}`)) as typeof j;
    } catch (e) {
      j = { success: false, message: String(e) };
    }
    if (!j.success) {
      strikes.push(i);
      if (strikes.length > 20) break;
    } else {
      strikes.length = 0;
      const e = j.data!.entrant;
      players.push({
        e_id: e["id"] as number,
        name: String(e["name"] ?? ""),
        g_id: (e["membersId"] as number) ?? 0,
      });
      console.log(`${i}: ${e["name"]}`);
    }
  }

  const songs: Song[] = Object.values(charts).map((c) =>
    Song.fromGroovestatsChart(c)
  );

  const scores: TournamentScore[] = [];
  strikes.length = 0;
  let total = 0;
  for (const c of Object.values(charts)) {
    total++;
    if (total > 10000) break;
    const i = (c["id"] as number) ?? 0;
    const hash = c["hash"] as string;

    let j: { success?: boolean; data?: { leaderboard?: Record<string, unknown>[] }; message?: string } =
      { success: false };
    for (let retries = 0; retries < 5; retries++) {
      await sleep(500);
      try {
        const r = await fetch(`${base}/api/score/chartTopScores`, {
          method: "POST",
          headers: { "Content-Type": "application/x-www-form-urlencoded" },
          body: `chartHash=${encodeURIComponent(hash)}`,
        });
        if (r.status > 400) continue;
        j = (await r.json()) as typeof j;
        break;
      } catch (e) {
        j = { success: false, message: String(e) };
      }
    }

    if (!j.success) {
      strikes.push(i);
      if (strikes.length > 20) break;
      continue;
    }
    strikes.length = 0;
    const leaderboard = j.data?.leaderboard ?? [];
    for (const row of leaderboard) {
      row["chartId"] = i;
      const entrantId = row["entrantId"] as number;
      scores.push(
        loadScoreFromGroovestatsLeaderboard(i, entrantId, row, scoreScalar)
      );
    }
    console.log(`${i}: ${leaderboard.length} scores`);
  }

  return { songs, players, scores };
}