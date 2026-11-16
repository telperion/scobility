import { readFile } from "node:fs/promises";
import { Song, type TournamentScore } from "../tournament/models.js";

export interface TourneyJson {
  songs: Record<string, unknown>[];
  players: Record<string, unknown>[];
  scores: Record<string, unknown>[];
}

export async function loadTourneyJson(
  path: string
): Promise<{
  songs: Song[];
  players: { e_id: number; name: string; g_id: number }[];
  scores: TournamentScore[];
}> {
  const raw = JSON.parse(await readFile(path, "utf-8")) as TourneyJson;
  const songs = raw.songs.map((s) => Song.fromDump(s));
  const players = raw.players.map((p) => ({
    e_id: p["e_id"] as number,
    name: String(p["name"] ?? ""),
    g_id: (p["g_id"] as number) ?? 0,
  }));
  const scores: TournamentScore[] = raw.scores.map((s) => ({
    s_id: s["s_id"] as number,
    e_id: s["e_id"] as number,
    plays: (s["plays"] as number) ?? 1,
    last_played: s["last_played"]
      ? new Date(String(s["last_played"]))
      : null,
    clear: (s["clear"] as number) ?? 1,
    value: s["value"] as number,
  }));
  return { songs, players, scores };
}