import { writeFile } from "node:fs/promises";
import path from "node:path";
import type { Song, Tournament } from "../tournament/models.js";

function csvEscape(value: string | number): string {
  const s = String(value);
  if (/[",\n\r]/.test(s)) {
    return `"${s.replace(/"/g, '""')}"`;
  }
  return s;
}

function spiceCsvForStyle(style: string, songs: Song[]): string {
  const header = [
    "chart_id",
    "style",
    "log2_spice",
    "spice",
    "hash",
    "title",
    "subtitle",
    "artist",
    "meter",
    "slot",
    "value_scoring",
    "value_passing",
    "value",
  ].join(",");
  const rows = songs
    .filter((s) => s.spice != null)
    .map((s) => {
      const log2 = Math.log2(s.spice!);
      return [
        s.s_id,
        style,
        log2.toFixed(6),
        s.spice,
        csvEscape(s.hash),
        csvEscape(s.title),
        csvEscape(s.subtitle),
        csvEscape(s.artist),
        s.meter,
        csvEscape(s.slot),
        s.value_scoring,
        s.value_passing,
        s.value,
      ].join(",");
    });
  return [header, ...rows].join("\n") + "\n";
}

function spiceLogForStyle(songs: Song[]): string {
  return (
    songs
      .filter((s) => s.spice != null)
      .map((s) => s.toSpiceLogLine())
      .join("\n") + "\n"
  );
}

function baseName(catalog: string, style: string): string {
  const safeCatalog = catalog.replace(/[^\w.-]+/g, "_");
  return `spice_ranking_${safeCatalog}_${style.toLowerCase()}`;
}

export async function echoSpiceRankings(
  catalog: string,
  t: Tournament,
  styles: string[],
  outDir = process.cwd()
): Promise<void> {
  for (const style of styles) {
    const ordering = t.ordering.get(style) ?? [];
    if (ordering.length === 0) {
      console.warn(`No ordering for style ${style}; skipping echo output`);
      continue;
    }
    const base = baseName(catalog, style);
    const logPath = path.join(outDir, `${base}.log`);
    const csvPath = path.join(outDir, `${base}.csv`);
    await writeFile(logPath, spiceLogForStyle(ordering), "utf-8");
    await writeFile(csvPath, spiceCsvForStyle(style, ordering), "utf-8");
    console.log(`Wrote ${logPath}`);
    console.log(`Wrote ${csvPath}`);
  }
}