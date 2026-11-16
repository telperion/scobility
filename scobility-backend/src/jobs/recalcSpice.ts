import type { DbTarget } from "../types/db.js";
import { withPool } from "../db/connection.js";
import {
  getCatalogByName,
  loadCharts,
  loadPlayers,
  loadScores,
} from "../db/catalog.js";
import { hydrateTournamentFromDb } from "../db/hydrateTournament.js";
import { writeSpiceUpdates } from "../db/writeSpice.js";
import { globalChartId } from "../catalog/idMapping.js";
import { runSpicePipeline } from "../tournament/spicePipeline.js";
import { echoSpiceRankings } from "../export/spiceEcho.js";

export interface RecalcSpiceOptions {
  target: DbTarget;
  catalog: string;
  style: "Single" | "Double" | "all";
  dryRun: boolean;
  verbose: boolean;
  echo: boolean;
  echoDir?: string;
}

export async function runRecalcSpice(opts: RecalcSpiceOptions): Promise<void> {
  const styles =
    opts.style === "all" ? ["Single", "Double"] : [opts.style];
  const spiceCalcTime = new Date();

  await withPool(opts.target, async (pool) => {
    const catalog = await getCatalogByName(pool, opts.catalog);
    const [charts, players, scores] = await Promise.all([
      loadCharts(pool, catalog.catalog_id),
      loadPlayers(pool, catalog.catalog_id),
      loadScores(pool, catalog.catalog_id),
    ]);

    if (charts.length === 0) {
      throw new Error(`No charts in catalog ${opts.catalog}`);
    }

    console.log(
      `Loaded ${charts.length} charts, ${players.length} players, ${scores.length} scores`
    );

    const t = hydrateTournamentFromDb(opts.catalog, charts, players, scores);
    runSpicePipeline(t, styles, opts.verbose);

    const updates: { global_chart_id: number; spice: number }[] = [];
    for (const style of styles) {
      const ordering = t.ordering.get(style) ?? [];
      for (const song of ordering) {
        if (song.spice !== null && song.spice !== undefined) {
          updates.push({
            global_chart_id: globalChartId(catalog.catalog_id, song.s_id),
            spice: song.spice,
          });
        }
      }
    }

    const byChart = new Map<number, number>();
    for (const u of updates) {
      byChart.set(u.global_chart_id, u.spice);
    }
    const uniqueUpdates = [...byChart.entries()].map(([global_chart_id, spice]) => ({
      global_chart_id,
      spice,
    }));

    const n = await writeSpiceUpdates(
      pool,
      catalog.catalog_id,
      uniqueUpdates,
      spiceCalcTime,
      opts.dryRun
    );
    console.log(`Updated spice on ${n} charts.`);

    if (opts.echo) {
      await echoSpiceRankings(
        opts.catalog,
        t,
        styles,
        opts.echoDir ?? process.cwd()
      );
    }
  });
}