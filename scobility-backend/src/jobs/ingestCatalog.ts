import type { DbTarget } from "../types/db.js";
import { withPool } from "../db/connection.js";
import { getCatalogByName } from "../db/catalog.js";
import { replaceCatalogData } from "../db/replaceCatalogData.js";
import { mapIngestToDbRows } from "../ingest/mapToDbRows.js";
import { scrapeGroovestats } from "../scrape/groovestats.js";
import { loadScrapeDir } from "../scrape/loadScrapeDir.js";
import { loadTourneyJson } from "../scrape/loadTourneyJson.js";

export type IngestSource = "groovestats" | "dir" | "json";

export interface IngestCatalogOptions {
  target: DbTarget;
  catalog: string;
  source: IngestSource;
  path?: string;
  allowProduction: boolean;
}

export async function runIngestCatalog(opts: IngestCatalogOptions): Promise<void> {
  if (opts.target === "production" && !opts.allowProduction) {
    throw new Error("ingest-catalog on production requires --allow-production");
  }

  let data;
  if (opts.source === "groovestats") {
    console.log(`Scraping GrooveStats for catalog ${opts.catalog}...`);
    data = await scrapeGroovestats(opts.catalog);
  } else if (opts.source === "dir") {
    if (!opts.path) throw new Error("--path required for --source dir");
    console.log(`Loading scrape dir ${opts.path}...`);
    data = await loadScrapeDir(opts.path, opts.catalog);
  } else {
    if (!opts.path) throw new Error("--path required for --source json");
    console.log(`Loading tournament JSON ${opts.path}...`);
    data = await loadTourneyJson(opts.path);
  }

  await withPool(opts.target, async (pool) => {
    const catalog = await getCatalogByName(pool, opts.catalog);
    const rows = mapIngestToDbRows(catalog.catalog_id, opts.catalog, data);
    console.log(
      `Inserting ${rows.charts.length} charts, ${rows.players.length} players, ${rows.scores.length} scores...`
    );
    await replaceCatalogData(
      pool,
      catalog.catalog_id,
      rows.charts,
      rows.players,
      rows.scores
    );
    console.log("Ingest complete.");
  });
}