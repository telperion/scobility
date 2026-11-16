import sql from "mssql";
import type { ChartRow, PlayerRow, ScoreRow } from "../types/db.js";

const CHUNK = 500;

export async function replaceCatalogData(
  pool: sql.ConnectionPool,
  catalogId: number,
  charts: ChartRow[],
  players: PlayerRow[],
  scores: ScoreRow[]
): Promise<void> {
  const tx = new sql.Transaction(pool);
  await tx.begin();
  try {
    const req = new sql.Request(tx);
    await req.input("catalogId", catalogId).query(
      `DELETE FROM Score WHERE catalog_id = @catalogId;
       DELETE FROM Player WHERE catalog_id = @catalogId;
       DELETE FROM Chart WHERE catalog_id = @catalogId;`
    );

    await insertBatched(tx, "Chart", charts, insertChart);
    await insertBatched(tx, "Player", players, insertPlayer);
    await insertBatched(tx, "Score", scores, insertScore);

    await new sql.Request(tx)
      .input("catalogId", catalogId)
      .query(`UPDATE Catalog SET last_update = SYSUTCDATETIME() WHERE catalog_id = @catalogId`);

    await tx.commit();
  } catch (e) {
    await tx.rollback();
    throw e;
  }
}

async function insertBatched<T>(
  tx: sql.Transaction,
  label: string,
  rows: T[],
  inserter: (req: sql.Request, row: T) => Promise<void>
): Promise<void> {
  for (let i = 0; i < rows.length; i += CHUNK) {
    const slice = rows.slice(i, i + CHUNK);
    for (const row of slice) {
      await inserter(new sql.Request(tx), row);
    }
    console.log(`${label}: ${Math.min(i + CHUNK, rows.length)} / ${rows.length}`);
  }
}

async function insertChart(req: sql.Request, c: ChartRow): Promise<void> {
  await req
    .input("global_chart_id", c.global_chart_id)
    .input("catalog_id", c.catalog_id)
    .input("chart_id", c.chart_id)
    .input("hash", c.hash)
    .input("title", c.title)
    .input("subtitle", c.subtitle)
    .input("artist", c.artist)
    .input("meter", c.meter)
    .input("slot", c.slot)
    .input("style", c.style)
    .input("value", c.value)
    .input("value_scoring", c.value_scoring)
    .input("value_passing", c.value_passing)
    .input("spice", c.spice)
    .input("spice_calc_time", c.spice_calc_time)
    .query(
      `INSERT INTO Chart (global_chart_id, catalog_id, chart_id, hash, title, subtitle, artist,
        meter, slot, style, value, value_scoring, value_passing, spice, spice_calc_time)
       VALUES (@global_chart_id, @catalog_id, @chart_id, @hash, @title, @subtitle, @artist,
        @meter, @slot, @style, @value, @value_scoring, @value_passing, @spice, @spice_calc_time)`
    );
}

async function insertPlayer(req: sql.Request, p: PlayerRow): Promise<void> {
  await req
    .input("performance_id", p.performance_id)
    .input("catalog_id", p.catalog_id)
    .input("entrant_id", p.entrant_id)
    .input("groovestats_id", p.groovestats_id)
    .input("boogiestats_id", p.boogiestats_id)
    .input("name", p.name)
    .input("scobility", p.scobility)
    .input("timing_power", p.timing_power)
    .input("comfort_zone", p.comfort_zone)
    .input("scobility_calc_time", p.scobility_calc_time)
    .query(
      `INSERT INTO Player (performance_id, catalog_id, entrant_id, groovestats_id, boogiestats_id,
        name, scobility, timing_power, comfort_zone, scobility_calc_time)
       VALUES (@performance_id, @catalog_id, @entrant_id, @groovestats_id, @boogiestats_id,
        @name, @scobility, @timing_power, @comfort_zone, @scobility_calc_time)`
    );
}

async function insertScore(req: sql.Request, s: ScoreRow): Promise<void> {
  await req
    .input("score_id", s.score_id)
    .input("catalog_id", s.catalog_id)
    .input("global_chart_id", s.global_chart_id)
    .input("performance_id", s.performance_id)
    .input("plays", s.plays)
    .input("last_played", s.last_played)
    .input("clear", s.clear)
    .input("score", s.score)
    .input("prediction", s.prediction)
    .input("prediction_calc_time", s.prediction_calc_time)
    .query(
      `INSERT INTO Score (score_id, catalog_id, global_chart_id, performance_id, plays,
        last_played, clear, score, prediction, prediction_calc_time)
       VALUES (@score_id, @catalog_id, @global_chart_id, @performance_id, @plays,
        @last_played, @clear, @score, @prediction, @prediction_calc_time)`
    );
}