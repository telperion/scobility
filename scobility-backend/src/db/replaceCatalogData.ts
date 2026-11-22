import sql from "mssql";
import type { ChartRow, PlayerRow, ScoreRow } from "../types/db.js";
import {
  clipChartText,
  clipPlayerText,
  clipScoreText,
  logSqlError,
} from "./sqlError.js";

const DELETE_BATCH = 25_000;
const CHART_INSERT_BATCH = 25;
const PLAYER_INSERT_BATCH = 50;
const SCORE_INSERT_BATCH = 40;

function sqlNull<T>(v: T | null | undefined): T | null {
  return v === undefined ? null : v;
}

export async function replaceCatalogData(
  pool: sql.ConnectionPool,
  catalogId: number,
  charts: ChartRow[],
  players: PlayerRow[],
  scores: ScoreRow[]
): Promise<void> {
  for (const c of charts) {
    clipChartText(c);
  }
  for (const p of players) {
    clipPlayerText(p);
  }
  for (const s of scores) {
    clipScoreText(s);
  }

  await deleteCatalogRows(pool, catalogId);

  console.log("Inserting charts, players, scores (batched SQL)...");
  try {
    await insertChartsBatched(pool, charts);
    await insertPlayersBatched(pool, players);
    await insertScoresBatched(pool, scores);
    await pool
      .request()
      .input("catalogId", catalogId)
      .query(
        `UPDATE Catalog SET last_update = SYSUTCDATETIME() WHERE catalog_id = @catalogId`
      );
  } catch (err) {
    logSqlError("replaceCatalogData", err);
    throw err;
  }
}

async function deleteCatalogRows(
  pool: sql.ConnectionPool,
  catalogId: number
): Promise<void> {
  console.log(`Removing existing rows for catalog_id=${catalogId} (batched)...`);
  const tables = ["dbo.Score", "dbo.Player", "dbo.Chart"] as const;
  for (const table of tables) {
    let removed = 0;
    for (;;) {
      const result = await pool
        .request()
        .input("catalogId", catalogId)
        .input("batch", DELETE_BATCH)
        .query(
          `DELETE TOP (@batch) FROM ${table} WHERE catalog_id = @catalogId`
        );
      const n = rowsDeleted(result);
      if (n === 0) {
        break;
      }
      removed += n;
      console.log(`  ${table}: ${removed} deleted`);
    }
  }
}

function rowsDeleted(result: sql.IResult<unknown>): number {
  if (!result.rowsAffected?.length) {
    return 0;
  }
  return result.rowsAffected.reduce((sum, n) => sum + n, 0);
}

async function insertChartsBatched(
  pool: sql.ConnectionPool,
  charts: ChartRow[]
): Promise<void> {
  for (let i = 0; i < charts.length; i += CHART_INSERT_BATCH) {
    const chunk = charts.slice(i, i + CHART_INSERT_BATCH);
    const req = pool.request();
    const tuples: string[] = [];
    chunk.forEach((c, j) => {
      req.input(`g${j}`, c.global_chart_id);
      req.input(`c${j}`, c.catalog_id);
      req.input(`i${j}`, c.chart_id);
      req.input(`h${j}`, c.hash);
      req.input(`t${j}`, sql.NVarChar, sqlNull(c.title));
      req.input(`st${j}`, sql.NVarChar, sqlNull(c.subtitle));
      req.input(`a${j}`, sql.NVarChar, sqlNull(c.artist));
      req.input(`m${j}`, sql.Float, sqlNull(c.meter));
      req.input(`sl${j}`, sql.NVarChar, sqlNull(c.slot));
      req.input(`sy${j}`, sql.NVarChar, sqlNull(c.style));
      req.input(`v${j}`, sql.Float, sqlNull(c.value));
      req.input(`vs${j}`, sql.Float, sqlNull(c.value_scoring));
      req.input(`vp${j}`, sql.Float, sqlNull(c.value_passing));
      req.input(`sp${j}`, sql.Float, sqlNull(c.spice));
      req.input(`sct${j}`, sql.DateTime2, sqlNull(c.spice_calc_time));
      tuples.push(
        `(@g${j},@c${j},@i${j},@h${j},@t${j},@st${j},@a${j},@m${j},@sl${j},@sy${j},@v${j},@vs${j},@vp${j},@sp${j},@sct${j})`
      );
    });
    try {
      await req.query(
        `INSERT INTO dbo.Chart (global_chart_id, catalog_id, chart_id, hash, title, subtitle,
          artist, meter, slot, style, value, value_scoring, value_passing, spice, spice_calc_time)
         VALUES ${tuples.join(",")}`
      );
    } catch (err) {
      logSqlError(`Chart insert @${i}`, err);
      throw err;
    }
    console.log(
      `Chart: ${Math.min(i + CHART_INSERT_BATCH, charts.length)} / ${charts.length}`
    );
  }
}

async function insertPlayersBatched(
  pool: sql.ConnectionPool,
  players: PlayerRow[]
): Promise<void> {
  for (let i = 0; i < players.length; i += PLAYER_INSERT_BATCH) {
    const chunk = players.slice(i, i + PLAYER_INSERT_BATCH);
    const req = pool.request();
    const tuples: string[] = [];
    chunk.forEach((p, j) => {
      req.input(`p${j}`, p.performance_id);
      req.input(`c${j}`, p.catalog_id);
      req.input(`e${j}`, p.entrant_id);
      req.input(`g${j}`, sql.Int, sqlNull(p.groovestats_id));
      req.input(`b${j}`, sql.Int, sqlNull(p.boogiestats_id));
      req.input(`n${j}`, sql.NVarChar, sqlNull(p.name));
      req.input(`sc${j}`, sql.Float, sqlNull(p.scobility));
      req.input(`tp${j}`, sql.Float, sqlNull(p.timing_power));
      req.input(`cz${j}`, sql.Float, sqlNull(p.comfort_zone));
      req.input(`sct${j}`, sql.DateTime2, sqlNull(p.scobility_calc_time));
      tuples.push(
        `(@p${j},@c${j},@e${j},@g${j},@b${j},@n${j},@sc${j},@tp${j},@cz${j},@sct${j})`
      );
    });
    try {
      await req.query(
        `INSERT INTO dbo.Player (performance_id, catalog_id, entrant_id, groovestats_id,
          boogiestats_id, name, scobility, timing_power, comfort_zone, scobility_calc_time)
         VALUES ${tuples.join(",")}`
      );
    } catch (err) {
      logSqlError(`Player insert @${i}`, err);
      throw err;
    }
    console.log(
      `Player: ${Math.min(i + PLAYER_INSERT_BATCH, players.length)} / ${players.length}`
    );
  }
}

async function insertScoresBatched(
  pool: sql.ConnectionPool,
  scores: ScoreRow[]
): Promise<void> {
  for (let i = 0; i < scores.length; i += SCORE_INSERT_BATCH) {
    const chunk = scores.slice(i, i + SCORE_INSERT_BATCH);
    const req = pool.request();
    const tuples: string[] = [];
    chunk.forEach((s, j) => {
      req.input(`id${j}`, sql.BigInt, s.score_id);
      req.input(`c${j}`, s.catalog_id);
      req.input(`gc${j}`, s.global_chart_id);
      req.input(`pf${j}`, s.performance_id);
      req.input(`pl${j}`, sql.Int, sqlNull(s.plays));
      req.input(`lp${j}`, sql.DateTime2, sqlNull(s.last_played));
      req.input(`cl${j}`, sql.NVarChar, sqlNull(s.clear));
      req.input(`sc${j}`, sql.Float, sqlNull(s.score));
      req.input(`pr${j}`, sql.Float, sqlNull(s.prediction));
      req.input(`pt${j}`, sql.DateTime2, sqlNull(s.prediction_calc_time));
      tuples.push(
        `(@id${j},@c${j},@gc${j},@pf${j},@pl${j},@lp${j},@cl${j},@sc${j},@pr${j},@pt${j})`
      );
    });
    try {
      await req.query(
        `INSERT INTO dbo.Score (score_id, catalog_id, global_chart_id, performance_id, plays,
          last_played, clear, score, prediction, prediction_calc_time)
         VALUES ${tuples.join(",")}`
      );
    } catch (err) {
      logSqlError(`Score insert @${i}`, err);
      throw err;
    }
    if (
      i % (SCORE_INSERT_BATCH * 20) === 0 ||
      i + SCORE_INSERT_BATCH >= scores.length
    ) {
      console.log(
        `Score: ${Math.min(i + SCORE_INSERT_BATCH, scores.length)} / ${scores.length}`
      );
    }
  }
}