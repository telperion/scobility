import sql from "mssql";
import { logSqlError } from "./sqlError.js";

export interface SpiceUpdate {
  global_chart_id: number;
  spice: number;
}

const UPDATE_BATCH = 50;

export async function writeSpiceUpdates(
  pool: sql.ConnectionPool,
  catalogId: number,
  updates: SpiceUpdate[],
  spiceCalcTime: Date,
  dryRun: boolean
): Promise<number> {
  if (dryRun) {
    console.log(`Dry run: would update ${updates.length} chart spice values`);
    return updates.length;
  }

  console.log(`Writing spice to ${updates.length} charts...`);
  for (let i = 0; i < updates.length; i += UPDATE_BATCH) {
    const chunk = updates.slice(i, i + UPDATE_BATCH);
    const req = pool.request().input("catalogId", catalogId).input("ts", spiceCalcTime);
    const tuples: string[] = [];
    chunk.forEach((u, j) => {
      req.input(`g${j}`, u.global_chart_id);
      req.input(`s${j}`, sql.Float, u.spice);
      tuples.push(`(@g${j}, @s${j})`);
    });
    try {
      await req.query(
        `UPDATE c SET c.spice = v.spice, c.spice_calc_time = @ts
         FROM dbo.Chart c
         INNER JOIN (VALUES ${tuples.join(",")}) AS v(global_chart_id, spice)
           ON c.global_chart_id = v.global_chart_id AND c.catalog_id = @catalogId`
      );
    } catch (err) {
      logSqlError(`Spice update @${i}`, err);
      throw err;
    }
    console.log(
      `Spice write: ${Math.min(i + UPDATE_BATCH, updates.length)} / ${updates.length}`
    );
  }
  return updates.length;
}