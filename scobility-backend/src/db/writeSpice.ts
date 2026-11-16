import sql from "mssql";

export interface SpiceUpdate {
  global_chart_id: number;
  spice: number;
}

const BATCH = 500;

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
  const tx = new sql.Transaction(pool);
  await tx.begin();
  try {
  const req = new sql.Request(tx);
    await req.query(
      `CREATE TABLE #SpiceBatch (
        global_chart_id INT NOT NULL PRIMARY KEY,
        spice FLOAT NOT NULL
      )`
    );
    for (let i = 0; i < updates.length; i += BATCH) {
      const chunk = updates.slice(i, i + BATCH);
      const table = new sql.Table("#SpiceBatch");
      table.create = false;
      table.columns.add("global_chart_id", sql.Int, { nullable: false });
      table.columns.add("spice", sql.Float, { nullable: false });
      for (const u of chunk) {
        table.rows.add(u.global_chart_id, u.spice);
      }
      await req.bulk(table);
    }
    await req
      .input("catalogId", catalogId)
      .input("ts", spiceCalcTime)
      .query(
        `UPDATE c SET c.spice = b.spice, c.spice_calc_time = @ts
         FROM Chart c
         INNER JOIN #SpiceBatch b ON c.global_chart_id = b.global_chart_id
         WHERE c.catalog_id = @catalogId`
      );
    await req.query(`DROP TABLE #SpiceBatch`);
    await tx.commit();
    return updates.length;
  } catch (e) {
    await tx.rollback();
    throw e;
  }
}