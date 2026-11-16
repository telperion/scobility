import sql from "mssql";

export interface SpiceUpdate {
  global_chart_id: number;
  spice: number;
}

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
    let n = 0;
    for (const u of updates) {
      await new sql.Request(tx)
        .input("spice", u.spice)
        .input("ts", spiceCalcTime)
        .input("id", u.global_chart_id)
        .input("catalogId", catalogId)
        .query(
          `UPDATE Chart SET spice = @spice, spice_calc_time = @ts
           WHERE global_chart_id = @id AND catalog_id = @catalogId`
        );
      n++;
    }
    await tx.commit();
    return n;
  } catch (e) {
    await tx.rollback();
    throw e;
  }
}