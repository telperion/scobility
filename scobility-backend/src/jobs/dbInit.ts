import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import sql from "mssql";
import type { DbTarget } from "../types/db.js";
import { mssqlPoolConfig, resolveProfile, withPool } from "../db/connection.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const sqlRoot = path.join(__dirname, "..", "..", "sql");

async function runSqlFile(pool: sql.ConnectionPool, fileName: string): Promise<void> {
  const content = await readFile(path.join(sqlRoot, fileName), "utf-8");
  const batches = content
    .split(/^\s*GO\s*$/gim)
    .map((b) => b.trim())
    .filter(Boolean);
  for (const batch of batches) {
    await pool.request().query(batch);
  }
}

export async function runDbInit(target: DbTarget, force = false): Promise<void> {
  if (target === "production" && !force) {
    throw new Error("db-init on production requires --force");
  }

  if (target === "local") {
    const profile = resolveProfile("local");
    const master = await sql.connect(
      mssqlPoolConfig(profile, "master")
    );
    try {
      const dbName = profile.database.replace(/]/g, "]]");
      await master.request().query(
        `IF DB_ID(N'${dbName.replace(/'/g, "''")}') IS NULL CREATE DATABASE [${dbName}]`
      );
    } finally {
      await master.close();
    }
  }

  await withPool(target, async (pool) => {
    console.log("Applying schema.sql...");
    await runSqlFile(pool, "schema.sql");
    console.log("Applying seed-catalogs.sql...");
    await runSqlFile(pool, "seed-catalogs.sql");
    const counts = await pool.request().query(`
      SELECT 'Catalog' AS tbl, COUNT(*) AS n FROM Catalog
      UNION ALL SELECT 'Chart', COUNT(*) FROM Chart
      UNION ALL SELECT 'Player', COUNT(*) FROM Player
      UNION ALL SELECT 'Score', COUNT(*) FROM Score`);
    console.table(counts.recordset);
  });
}