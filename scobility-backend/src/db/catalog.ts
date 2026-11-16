import type sql from "mssql";
import type { CatalogRow, ChartRow, PlayerRow, ScoreRow } from "../types/db.js";

export async function getCatalogByName(
  pool: sql.ConnectionPool,
  name: string
): Promise<CatalogRow> {
  const result = await pool
    .request()
    .input("name", name)
    .query<CatalogRow>(
      `SELECT catalog_id, name, active, last_update, perfect_offset, perfect_score
       FROM Catalog WHERE name = @name`
    );
  if (result.recordset.length !== 1) {
    throw new Error(`Catalog not found: ${name}`);
  }
  return result.recordset[0];
}

export async function loadCharts(
  pool: sql.ConnectionPool,
  catalogId: number
): Promise<ChartRow[]> {
  const result = await pool
    .request()
    .input("catalogId", catalogId)
    .query<ChartRow>(`SELECT * FROM Chart WHERE catalog_id = @catalogId`);
  return result.recordset;
}

export async function loadPlayers(
  pool: sql.ConnectionPool,
  catalogId: number
): Promise<PlayerRow[]> {
  const result = await pool
    .request()
    .input("catalogId", catalogId)
    .query<PlayerRow>(`SELECT * FROM Player WHERE catalog_id = @catalogId`);
  return result.recordset;
}

export async function loadScores(
  pool: sql.ConnectionPool,
  catalogId: number
): Promise<ScoreRow[]> {
  const result = await pool
    .request()
    .input("catalogId", catalogId)
    .query<ScoreRow>(`SELECT * FROM Score WHERE catalog_id = @catalogId`);
  return result.recordset;
}