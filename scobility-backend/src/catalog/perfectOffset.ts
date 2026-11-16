import type { CatalogRow } from "../types/db.js";

/** Asymptote `1 + catalog.perfect_offset` used in score-quality and target EX math. */
export function perfectOffsetSum(catalog: Pick<CatalogRow, "perfect_offset">): number {
  const o = catalog.perfect_offset;
  if (o == null || !Number.isFinite(o) || o < 0) {
    throw new Error("Catalog.perfect_offset must be set in the database");
  }
  return 1 + o;
}

export function requireCatalogPerfectOffset(catalog: CatalogRow): number {
  return perfectOffsetSum(catalog) - 1;
}