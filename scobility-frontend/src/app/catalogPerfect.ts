/** Catalog row fields from GET /catalog/{name} used in score quality math. */
export interface LoadedCatalogMeta {
  catalog_id: number;
  name: string;
  active?: boolean;
  last_update?: string | null;
  perfect_offset: number;
  perfect_score: number | null;
}

let catalogMeta: LoadedCatalogMeta | null = null;

export function setCatalogParams(meta: LoadedCatalogMeta): void {
  if (
    meta.perfect_offset == null ||
    !Number.isFinite(meta.perfect_offset) ||
    meta.perfect_offset < 0
  ) {
    throw new Error(
      `Invalid catalog perfect_offset for ${meta.name ?? "catalog"}`
    );
  }
  catalogMeta = meta;
}

export function clearCatalogParams(): void {
  catalogMeta = null;
}

export function isCatalogParamsLoaded(): boolean {
  return catalogMeta != null;
}

export function requireCatalogMeta(): LoadedCatalogMeta {
  if (!catalogMeta) {
    throw new Error(
      "Catalog metadata not loaded; fetch GET /catalog/{name} before score quality math"
    );
  }
  return catalogMeta;
}

/** `1 + perfect_offset` from the Catalog table (avoids log singularity at perfect). */
export function catalogPerfectOffsetSum(): number {
  return 1 + requireCatalogMeta().perfect_offset;
}

export function catalogPerfectScore(): number | null {
  return requireCatalogMeta().perfect_score;
}

export function catalogLastUpdate(): string | null {
  return requireCatalogMeta().last_update ?? null;
}

/**
 * Score quality in log-spice space (higher = better).
 * `dbScore` is the API `Score.score` field (1 − in-game value for ITL EX).
 */
export function scoreQualityFromDb(
  logSpice: number,
  dbScore: number
): number {
  const meta = requireCatalogMeta();
  const offsetSum = 1 + meta.perfect_offset;
  const pMin = Math.log2(offsetSum);
  const perfectScore = meta.perfect_score;
  if (perfectScore != null && perfectScore > 0 && dbScore >= 1) {
    const rescaled = 1 - dbScore / perfectScore;
    const pScore = Math.log2(meta.perfect_offset + rescaled) - pMin;
    return logSpice - pScore;
  }
  return logSpice - Math.log2(offsetSum - dbScore);
}