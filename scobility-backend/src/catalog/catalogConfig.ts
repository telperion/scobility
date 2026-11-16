export interface CatalogRuntimeConfig {
  tourneyHost?: string;
  scoreScalar: number;
  perfectOffset: number;
  relationshipMaxNegLimit: number;
  relationshipMinNegLimit: number;
}

const DEFAULTS: CatalogRuntimeConfig = {
  scoreScalar: 0.0001,
  perfectOffset: 0.003,
  relationshipMaxNegLimit: 0.3,
  relationshipMinNegLimit: 0.0002,
};

const BY_NAME: Record<string, Partial<CatalogRuntimeConfig>> = {
  ITL2022: { tourneyHost: "itl2022" },
  ITL2023: { tourneyHost: "itl2023" },
  ITL2024: { tourneyHost: "itl2024" },
  ITL2025: { tourneyHost: "itl2025" },
  ITL2026: { tourneyHost: "itl2026" },
  GS: {
    tourneyHost: "gs",
    scoreScalar: 1,
    perfectOffset: 0.03,
    relationshipMaxNegLimit: 0.2,
    relationshipMinNegLimit: 0.001,
  },
};

export function catalogConfig(catalogName: string): CatalogRuntimeConfig {
  const overrides = BY_NAME[catalogName] ?? {};
  return { ...DEFAULTS, ...overrides };
}

export function tourneyHostForCatalog(catalogName: string): string {
  const cfg = catalogConfig(catalogName);
  if (!cfg.tourneyHost) {
    throw new Error(`No GrooveStats tourney host mapped for catalog ${catalogName}`);
  }
  return cfg.tourneyHost;
}