export function globalChartId(catalogId: number, chartId: number): number {
  return catalogId * 1000 + chartId;
}

export function performanceId(catalogId: number, entrantId: number): number {
  return catalogId * 10000 + entrantId;
}

export function scoreId(catalogId: number, rowIndex: number): number {
  return catalogId * 1_000_000 + rowIndex;
}