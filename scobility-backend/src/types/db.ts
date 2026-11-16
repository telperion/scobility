export type DbTarget = "local" | "production";

export interface CatalogRow {
  catalog_id: number;
  name: string;
  active: boolean;
  last_update: Date | null;
  perfect_offset: number | null;
  perfect_score: number | null;
}

export interface ChartRow {
  global_chart_id: number;
  catalog_id: number;
  chart_id: number;
  hash: string;
  title: string | null;
  subtitle: string | null;
  artist: string | null;
  meter: number | null;
  slot: string | null;
  style: string | null;
  value: number | null;
  value_scoring: number | null;
  value_passing: number | null;
  spice: number | null;
  spice_calc_time: Date | null;
}

export interface PlayerRow {
  performance_id: number;
  catalog_id: number;
  entrant_id: number;
  groovestats_id: number | null;
  boogiestats_id: number | null;
  name: string | null;
  scobility: number | null;
  timing_power: number | null;
  comfort_zone: number | null;
  scobility_calc_time: Date | null;
}

export interface ScoreRow {
  score_id: number;
  catalog_id: number;
  global_chart_id: number;
  performance_id: number;
  plays: number | null;
  last_played: Date | null;
  clear: string | null;
  score: number | null;
  prediction: number | null;
  prediction_calc_time: Date | null;
}

export interface RelationshipRow {
  relationship_id: number;
  x_id: number;
  y_id: number;
  common: number | null;
  relation: number | null;
  strength: number | null;
}