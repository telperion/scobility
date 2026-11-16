export const TOURNAMENT_DEFAULTS = {
  MONO_THRESHOLD: 0.999999,
  MIN_COMMON_PLAYERS: 6,
  ITERATIONS_MONOTONIC_SORT: 10,
  ITERATIONS_SCOBILITY_SORT: 500,
  SCOBILITY_WINDOW_LOWER: 10,
  SCOBILITY_WINDOW_UPPER: 6,
};

export const RELATIONSHIP_DEFAULTS = {
  WEIGHT_OFFSET: 0.5,
  MIN_COMMON_PLAYERS: 6,
};

export const CLEAR_NAMES = [
  "FAIL",
  "PASS",
  "BATTERY",
  "FC",
  "FEC",
  "QUAD",
  "QUINT",
] as const;

export function clearName(clearInt: number): string {
  return CLEAR_NAMES[clearInt] ?? "PASS";
}