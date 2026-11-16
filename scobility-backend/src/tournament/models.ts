import type { CatalogRuntimeConfig } from "../catalog/catalogConfig.js";

export interface TournamentScore {
  s_id: number;
  e_id: number;
  plays: number;
  last_played: Date | null;
  clear: number;
  value: number;
}

export class Song {
  s_id: number;
  hash: string;
  title: string;
  subtitle: string;
  artist: string;
  meter: number;
  style: string;
  slot: string;
  value: number;
  value_scoring: number;
  value_passing: number;
  scores: Map<number, TournamentScore>;
  spice: number | null;

  constructor(init?: Partial<Song> & { s_id?: number }) {
    this.s_id = init?.s_id ?? -1;
    this.hash = init?.hash ?? "";
    this.title = init?.title ?? "";
    this.subtitle = init?.subtitle ?? "";
    this.artist = init?.artist ?? "";
    this.meter = init?.meter ?? 0;
    this.style = init?.style ?? "Single";
    this.slot = init?.slot ?? "Challenge";
    this.value = init?.value ?? 0;
    this.value_scoring = init?.value_scoring ?? 0;
    this.value_passing = init?.value_passing ?? 0;
    this.scores = init?.scores ?? new Map();
    this.spice = init?.spice ?? null;
  }

  static fromDump(data: Record<string, unknown>): Song {
    const s = new Song();
    const fields = [
      "s_id",
      "hash",
      "title",
      "subtitle",
      "artist",
      "meter",
      "style",
      "slot",
      "value",
      "value_scoring",
      "value_passing",
      "spice",
    ] as const;
    for (const fn of fields) {
      const v = data[fn];
      if (v !== undefined && v !== null) {
        (s as unknown as Record<string, unknown>)[fn] = v;
      }
    }
    return s;
  }

  static fromGroovestatsChart(data: Record<string, unknown>): Song {
    const s = new Song();
    s.s_id = data["id"] as number;
    s.hash = data["hash"] as string;
    s.value = (data["points"] as number) ?? 0;
    s.value_scoring = (data["pointsScoring"] as number) ?? s.value;
    s.value_passing = (data["pointsPassing"] as number) ?? 0;

    const titleRomaji = String(data["titleRomaji"] ?? "").trim();
    s.title =
      String(data["title"] ?? "") +
      (titleRomaji !== "" ? ` (${titleRomaji})` : "");

    const subtitleRomaji = String(data["subtitleRomaji"] ?? "").trim();
    s.subtitle =
      String(data["subtitle"] ?? "") +
      (subtitleRomaji !== "" ? ` (${subtitleRomaji})` : "");

    const artistRomaji = String(data["artistRomaji"] ?? "").trim();
    s.artist =
      String(data["artist"] ?? "") +
      (artistRomaji !== "" ? ` (${artistRomaji})` : "");

    s.meter = (data["meter"] as number) ?? 0;
    s.slot = String(data["difficulty"] ?? "Challenge");
    const playstyle = (data["playstyle"] as number) ?? 1;
    s.style = playstyle === 2 ? "Double" : "Single";
    s.scores = new Map();
    s.spice = null;
    return s;
  }

  get fullTitle(): string {
    return this.subtitle.length > 0
      ? `${this.title} ${this.subtitle}`
      : this.title;
  }

  /** Matches scratch/scobility.py Song.__str__ */
  toRankingString(): string {
    const maxPts = this.value_passing + this.value_scoring;
    return `#${this.s_id} ${this.fullTitle} (${this.style} ${this.slot} ${this.meter}) (${this.value_passing} passing + ${this.value_scoring} scoring = ${maxPts} max pts.)`;
  }

  /** Matches scratch/scobility.py view_spice_ranking line format */
  toSpiceLogLine(): string {
    const log2Spice = Math.log2(this.spice ?? 1);
    const head = `${log2Spice.toFixed(3).padStart(5)}🌶`;
    return `${head} ${this.toRankingString().padStart(60)}`;
  }
}

export class Relationship {
  x: Song;
  y: Song;
  e_common: Set<number>;
  relation: number | null;
  strength: number | null;
  use_link: Song | null;

  constructor(x: Song, y: Song) {
    this.x = x;
    this.y = y;
    const xPlayers = new Set(x.scores.keys());
    const yPlayers = new Set(y.scores.keys());
    this.e_common = new Set([...xPlayers].filter((e) => yPlayers.has(e)));
    this.relation = null;
    this.strength = null;
    this.use_link = null;
  }

  get keyTitle(): string {
    return `${this.x.s_id},${this.y.s_id}`;
  }

  compareTitle(): string {
    return `${this.y} vs. ${this.x}`;
  }
}

export class Tournament {
  cfg: CatalogRuntimeConfig;
  songs: Map<number, Song>;
  relationships: Map<string, Relationship>;
  ordering: Map<string, Song[]>;

  monoThreshold = 0.999999;
  minCommonPlayers = 6;
  iterationsMonotonicSort = 10;
  iterationsScobilitySort = 500;
  scobilityWindowLower = 10;
  scobilityWindowUpper = 6;

  relationshipMaxNegLimit: number;
  relationshipMinNegLimit: number;
  relationshipWeightOffset = 0.5;
  relationshipMinCommonPlayers = 6;

  constructor(cfg: CatalogRuntimeConfig) {
    this.cfg = cfg;
    this.songs = new Map();
    this.relationships = new Map();
    this.ordering = new Map();
    this.relationshipMaxNegLimit = cfg.relationshipMaxNegLimit;
    this.relationshipMinNegLimit = cfg.relationshipMinNegLimit;
  }

  static linkThrough(a: Relationship, b: Relationship): Relationship {
    if (a.y.s_id !== b.x.s_id) {
      throw new Error(`Discontinuous link ${a.y.s_id} -> ${b.x.s_id}`);
    }
    const r = new Relationship(a.x, b.y);
    r.e_common = new Set(
      [...a.e_common].filter((e) => b.e_common.has(e))
    );
    r.relation = (a.relation ?? 0) * (b.relation ?? 0);
    const ar = a.relation ?? 0;
    const br = b.relation ?? 0;
  const as = a.strength ?? 0;
  const bs = b.strength ?? 0;
    r.strength = (ar + br) / (ar * as + br * bs);
    r.use_link = a.y;
    return r;
  }
}