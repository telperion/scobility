// scobility v2024.5
// :chili_pepper:

import fetch from "node-fetch";

// Internal spice calculation parameter that also plays a part in
// achievable score predictions
const perfect_offset = 1.003

class ScobilityCoefficients {
  version: number = 2024.5;
  cut_point: number = -1;
  timing_power: number = -1;
  horizon_spice: number = -1;
  horizon_quality: number = -1;
  mild_slope: number = -1;
  hot_slope: number = -1;
  residual: number = -1;
  quality_fit: Function = (s: number) => 0;

  valid() {
    return this.timing_power >= 0;
  }

  targetFromSpice(spice: number) {
    const target = perfect_offset - Math.pow(2, spice - this.quality_fit(spice));
    return target < 0 ? 0 : target > 1 ? 1 : target;
  }

  describeTimingPower() {
    // Both v2023 and v2024 have a meaningful Y-intercept.
    return (
      "timing power:\n" +
      (this.valid() ? this.timing_power.toFixed(2) : "❓") +
      " ✨"
    );
  }

  describeMild() {
    // Need to distinguish between v2023 and v2024 for this coefficient.
    let readableName = "mild sauce:\n";
    if (this.version < 2024) {
      readableName = "spice tolerance:\n";
    }

    return (
      readableName +
      (this.valid() ? this.mild_slope.toFixed(2) : "❓") +
      " ✨/🌶️"
    );
  }

  describeHot() {
    if (this.version < 2024) {
      // v2023 only has one slope coefficient and it's not this one.
      return "(2023 version)";
    }

    return (
      "hot sauce:\n" +
      (this.valid() ? this.hot_slope.toFixed(2) : "❓") +
      " ✨/🌶️"
    );
  }

  describeSpiceHorizon() {
    if (this.version < 2024) {
      // v2023 doesn't have an inflection point.
      return "(2023 version)";
    }

    return (
      "spice horizon:\n(" +
      (this.valid() ? this.horizon_spice.toFixed(2) : "❓") +
      "🌶️, " +
      (this.valid() ? this.horizon_quality.toFixed(2) : "❓") +
      "✨)"
    );
  }

  strategize() {
    if (!this.valid()) {
      return "🌶️🌶️🌶️";
    }

    if (this.version < 2024) {
      if (this.hot_slope < 0) {
        return "Train spice tolerance.";
      } else {
        return "Train precise timing.";
      }
    }

    if (this.mild_slope < 0) {
      if (this.hot_slope < 0) {
        return "Choose spicier charts than you normally play.";
      } else {
        return (
          "Train charts with a spice rating around " +
          this.horizon_spice.toFixed(2) +
          "🌶️."
        );
      }
    } else {
      if (this.hot_slope < 0) {
        return "Train mild precision or spicy tolerance.";
      } else {
        return "Train precise timing on mild charts.";
      }
    }
  }

  explain_horizon() {
    if (!this.valid()) {
      return "🌶️🌶️🌶️";
    }

    if (this.version < 2024) {
      return "🌶️🌶️🌶️";
    } 

    const horizon_selector = (
      ((this.mild_slope < 0) ? 0 : 4) +
      ((this.hot_slope < 0) ? 0 : 2) + 
      ((this.mild_slope < this.hot_slope) ? 0 : 1)
    )
    const horizon_explanations = [
      "levels off",
      "takes a turn for the worse",
      "reaches a minimum",
      "does something REALLY strange",    // (-mild, +hot, mild > hot) can't happen.
      "does something REALLY strange",    // (+mild, -hot, mild < hot) can't happen.
      "reaches a maximum",
      "takes a turn for the better",
      "levels off",
    ]
    return "Your spice tolerance compared to your peers " + horizon_explanations[horizon_selector] + " at " + this.horizon_spice.toFixed(2) + "🌶️, where your predicted score is " + (100 * this.targetFromSpice(this.horizon_spice)).toFixed(2) + "% EX.";
  }
}

class ScobilityStats {
  entrant_id: bigint = BigInt(-1);
  name: string = "";
  tourney_power: number = -1;
  coefs: ScobilityCoefficients = new ScobilityCoefficients();
}

export interface LoadedPlayer {
  entrant_id: bigint;
  name: string;
  scobility_calc_time: Date | string;
}

export interface LoadedChart {
  global_chart_id: bigint;
  catalog_id: bigint;
  chart_id: bigint;
  hash: string;
  title: string;
  subtitle: string;
  artist: string;
  meter: number;
  slot: string;
  style: string | number;
  value: number;
  spice: number;
  spice_calc_time: Date | string;
}

export interface LoadedScore {
  score_id: bigint;
  entrant_id: bigint;
  chart_id: bigint;
  hash: string;
  score: number;
  plays: number;
  last_played: Date | string;
}

export interface ProcessedScore {
  key: React.Key;
  entrant_id: bigint;
  entrant_name: string;
  chart_id: bigint;
  title: string;
  meter: number;
  score: number;
  style: string;
  spice: number;
  value: number;
  quality: number;
  relative_quality: number;
  plays: number;
  last_played: Date;
  current_sp: number;
  current_ep: number;
  current_rp: number;
  contributes_sp: boolean;
  contributes_ep: boolean;
  target_score: number;
  target_sp: number;
  target_ep: number;
  recoverable_sp: number;
  recoverable_ep: number;
  recoverable_rp: number;
}

export interface ScobilityDBResponse<T> {
  status: boolean;
  data: Map<string, T>;
  message: string;
}
function no_db_response<T>(): ScobilityDBResponse<T> {
  return {
    status: false,
    data: new Map<string, T>(),
    message: "Something happened"
  };
}

async function loadSpiceData(catalog: string): Promise<ScobilityDBResponse<LoadedChart>> {
  return fetch(
    `https://scobility.azurewebsites.net/catalog/${catalog}/chart/all/detail/all`
  )
    .then((response) => {
      if (!response.ok) {
        throw new Error(response.statusText);
      }
      return response.json() as Promise<ScobilityDBResponse<LoadedChart>>;
    })
    .catch((error: Error) => {
      console.error("Error loading spice data: ", error);
      return no_db_response<LoadedChart>();
    })
    .then((data: ScobilityDBResponse<LoadedChart>) => {
      data.data = new Map(Object.entries(data.data));
      return data;
    })
    .catch((error: Error) => {
      console.error("Error converting spice data: ", error)
      return no_db_response<LoadedChart>();
    })
}

async function loadPlayerData_test(catalog: string): Promise<ScobilityDBResponse<LoadedPlayer>> {
  return fetch(`https://scobility.azurewebsites.net/catalog/${catalog}/players`)
    .then((response) => {
      if (!response.ok) {
        throw new Error(response.statusText);
      }
      return response.json() as Promise<ScobilityDBResponse<LoadedPlayer>>;
    })
    .catch((error: Error) => {
      console.error("Error loading player data: ", error)
      return no_db_response<LoadedPlayer>();
    })
    .then((data: ScobilityDBResponse<LoadedPlayer>) => {
      data.data = new Map(Object.entries(data.data));
      return data;
    })
    .catch((error: Error) => {
      console.error("Error converting player data: ", error)
      return no_db_response<LoadedPlayer>();
    })
}

async function loadScoreData_test(
  catalog: string,
  player_id: number
): Promise<ScobilityDBResponse<LoadedScore>> {
  return fetch(
    `https://scobility.azurewebsites.net/catalog/${catalog}/score/${player_id}`
  )
    .then((response) => {
      if (!response.ok) {
        throw new Error(response.statusText);
      }
      return response.json() as Promise<ScobilityDBResponse<LoadedScore>>;
    })
    .catch((error: Error) => {
      console.error("Error loading score data: ", error)
      return no_db_response<LoadedScore>();
    })
    .then((data: ScobilityDBResponse<LoadedScore>) => {
      data.data = new Map(Object.entries(data.data));
      return data;
    })
    .catch((error: Error) => {
      console.error("Error converting score data: ", error)
      return no_db_response<LoadedScore>();
    })
}

function transformLoadedScore(
  row: LoadedScore,
  charts: Map<string, LoadedChart>,
  players: Map<string, LoadedPlayer>,
): ProcessedScore | null {
  const chart_info = charts.get(row.hash);
  if (!chart_info) {
    console.error(
      `Couldn't find chart ID matching ${row.hash} in current catalog`
    );
    return null;
  }

  // HACK? my dumbass didn't populate this field correctly in the database
  const true_style = row.chart_id > 400 ? "dance-double" : "dance-single";

  return {
    key: row.chart_id,
    entrant_id: row.entrant_id,
    entrant_name: players.get(row.entrant_id.toString())?.name ?? "",
    chart_id: row.chart_id,
    title:
      { "dance-single": "[S", "dance-double": "[D" }[true_style] +
      chart_info.meter.toString().padStart(2, "0") +
      "] " +
      chart_info.title, // e.g. [S08] I Can't Stop Me
    meter: chart_info.meter,
    score: row.score,
    style: true_style,
    spice: Math.log2(chart_info.spice),
    value: chart_info.value,
    quality: Math.log2(chart_info.spice) - Math.log2(perfect_offset - row.score),
    relative_quality: 0,
    plays: row.plays,
    last_played: typeof(row.last_played) == "string" ? new Date(row.last_played) : row.last_played,
    current_sp: 0,
    current_ep: 0,
    current_rp: 0,
    contributes_sp: false,
    contributes_ep: false,
    target_score: 0,
    target_sp: 0,
    target_ep: 0,
    recoverable_sp: 0,
    recoverable_ep: 0,
    recoverable_rp: 0,
  };
}

function filterScores(
  score_data: ProcessedScore[],
  style_filter: string = "dance-single"
) {
  return score_data
    .filter((row) => row.style == style_filter)
    .sort((a, b) => a.spice - b.spice)
}

// Least squares!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
const _sum = ([...arr]) => arr.reduce((sum, v) => sum + v, 0);
const _sum_squared = ([...arr]) => arr.reduce((sum, v) => sum + v * v, 0);
const _pair = ([...x], [...y]) => x.map((v, i) => [v, y[i]]);

const dumbass_least_squares_components = (
  a: Array<number>,
  b: Array<number>
) => {
  return {
    ones: a.length,
    s: _sum(a),
    s2: _sum_squared(a),
    q: _sum(b),
    sq: _sum(_pair(a, b).map((v) => v[0] * v[1])),
  };
};

const dumbass_least_squares_free = (a: Array<number>, b: Array<number>) => {
  // Plain ol' unanchored least squares best-fit.
  const components = dumbass_least_squares_components(a, b);
  const det = components.ones * components.s2 - components.s * components.s;
  const c1 =
    (-components.s * components.q + components.ones * components.sq) / det;
  const c0 =
    (components.s2 * components.q - components.s * components.sq) / det;
  const residual = _sum_squared(a.map((v, i) => b[i] - (c1 * v + c0)));
  return {
    c0: c0,
    c1: c1,
    residual: residual,
  };
};

const dumbass_least_squares_with_cut_point = (
  a: Array<number>,
  b: Array<number>,
  anchor: number
): ScobilityCoefficients => {
  // Plain' ol unanchored least squares best-fit
  // (but there's two of them!)
  const a_l = a.slice(0, anchor);
  const a_r = a.slice(anchor);
  const b_l = b.slice(0, anchor);
  const b_r = b.slice(anchor);
  const lsq_l = dumbass_least_squares_free(a_l, b_l);
  const lsq_r = dumbass_least_squares_free(a_r, b_r);
  const horizon_spice = (lsq_r.c1 - lsq_l.c1) / (lsq_l.c0 - lsq_r.c0);
  const horizon_quality = lsq_l.c1 * horizon_spice + lsq_l.c0;
  const timing_power = horizon_spice > 0 ? lsq_l.c0 : lsq_r.c0;
  return Object.assign(new ScobilityCoefficients(), {
    cut_point: anchor,
    timing_power: timing_power,
    horizon_spice: horizon_spice,
    horizon_quality: horizon_quality,
    mild_slope: lsq_l.c1,
    hot_slope: lsq_r.c1,
    residual: lsq_l.residual + lsq_r.residual,
  });
};

const dumbass_least_squares_anchored = (
  a: Array<number>,
  b: Array<number>,
  anchor: number
) => {
  // Solve a special condition of least-squares where a set of points is
  // broken up into two lines, and the X coordinate of the intersection is
  // fixed. This makes the independent variables the slopes of the two lines
  // and the Y coordinate of the intersection.
  // Here we're choosing the X coordinate as a[anchor].
  const a_offset = a.map((v) => v - a[anchor]);
  const a_l = a_offset.slice(0, anchor);
  const a_r = a_offset.slice(anchor);
  const b_l = b.slice(0, anchor);
  const b_r = b.slice(anchor);

  // Borrow some of the calculations from the naive least squares method.
  const comp_l = dumbass_least_squares_components(a_l, b_l);
  const comp_r = dumbass_least_squares_components(a_r, b_r);
  const ones = a.length;
  const q = _sum(b);

  // Special 3x3 symmetric matrix inversion
  const m11 = ones * comp_r.s2 - comp_r.s * comp_r.s;
  const m12 = comp_l.s * comp_r.s;
  const m13 = -comp_l.s * comp_r.s2;
  const m22 = ones * comp_l.s2 - comp_l.s * comp_l.s;
  const m23 = -comp_l.s2 * comp_r.s;
  const m33 = comp_l.s2 * comp_r.s2;
  const det =
    ones * comp_r.s2 * comp_l.s2 -
    comp_r.s * comp_r.s * comp_l.s2 -
    comp_l.s * comp_l.s * comp_r.s2;
  // Equivalent formulation of determinant
  // const det = m11*comp_l.s2 + m13*comp_l.s

  // Evaluate the two slopes and the X coordinate at the anchor.
  const c1_l = (m11 * comp_l.sq + m12 * comp_r.sq + m13 * q) / det;
  const c1_r = (m12 * comp_l.sq + m22 * comp_r.sq + m23 * q) / det;
  const c0 = (m13 * comp_l.sq + m23 * comp_r.sq + m33 * q) / det;

  const residual = _sum_squared(
    a.map((v, i) => {
      if (i < anchor) {
        return b[i] - (c1_l * (v - a[anchor]) + c0);
      } else {
        return b[i] - (c1_r * (v - a[anchor]) + c0);
      }
    })
  );
  return {
    c0: c0,
    c1_l: c1_l,
    c1_r: c1_r,
    residual: residual,
  };
};

// Scobilitous piecewise least squares!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// Unga bunga!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
const spice_horizon_fit = (a: Array<number>, b: Array<number>) => {
  // Requires independent variable to be sorted.

  let best_fit_so_far = new ScobilityCoefficients();

  // Don't unga bunga too close to the edges of the spice spread.
  const horizon_centering = Math.sqrt(a.length);

  // Test the fitness of the piecewise linear approximation between each
  // pair of spice values.
  // Even in scobility we can have little a DP (dynamic programming),
  // as a treat
  for (let i in Array(a.length)
    .fill(0)
    .map((e, i) => i)) {
    const j = parseInt(i);

    if (j < horizon_centering || j >= a.length - horizon_centering) {
      continue;
    }

    // Try fitting an unanchored dual least squares first.
    // If the intersection lands between this pair of spice values, it
    // automatically wins the optimization for this step of the DP algorithm.
    const best_fit_here_naive = dumbass_least_squares_with_cut_point(a, b, j)
    let best_fit_here = best_fit_here_naive
    if (best_fit_here.horizon_spice < a[j] || best_fit_here.horizon_spice > a[j+1]) {
      // The intersection (a.k.a. spice horizon) didn't land between this
      // pair of spice values, so it can't satisfy the optimization constraint.
      // Let's evaluate what the best fits are on the boundaries and see which
      // wins among those two.
      best_fit_here = new ScobilityCoefficients();
      const best_fit_here_l = dumbass_least_squares_anchored(a, b, j);
      const best_fit_here_r = dumbass_least_squares_anchored(a, b, j + 1);
      if (best_fit_here_l.residual < best_fit_here_r.residual) {
        best_fit_here = Object.assign(new ScobilityCoefficients(), {
          cut_point: j,
          horizon_spice: a[j],
          horizon_quality: best_fit_here_l.c0,
          mild_slope: best_fit_here_l.c1_l,
          hot_slope: best_fit_here_l.c1_r,
          timing_power: best_fit_here_l.c0 - best_fit_here_l.c1_l * a[j],
          residual: best_fit_here_l.residual,
        });
      } else {
        best_fit_here = Object.assign(new ScobilityCoefficients(), {
          cut_point: j,
          horizon_spice: a[j + 1],
          horizon_quality: best_fit_here_r.c0,
          mild_slope: best_fit_here_r.c1_l,
          hot_slope: best_fit_here_r.c1_r,
          timing_power: best_fit_here_l.c0 - best_fit_here_l.c1_l * a[j],
          residual: best_fit_here_r.residual,
        });
      }
    }

    if (
      best_fit_so_far.cut_point < 0 ||
      best_fit_here.residual < best_fit_so_far.residual
    ) {
      best_fit_so_far = best_fit_here;
    }
  }
  return best_fit_so_far;
};

// SP calculation functions for ITL2023/ITL2024
const sp_log_base = 1.1032889141348;
const sp_pow_base = 61;
const sp_inflect = 50;
const expct_to_sppct = (expct: number) => {
  // EX % to SP %
  const v_lo = expct < 50 ? expct : 50;
  const v_hi = expct > 50 ? expct : 50;

  return (
    Math.log(v_lo + 1) / Math.log(sp_log_base) +
    Math.pow(sp_pow_base, (v_hi - sp_inflect) / (100 - sp_inflect)) -
    1
  );
};
const sppct_to_expct = (sppct: number) => {
  // SP % to EX %
  const piecewise_border = Math.log(sp_inflect + 1) / Math.log(sp_log_base) - 1;
  if (sppct < piecewise_border) {
    return Math.pow(sp_log_base, sppct) - 1;
  } else {
    return (
      ((100 - sp_inflect) * Math.log(sppct - piecewise_border)) /
        Math.log(sp_pow_base) +
      sp_inflect
    );
  }
};

// EP calculation function for ITL2024
const ep_curve_cutoff = 85.0;
const expct_curve = (expct: number) => {
  // EX % to EP
  return (
    (Math.pow(
      100,
      (expct < ep_curve_cutoff ? 0 : expct - ep_curve_cutoff) / (100.0 - ep_curve_cutoff)
    ) -
      1) *
    (1000.0 / 99.0)
  );
};

const calculateScobility = (
  score_data: ProcessedScore[],
  player_data: LoadedPlayer | undefined,
  fit_algorithm: boolean = true
): ScobilityStats => {
  // List out spice and quality for each played chart.
  const spice_values = score_data.map((row) => row.spice);
  const quality_values = score_data.map((row) => row.quality);

  // Tourney power rating (this is kinda spitballed I might adjust later)
  const tourney_power =
    0.5 * Math.log2(_sum(quality_values.map((v) => Math.pow(2, v * 2))));

  if (fit_algorithm) {
    // scobility v2024
    const coefs = spice_horizon_fit(spice_values, quality_values);
    coefs.quality_fit = (s: number) => {
      if (s <= coefs.horizon_spice) {
        return coefs.mild_slope * (s - coefs.horizon_spice) + coefs.horizon_quality;
      } else {
        return coefs.hot_slope * (s - coefs.horizon_spice) + coefs.horizon_quality;
      }
    };
    return {
      entrant_id: player_data?.entrant_id ?? BigInt(-1),
      name: player_data?.name ?? "[n/a]",
      tourney_power: tourney_power,
      coefs: coefs,
    };
  } else {
    // scobility v2023 can still be calculated for comparison :)
    const coefs_line = dumbass_least_squares_free(spice_values, quality_values);
    const coefs = Object.assign(new ScobilityCoefficients(), {
      version: 2023,
      cut_point: 0,
      timing_power: coefs_line.c0,
      horizon_spice: 0,
      horizon_quality: coefs_line.c0,
      mild_slope: coefs_line.c1,
      hot_slope: coefs_line.c1,
      residual: coefs_line.residual,
      quality_fit: (s: number) => {
        if (s <= coefs.horizon_spice) {
          return coefs.mild_slope * (s - coefs.horizon_spice) + coefs.horizon_quality;
        } else {
          return coefs.hot_slope * (s - coefs.horizon_spice) + coefs.horizon_quality;
        }
      }
    });
    return {
      entrant_id: player_data?.entrant_id ?? BigInt(-1),
      name: player_data?.name ?? "[n/a]",
      tourney_power: tourney_power,
      coefs: coefs,
    };
  }
};

const sp_hand_size_map = new Map([
  ["dance-single", 75],
  ["dance-double", 50],
]);
const ep_hand_size_map = new Map([
  [7, 1],
  [8, 2],
  [9, 3],
  [10, 4],
  [11, 4],
  [12, 3],
  [13, 2],
  [14, 1],
]);

function hydrateProcessedScores(
  score_data: ProcessedScore[],
  scobility: ScobilityStats,
  style_filter: string = "dance-single"
) {
  // Start by calculating target score and current/achievable SP/EP.
  for (let row of score_data) {
    // TODO: double-check this math
    row.relative_quality = row.quality - scobility.coefs.quality_fit(row.spice);
    row.target_score = scobility.coefs.targetFromSpice(row.spice);
    row.target_sp =
      row.target_score > 0.99999
        ? row.value
        : Math.floor(
            (expct_to_sppct(row.target_score * 100) * row.value) / 100
          );
    row.current_sp =
      row.score > 0.99999
        ? row.value
        : Math.floor((expct_to_sppct(row.score * 100) * row.value) / 100);
    row.target_ep =
      row.target_score > 0.99999
        ? 1000
        : Math.floor(expct_curve(row.target_score * 100));
    row.current_ep =
      row.score > 0.99999 ? 1000 : Math.floor(expct_curve(row.score * 100));
  }

  // Understand which charts contribute SP and EP to the player's ranking points.
  // If we're in the ITL2024 website ecosystem, probably easier to fill
  // sp_contenders/ep_contenders and sp_cutoff/ep_cutoff directly
  // rather than re-deriving here.

  // SP is easy - pull the top slice of current SP values.
  const sp_hand_size = sp_hand_size_map.get(style_filter) || 0;
  const ranked_by_current_sp = score_data.toSorted(
    (a, b) => b.current_sp - a.current_sp
  ); // descending order
  const sp_cutoff =
    ranked_by_current_sp.length < sp_hand_size
      ? 0
      : ranked_by_current_sp[sp_hand_size - 1].current_sp; // has to replace something
  const sp_contenders = ranked_by_current_sp
    .slice(0, sp_hand_size)
    .map((row) => row.key);
  // console.log(ranked_by_current_sp);
  // console.log(sp_contenders);
  // console.log(sp_cutoff);

  // EP is a little more difficult - the chart's meter has an effect on whether the chart has the opportunity to contribute or not.
  const ranked_by_current_ep = score_data.toSorted(
    (a, b) => b.current_ep - a.current_ep
  ); // descending order
  const ep_cutoff = Object.fromEntries(
    Array.from(ep_hand_size_map.keys()).map((key) => [key, 1000])
  );
  const ep_contenders = Object.fromEntries(
    Array.from(ep_hand_size_map.keys()).map((key) => [key, new Array<ProcessedScore>()])
  );
  for (let row of ranked_by_current_ep) {
    if (ep_hand_size_map.has(row.meter)) {
      const ep_hand_size = ep_hand_size_map.get(row.meter) || 0;
      if (ep_contenders[row.meter].length < ep_hand_size) {
        ep_contenders[row.meter].push(row);
        ep_cutoff[row.meter] = Math.min(ep_cutoff[row.meter], row.current_ep);
      }
    }
  }
  // console.log(ranked_by_current_ep);
  // console.log(ep_contenders);
  // console.log(ep_cutoff);

  let total_sp = 0
  let total_ep = 0
  let total_rp = 0
  let total_tp = 0
  for (let row of score_data) {
    // Does this score currently contribute SP or EP to our total RP?
    row.contributes_sp = sp_contenders.includes(row.key);
    row.contributes_ep =
      row.meter in ep_contenders &&
      ep_contenders[row.meter].map((r) => r.key).includes(row.key);
    row.current_rp =
      (row.contributes_sp ? row.current_sp : 0) +
      (row.contributes_ep ? row.current_ep : 0);

    // Would raising this score to scobility's prediction contribute (more) SP or EP to our total RP?
    row.recoverable_sp = Math.max(
      row.target_sp - Math.max(row.current_sp, sp_cutoff),
      0
    );
    row.recoverable_ep = ep_hand_size_map.has(row.meter)
      ? Math.max(
          row.target_ep - Math.max(row.current_ep, ep_cutoff[row.meter]),
          0
        )
      : 0;
    row.recoverable_rp = row.recoverable_sp + row.recoverable_ep;

    // "Checksums"
    total_sp += row.contributes_sp ? row.current_sp : 0
    total_ep += row.contributes_ep ? row.current_ep : 0
    total_rp += row.current_rp
    total_tp += row.current_sp + (row.contributes_ep ? row.current_ep : 0) // TODO: why this
  }
  console.log(`Total SP: ${total_sp} | Total EP: ${total_ep} | Total RP: ${total_rp} | Total TP: ${total_tp}`)
  return score_data;
}

export {
  loadSpiceData,
  loadPlayerData_test,
  loadScoreData_test,
  transformLoadedScore,
  filterScores,
  calculateScobility,
  hydrateProcessedScores,
  ScobilityCoefficients,
  ScobilityStats,
};
