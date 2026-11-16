import { relationshipLookup } from "./tournament.js";
import { Song, Tournament } from "./models.js";

export function orderSongsInitial(
  t: Tournament,
  style = "Single",
  verbal = false
): void {
  const songSimilarity = [...t.relationships.values()].filter(
    (r) =>
      r.relation !== null &&
      r.e_common.size >= t.minCommonPlayers &&
      r.x.style === style &&
      r.y.style === style
  );
  songSimilarity.sort((a, b) => (a.relation ?? 0) - (b.relation ?? 0));

  const buildOrdering: Song[] = [];

  for (let iPair = 0; iPair < songSimilarity.length; iPair++) {
    const r = songSimilarity[songSimilarity.length - 1 - iPair];
    if ((r.relation ?? 0) < 1) {
      break;
    }

    if (buildOrdering.length === 0) {
      buildOrdering.push(r.x, r.y);
      continue;
    }

    if (!buildOrdering.includes(r.x)) {
      let iEnd =
        buildOrdering.indexOf(r.y) >= 0
          ? buildOrdering.indexOf(r.y)
          : buildOrdering.length - 1;
      let i = iEnd - 1;
      for (; i >= 0; i--) {
        const s = buildOrdering[i];
        const rComp = relationshipLookup(t, r.x, s);
        if (
          rComp.relation !== null &&
          rComp.relation < t.monoThreshold
        ) {
          break;
        }
      }
      buildOrdering.splice(i + 1, 0, r.x);
    }

    if (!buildOrdering.includes(r.y)) {
      const iStart =
        buildOrdering.indexOf(r.x) >= 0 ? buildOrdering.indexOf(r.x) : 0;
      let i = iStart;
      for (; i < buildOrdering.length; i++) {
        const s = buildOrdering[i];
        const rComp = relationshipLookup(t, s, r.y);
        if (
          rComp.relation !== null &&
          rComp.relation < t.monoThreshold
        ) {
          break;
        }
      }
      buildOrdering.splice(i, 0, r.y);
    }
  }

  t.ordering.set(style, buildOrdering);
  if (verbal) {
    console.log(`Initial ordering for ${style}: ${buildOrdering.length} songs`);
  }
}

export function orderRefineMonotonic(
  t: Tournament,
  style = "Single",
  verbal = false
): void {
  let refinedOrdering = [...(t.ordering.get(style) ?? [])];
  for (let a = 0; a < t.iterationsMonotonicSort; a++) {
    const prevOrdering = [...refinedOrdering];
    for (let i = 0; i < refinedOrdering.length - 1; i++) {
      const x = refinedOrdering[i];
      const y = refinedOrdering[i + 1];
      const rFwd = relationshipLookup(t, x, y, true);
      const rRev = relationshipLookup(t, y, x, true);
      if (
        rFwd.relation &&
        rFwd.relation < t.monoThreshold &&
        rRev.relation &&
        rRev.relation > rFwd.relation
      ) {
        refinedOrdering = [
          ...refinedOrdering.slice(0, i),
          y,
          x,
          ...refinedOrdering.slice(i + 2),
        ];
      }
    }
    if (verbal) {
      for (let i = 0; i < prevOrdering.length; i++) {
        if (prevOrdering[i].s_id !== refinedOrdering[i].s_id) {
          console.log(`#${i}: ${prevOrdering[i].s_id} -> ${refinedOrdering[i].s_id}`);
        }
      }
    }
  }
  t.ordering.set(style, refinedOrdering);
}

export function orderRefineBySpice(
  t: Tournament,
  style = "Single",
  verbal = false
): number[] {
  const spiceList: number[] = [];
  let refinedOrdering = [...(t.ordering.get(style) ?? [])];

  for (let i = 0; i < refinedOrdering.length; i++) {
    if (i === 0) {
      spiceList.push(1);
      continue;
    }
    let iNearest = i - 1;
    let r = relationshipLookup(
      t,
      refinedOrdering[iNearest],
      refinedOrdering[i]
    );
    while (r.relation === null && iNearest > 0) {
      iNearest -= 1;
      r = relationshipLookup(
        t,
        refinedOrdering[iNearest],
        refinedOrdering[i]
      );
    }
    if (r.relation === null) {
      spiceList.push(spiceList[spiceList.length - 1]);
    } else {
      spiceList.push(spiceList[iNearest] * r.relation);
    }
  }

  const convergence: number[] = [Math.log2(spiceList[spiceList.length - 1])];

  for (let k = 0; k < t.iterationsScobilitySort; k++) {
    const spicePrev = [...spiceList];
    for (let i = 0; i < refinedOrdering.length; i++) {
      const lo = Math.max(0, i - t.scobilityWindowLower);
      const hi = Math.min(
        i + t.scobilityWindowUpper + 1,
        refinedOrdering.length
      );
      const window = refinedOrdering.slice(lo, hi);
      const nearbySpice = spicePrev.slice(lo, hi);
      const relWindow = window.map((w) =>
        relationshipLookup(t, w, refinedOrdering[i])
      );
      const nearbyRelation = relWindow.map((rel) => rel.relation);
      const nearbyStrength = relWindow.map((rel) => rel.strength);
      const contrib: number[] = [];
      const strengths: number[] = [];
      for (let j = 0; j < nearbyRelation.length; j++) {
        const rel = nearbyRelation[j];
        const str = nearbyStrength[j];
        const sp = nearbySpice[j];
        if (str !== null && rel !== null) {
          contrib.push(rel * str * sp);
          strengths.push(str);
        }
      }
      const influence = contrib.reduce((a, b) => a + b, 0) / strengths.reduce((a, b) => a + b, 0);
      spiceList[i] = influence;
    }

    const spiceSort = spiceList.map((s, idx) => [s, refinedOrdering[idx]] as const);
    spiceSort.sort((a, b) => a[0] - b[0]);
    const minSpice = spiceSort[0][0];
    for (let i = 0; i < spiceSort.length; i++) {
      spiceList[i] = spiceSort[i][0] / minSpice;
      refinedOrdering[i] = spiceSort[i][1];
    }

    convergence.push(Math.log2(spiceList[spiceList.length - 1]));
    if (verbal && (k < 5 || k % 50 === 49)) {
      console.log(
        `>>> Iteration ${k + 1}: max spice = ${convergence[convergence.length - 1].toFixed(6)}`
      );
    }

    for (let i = 0; i < refinedOrdering.length; i++) {
      refinedOrdering[i].spice = spiceList[i];
    }
  }

  t.ordering.set(style, refinedOrdering);
  return convergence;
}