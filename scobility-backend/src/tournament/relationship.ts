import { lstsq } from "../math/lstsq.js";
import { Relationship, Tournament } from "./models.js";

export function calcRelationship(
  r: Relationship,
  t: Tournament,
  scoreScalar: number
): void {
  const xPlayers = new Set(r.x.scores.keys());
  const yPlayers = new Set(r.y.scores.keys());
  r.e_common = new Set([...xPlayers].filter((e) => yPlayers.has(e)));

  if (r.e_common.size < t.relationshipMinCommonPlayers) {
    throw new Error(
      `Not enough players to relate ${r.compareTitle()} (${r.e_common.size} players, need ${t.relationshipMinCommonPlayers})`
    );
  }

  const eList = [...r.e_common];
  const xScores = eList.map((e) => r.x.scores.get(e)!.value);
  const yScores = eList.map((e) => r.y.scores.get(e)!.value);

  const screenMaxNeg = xScores.map((_, i) => Math.max(xScores[i], yScores[i]));
  const screenMinNeg = xScores.map((_, i) => Math.min(xScores[i], yScores[i]));

  const lfit1: number[] = [];
  for (let i = 0; i < xScores.length; i++) {
    if (
      screenMaxNeg[i] < t.relationshipMaxNegLimit &&
      screenMinNeg[i] > t.relationshipMinNegLimit
    ) {
      lfit1.push(i);
    }
  }

  if (lfit1.length < t.relationshipMinCommonPlayers) {
    const minScoreCheck = (1 - t.relationshipMaxNegLimit) / scoreScalar;
    const maxScoreCheck = (1 - t.relationshipMinNegLimit) / scoreScalar;
    throw new Error(
      `Not enough scores between ${minScoreCheck.toFixed(4)} and ${maxScoreCheck.toFixed(4)} to relate ${r.compareTitle()} (${r.e_common.size} players, need ${t.relationshipMinCommonPlayers})`
    );
  }

  const a = new Float64Array(lfit1.map((i) => xScores[i]));
  const b = new Float64Array(lfit1.map((i) => yScores[i]));
  const w = new Float64Array(
    lfit1.map(
      (_, j) => 1 - 1 / Math.pow(a[j] + t.relationshipWeightOffset, 2)
    )
  );

  const coefs = lstsq(a, b, w);
  r.relation = coefs[0];

  let resid = 0;
  for (let j = 0; j < a.length; j++) {
    const pred = coefs[0] * a[j];
    resid += (b[j] - pred) ** 2;
  }
  r.strength = Math.sqrt(a.length / resid);
}