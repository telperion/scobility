import { calcRelationship } from "./relationship.js";
import { Relationship, Song, Tournament } from "./models.js";

export function setupRelationships(t: Tournament): void {
  const songIds = [...t.songs.keys()];
  for (let i = 0; i < songIds.length; i++) {
    const x = songIds[i];
    for (let j = 0; j < i; j++) {
      const y = songIds[j];
      const sx = t.songs.get(x)!;
      const sy = t.songs.get(y)!;
      if (sx.style === sy.style) {
        const r1 = new Relationship(sx, sy);
        t.relationships.set(r1.keyTitle, r1);
        const r2 = new Relationship(sy, sx);
        t.relationships.set(r2.keyTitle, r2);
      }
    }
    if ((i + 1) % 50 === 0 || i === songIds.length - 1) {
      console.log(`::: Relationships set up for ${i + 1} songs`);
    }
  }
}

export function calcRelationships(t: Tournament, verbose = false): void {
  let successes = 0;
  const relList = [...t.relationships.values()];
  for (let i = 0; i < relList.length; i++) {
    const r = relList[i];
    try {
      calcRelationship(r, t, t.cfg.scoreScalar);
      if (verbose) {
        console.log(`--- ${i} #${r.y.s_id} vs #${r.x.s_id}: ${r.relation}`);
      }
      successes++;
    } catch (e) {
      if (verbose) {
        console.log(`!!! ${i} ${(e as Error).message}`);
      }
    }
  }
  console.log(
    `Completed relationship calculations (${successes} out of ${relList.length} strong pairs)`
  );
}

export function relationshipLookup(
  t: Tournament,
  x: Song,
  y: Song,
  allowLink = false
): Relationship {
  if (x.s_id === y.s_id) {
    const r = new Relationship(x, x);
    r.relation = 1.0;
    r.strength = r.e_common.size;
    return r;
  }

  if (x.style !== y.style) {
    throw new Error(
      `Not comparing ${x.style} (#${x.s_id}) and ${y.style} (#${y.s_id}) charts`
    );
  }

  const rKey = new Relationship(x, y);
  const direct = t.relationships.get(rKey.keyTitle);
  if (direct) {
    return direct;
  }

  const rKeyRev = new Relationship(y, x);
  const storedRev = t.relationships.get(rKeyRev.keyTitle);
  if (storedRev) {
    if (storedRev.relation === null) {
      return rKeyRev;
    }
    rKeyRev.relation = 1 / storedRev.relation;
    rKeyRev.e_common = storedRev.e_common;
    rKeyRev.strength = storedRev.strength;
    return rKeyRev;
  }

  if (!allowLink) {
    throw new Error(`No relationship for ${rKey.compareTitle()}`);
  }

  const potentialLinks: Relationship[] = [];
  for (const z of t.songs.values()) {
    try {
      const xz = relationshipLookup(t, x, z, false);
      const zy = relationshipLookup(t, z, y, false);
      const xzStored = t.relationships.get(xz.keyTitle);
      const zyStored = t.relationships.get(zy.keyTitle);
      if (xzStored && zyStored) {
        potentialLinks.push(Tournament.linkThrough(xzStored, zyStored));
      }
    } catch {
      // One of the links doesn't exist.
    }
  }
  if (potentialLinks.length < 1) {
    throw new Error(
      `No relationship for ${rKey.compareTitle()}, even with linking!`
    );
  }
  potentialLinks.sort((a, b) => (a.strength ?? 0) - (b.strength ?? 0));
  return potentialLinks[potentialLinks.length - 1];
}