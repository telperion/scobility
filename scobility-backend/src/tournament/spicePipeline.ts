import {
  orderRefineBySpice,
  orderRefineMonotonic,
  orderSongsInitial,
} from "./ordering.js";
import { calcRelationships, setupRelationships } from "./tournament.js";
import type { Tournament } from "./models.js";

export function runSpicePipeline(
  t: Tournament,
  styles: string[] = ["Single", "Double"],
  verbal = true
): void {
  console.log("=== Setting up relationships...");
  setupRelationships(t);
  console.log("=== Calculating relationships...");
  calcRelationships(t, verbal);

  for (const style of styles) {
    console.log(`======= ${style} =========================================================`);
    console.log("=== Setting up closest-neighbor initial order...");
    orderSongsInitial(t, style, false);
    console.log("=== Bubbling out non-monotonicities...");
    orderRefineMonotonic(t, style, false);
    console.log("=== Refining spice ranking using neighborhood influence...");
    orderRefineBySpice(t, style, verbal);
    console.log("=== Spice ranking calculation complete!");
  }
}