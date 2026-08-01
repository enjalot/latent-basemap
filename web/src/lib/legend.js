import { fmt } from "./format.js";

// Collision-free decade ticks for the density ramp legend. Place decade ticks
// left->right, dropping any whose LABEL box would overlap the previous tick or
// the right-anchored cap label, so ticks never mash into an illegible run.
// Ported from viewer.js updateLegend (v1 item 4). Pure -> unit tested.
//
//   cap  : max count in the drawn grid (>=1)
//   barW : rendered pixel width of the ramp bar
// returns [{ v, pos }] with pos in [0,1]; the last entry is always the cap.
export function computeTicks(cap, barW) {
  cap = Math.max(1, Math.floor(cap));
  const denom = Math.log(cap + 1);
  const CHAR_PX = 6.2;
  const PAD = 5;
  const labW = (v) => Math.max(8, String(fmt(v)).length * CHAR_PX);
  const capLeft = barW - labW(cap); // cap is right-anchored at 100%
  const kept = [];
  let lastRight = -Infinity;
  for (let v = 1; v < cap; v *= 10) {
    const pos = Math.log(v + 1) / denom;
    const cx = pos * barW;
    const half = labW(v) / 2;
    const left = cx - half;
    const right = cx + half;
    if (right + PAD > capLeft) continue; // would touch the cap label
    if (left < lastRight + PAD) continue; // would touch the previous label
    kept.push({ v, pos });
    lastRight = right;
  }
  kept.push({ v: cap, pos: 1 }); // always show the max
  return kept;
}
