/** Colour ramps. YlGnBu is the combined-density look used on the research pages. */

export const YLGNBU: [number, number, number][] = [
  [255, 255, 217],
  [237, 248, 177],
  [199, 233, 180],
  [127, 205, 187],
  [65, 182, 196],
  [29, 145, 192],
  [34, 94, 168],
  [37, 52, 148],
  [8, 29, 88],
];

/** Fallback corpus colours when the manifest doesn't carry them. */
export const CORPUS_FALLBACK = [
  "#3987e5",
  "#eb6834",
  "#0ca30c",
  "#a05fd3",
  "#fab219",
  "#00a5a5",
  "#d03b3b",
  "#7a7a7a",
];

export function hexToRgb(hex: string): [number, number, number] {
  const rgb = /rgba?\(\s*([\d.]+)[\s,]+([\d.]+)[\s,]+([\d.]+)/.exec(hex);
  if (rgb) return [Number(rgb[1]), Number(rgb[2]), Number(rgb[3])];
  const h = hex.replace("#", "");
  const v =
    h.length === 3
      ? h
          .split("")
          .map((c) => c + c)
          .join("")
      : h;
  const n = parseInt(v, 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

export function rampCss(): string {
  return `linear-gradient(to right, ${YLGNBU.map(
    ([r, g, b]) => `rgb(${r},${g},${b})`,
  ).join(", ")})`;
}
