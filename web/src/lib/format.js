// Number formatting helpers shared by the header, legend, and tooltips.

export const fmt = (n) => Number(n).toLocaleString("en-US");

// Round a numeric metric to <=4 significant figures for display. Non-numeric
// values pass through untouched. (Ported verbatim from viewer.js sig4.)
export function sig4(v) {
  const n = Number(v);
  if (!isFinite(n)) return String(v);
  return String(Number(n.toPrecision(4)));
}

// "30,000 of 24,948,663 rows (sampled)" when sampled_of is present, else
// "24,948,663 rows". Addendum v2 item 2 / layer schema `sampled_of`.
export function rowsPhrase(rows, sampledOf) {
  if (sampledOf != null && Number(sampledOf) > 0) {
    return `${fmt(rows)} of ${fmt(sampledOf)} rows (sampled)`;
  }
  return `${fmt(rows)} rows`;
}
