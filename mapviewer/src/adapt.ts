/**
 * adapt.ts — normalise a pack manifest into the viewer's `Manifest` shape.
 *
 * Two producers exist:
 *   `basemap-map-pack-v1`  — mapviewer/fixtures/make_fixture_pack.py (this repo)
 *   `pack_format_version: "1"` — experiments/mappack/map_pack.py (the real builder)
 *
 * They agree on every byte-level format that matters (256² u32 planes y-down,
 * 8 B point records with corpus in the top 4 bits, 9 B LOD records, u64 index
 * files, "{z}/{x}_{y}" tile naming) and differ only in how the manifest names
 * things — plus two units:
 *   * real `points/tile_index.u64` and `lod.min_zoom_offsets` are BYTE offsets,
 *     the fixture's are record offsets. Normalised here via `unit`.
 *   * real levels carry `png_log_peak` = log1p(max bin count); the viewer wants
 *     the raw count.
 */

import type { DensityLevel, Manifest } from "./types";
import { fetchJSON } from "./net";
import { CORPUS_FALLBACK } from "./palette";

interface RealLevelIndex {
  z: number;
  tiles_per_side: number;
  png_log_peak?: number;
  tiles: Record<string, { n: number; corpora: number[] }>;
}

/** "fineweb-edu-sample-10BT-chunked-120-all-MiniLM-L6-v2" -> "FineWeb-Edu" */
function prettyCorpus(name: string): string {
  const known: [RegExp, string][] = [
    [/^fineweb-edu/i, "FineWeb-Edu"],
    [/^fineweb2?/i, "FineWeb"],
    [/^redpajama/i, "RedPajama-V2"],
    [/^pile/i, "The Pile"],
    [/^starcoder/i, "StarCoder"],
    [/^wikipedia/i, "Wikipedia"],
  ];
  for (const [re, label] of known) if (re.test(name)) return label;
  return name.split("-chunked-")[0].split("-sample-")[0];
}

function isRealPack(raw: Record<string, unknown>): boolean {
  return raw.pack_format_version !== undefined && raw.schema === undefined;
}

export async function loadManifest(base: string): Promise<Manifest> {
  const raw = await fetchJSON<Record<string, unknown>>(base + "manifest.json");
  if (!isRealPack(raw)) return raw as unknown as Manifest;
  return adaptReal(base, raw);
}

async function adaptReal(
  base: string,
  raw: Record<string, unknown>,
): Promise<Manifest> {
  const tiles = (raw.tiles ?? {}) as Record<string, unknown>;
  const frame = (raw.frame ?? {}) as Record<string, unknown>;
  const extentArr = (frame.extent ?? frame.raw_extent ?? [0, 1, 0, 1]) as number[];
  const tileSize = (tiles.tile_bins as number) ?? 256;
  const zmax = (tiles.max_zoom as number) ?? 0;

  const codes = (raw.corpus_codes ?? {}) as Record<string, string>;
  const corpora = Object.entries(codes)
    .map(([k, v], i) => ({
      code: Number(k),
      name: v,
      label: prettyCorpus(v),
      color: CORPUS_FALLBACK[i % CORPUS_FALLBACK.length],
    }))
    .sort((a, b) => a.code - b.code);

  // Per-level tile inventory: the real builder omits empty tiles and writes a
  // density/z{z}/index.json listing what exists. Fetch them so we never 404.
  const rawLevels = (tiles.levels ?? []) as Record<string, unknown>[];
  const levels: DensityLevel[] = await Promise.all(
    rawLevels.map(async (l) => {
      const z = l.z as number;
      const level: DensityLevel = {
        z,
        tiles: (l.tiles_per_side as number) ?? 1 << z,
        bins: (l.bins_per_side as number) ?? tileSize * (1 << z),
        max_count:
          l.png_log_peak !== undefined
            ? Math.round(Math.expm1(l.png_log_peak as number))
            : undefined,
      };
      try {
        const idx = await fetchJSON<RealLevelIndex>(`${base}density/z${z}/index.json`);
        level.tile_list = Object.keys(idx.tiles);
        level.tile_corpora = Object.fromEntries(
          Object.entries(idx.tiles).map(([k, v]) => [k, v.corpora]),
        );
      } catch {
        /* dense pack, or no index — fall back to "all tiles exist" */
      }
      return level;
    }),
  );

  const pts = (raw.points ?? {}) as Record<string, unknown>;
  const lod = (raw.lod ?? {}) as Record<string, unknown>;
  const binsRaw = (raw.bins ?? {}) as Record<string, unknown>;
  const textRaw = (raw.text ?? {}) as Record<string, unknown>;
  const nPoints = (raw.n_points as number) ?? 0;
  const binLevels = binsRaw.z !== undefined ? [binsRaw.z as number] : [];

  return {
    schema: "basemap-map-pack-v1",
    map_id: raw.map_id as string,
    title: (raw.map_id as string) ?? "map",
    description: raw.substrate_key as string | undefined,
    synthetic: false,
    N: nPoints,
    extent: {
      xmin: extentArr[0],
      xmax: extentArr[1],
      ymin: extentArr[2],
      ymax: extentArr[3],
    },
    tile_size: tileSize,
    zoom: { min: 0, max: zmax },
    corpora,
    quantization: { xy_bits: 16, xy_max: 65535, id_bits: 28, corpus_bits: 4, y_down: true },
    density: {
      pattern: "density/z{z}/{x}_{y}.{corpus}.u32",
      png: "density/z{z}/{x}_{y}.png",
      plane_dtype: "u32",
      plane_bytes: tileSize * tileSize * 4,
      levels,
    },
    points: {
      lod: lod.n_points
        ? {
            path: "points/lod.bin",
            record_bytes: (lod.record_bytes as number) ?? 9,
            count: lod.n_points as number,
            bytes: (lod.n_points as number) * ((lod.record_bytes as number) ?? 9),
            // real min_zoom_offsets are BYTE offsets -> records
            zoom_offsets: (lod.min_zoom_offsets as number[] | undefined)?.map((b) =>
              Math.round(b / ((lod.record_bytes as number) ?? 9)),
            ),
          }
        : undefined,
      deep: pts.n_points
        ? {
            path: "points/xy_id.bin",
            record_bytes: (pts.record_bytes as number) ?? 8,
            count: pts.n_points as number,
            bytes: (pts.n_points as number) * ((pts.record_bytes as number) ?? 8),
            tile_index: {
              path: "points/tile_index.u64",
              z: zmax,
              entries: ((pts.n_tiles as number) ?? (1 << zmax) ** 2) + 1,
              unit: "bytes",
            },
          }
        : undefined,
    },
    bins: {
      samples: { pattern: "bins/samples_z{z}.json", levels: binLevels },
      snippets: { pattern: "bins/snippets_z{z}.json", levels: binLevels },
      key: "{global_bin_x}_{global_bin_y}",
    },
    text: textRaw.text_available
      ? {
          offsets: "text/offsets.u64",
          blob: "text/blob.utf8",
          rows: nPoints,
          encoding: "utf-8",
        }
      : undefined,
    chunking: raw.chunking as Manifest["chunking"],
    // written by mirror_pack.py when the pack ships model/ (see types.ts)
    model: raw.model as Manifest["model"],
  };
}
