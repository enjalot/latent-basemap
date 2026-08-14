/** Types for the `basemap-map-pack-v1` contract (see fixtures/make_fixture_pack.py). */

export interface Corpus {
  code: number;
  name: string;
  label?: string;
  color?: string;
}

export interface Extent {
  xmin: number;
  ymin: number;
  xmax: number;
  ymax: number;
}

export interface DensityLevel {
  z: number;
  tiles: number;
  bins?: number;
  max_count?: number;
  /** optional sparse inventory: "x_y" keys that exist on disk */
  tile_list?: string[];
  /** optional: which corpus planes exist per tile */
  tile_corpora?: Record<string, number[]>;
}

export interface Manifest {
  schema?: string;
  map_id: string;
  title?: string;
  description?: string;
  synthetic?: boolean;
  N: number;
  extent: Extent;
  tile_size?: number;
  zoom: { min: number; max: number };
  corpora: Corpus[];
  quantization?: {
    xy_bits?: number;
    xy_max?: number;
    id_bits?: number;
    corpus_bits?: number;
    y_down?: boolean;
  };
  density?: {
    pattern?: string;
    png?: string;
    plane_dtype?: string;
    plane_bytes?: number;
    levels: DensityLevel[];
  };
  points?: {
    lod?: {
      path: string;
      record_bytes?: number;
      count?: number;
      bytes?: number;
      /** record offset where each min-zoom band starts; length zmax+2 */
      zoom_offsets?: number[];
    };
    deep?: {
      path: string;
      record_bytes?: number;
      count?: number;
      bytes?: number;
      tile_index?: {
        path: string;
        z: number;
        entries?: number;
        /** "records" (default) or "bytes" */
        unit?: string;
      };
    };
  };
  bins?: {
    samples?: { pattern?: string; levels: number[] };
    snippets?: { pattern?: string; levels: number[] };
    key?: string;
  };
  text?: {
    offsets: string;
    blob: string;
    rows?: number;
    encoding?: string;
  };
  chunking?: {
    part_bytes: number;
    suffix?: string;
    files: Record<string, { bytes: number; parts: number }>;
  };
  /**
   * In-browser projection assets, written by `scripts/mirror_pack.py` when the
   * pack ships `model/`. Declared in the manifest rather than probed, so the
   * viewer never 404s looking for a head that isn't there.
   */
  model?: {
    /** relative to the pack base — mappack-models-v1 */
    models_json: string;
    /** relative to the pack base — ONNX map head, fp32, dynamic batch */
    map_head: string;
    map_head_bytes?: number;
    encoder?: string;
  };
}

export interface PackIndexEntry {
  map_id: string;
  title?: string;
  path?: string;
  N?: number;
  zmax?: number;
  synthetic?: boolean;
  bytes?: number;
}

export interface PackIndex {
  schema?: string;
  packs: PackIndexEntry[];
}

/** How byte-range reads are satisfied against a given data source. */
export type RangeMode = "http-range" | "chunked" | "unavailable";
