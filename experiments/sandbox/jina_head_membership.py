#!/usr/bin/env python3
"""Exact member masks for jina heads within the 6.25M substrate (4th-review P1.6, 2026-08-29).

The jina-prompted 6.25M is ordered [EN x3: old||new] then [lang x20: old||new]. A head trained on
a SMALLER jina substrate is a NESTED subset of the 6.25M ONLY through the OLD blocks, which sit at
the START of each span — so its member set is NOT the first-N contiguous rows (the naive
member_cutoff=N, which is WRONG: the 6.25M's first 2M rows are ~2M of EN, not the 2M head's rows).

  * 2M head (jina-multi-2m = _jina_multi_load): 1M EN (old-block prefixes 333334/333333/333333 of
    the 3 EN spans) + 1M ML (first 50000 = the full old block of each of the 20 lang spans).
  * 4M head (build_jina_4m_head.py): explicit seed-42 64%-per-span draw; member_indices.npy holds
    the exact global 6.25M indices.

Span boundaries are reconstructed EXACTLY from the component block sizes (sum to 6,250,000).
"""
from pathlib import Path

import numpy as np

P = Path("/data/latent-basemap/substrates/jina-prompted")
FOURM = Path("/data/latent-basemap/substrates/jina-4m-head")
EN = ("fineweb-edu", "redpajama", "pile")
LANGS = ("arb_Arab", "ces_Latn", "cmn_Hani", "deu_Latn", "ell_Grek", "fra_Latn",
         "hin_Deva", "ind_Latn", "ita_Latn", "jpn_Jpan", "kor_Hang", "nld_Latn",
         "pol_Latn", "por_Latn", "rus_Cyrl", "spa_Latn", "swe_Latn", "tha_Thai",
         "tur_Latn", "vie_Latn")
# _jina_multi_load composition: EN old-block prefixes taken, ML full old blocks.
EN_2M_PREFIX = (333_334, 333_333, 333_333)
ML_2M_PREFIX = 50_000


def _rows(f):
    return int(np.load(P / f, mmap_mode="r").shape[0])


def span_bounds():
    """[(label, start, length)] for the 6.25M spans in build order (EN x3 then lang x20),
    exact from component block sizes."""
    out, off = [], 0
    for c in EN:
        ln = _rows(f"en-{c}.f16.npy") + _rows(f"en2-{c}.f16.npy")
        out.append((f"en-{c}", off, ln)); off += ln
    for l in LANGS:
        ln = _rows(f"ml-{l}.f16.npy") + _rows(f"ml2-{l}.f16.npy")
        out.append((f"ml-{l}", off, ln)); off += ln
    return out, off


def member_mask_2m(n6=None):
    """Exact boolean member mask (length n6) for the 2M head: old-block prefixes of each span."""
    bounds, total = span_bounds()
    if n6 is None:
        n6 = total
    assert n6 == total, f"n6 {n6} != reconstructed 6.25M {total}"
    mask = np.zeros(n6, dtype=bool)
    for i, (label, start, length) in enumerate(bounds):
        if label.startswith("en-"):
            k = EN_2M_PREFIX[i]
        else:
            k = ML_2M_PREFIX
        mask[start:start + k] = True
    assert mask.sum() == 2_000_000, f"2M mask sum {mask.sum()} != 2,000,000"
    return mask


def member_mask_4m(n6=None):
    """Exact boolean member mask (length n6) for the 4M head from member_indices.npy."""
    idx = np.load(FOURM / "member_indices.npy")
    if n6 is None:
        _, n6 = span_bounds()
    mask = np.zeros(n6, dtype=bool)
    mask[idx] = True
    assert mask.sum() == idx.shape[0], "4M mask sum mismatch"
    return mask


if __name__ == "__main__":
    b, tot = span_bounds()
    print(f"6.25M reconstructed rows: {tot}")
    m2 = member_mask_2m(tot); m4 = member_mask_4m(tot)
    print(f"2M mask sum: {m2.sum():,}  4M mask sum: {m4.sum():,}")
    print(f"2M ⊂ 4M? (overlap {np.logical_and(m2, m4).sum():,} of 2M {m2.sum():,})")
    print(f"first EN span members contiguous 0:{EN_2M_PREFIX[0]} -> {m2[:EN_2M_PREFIX[0]].all()}, "
          f"gap after -> {not m2[EN_2M_PREFIX[0]]}")
