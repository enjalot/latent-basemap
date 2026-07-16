"""O1 — CPU-only tests for the prompted-200k row resolution, prompting, and
manifest logic used by experiments/embed_prompted_200k.py and
experiments/build_prompted_graph.py. No GPU, no network, no /data dependency:
everything here runs against synthetic in-memory arrays."""
import sys, os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from experiments.embed_prompted_200k import (
    build_shard_offsets, locate_shard, assert_row_identity, apply_prompt,
    l2_norm_stats, build_manifest, validate_manifest_shape, PROMPT_PREFIX,
    REQUIRED_MANIFEST_KEYS,
)


# ---------------------------------------------------------------------
# shard offsets / row resolution
# ---------------------------------------------------------------------

def test_build_shard_offsets():
    offsets = build_shard_offsets([10, 20, 5])
    assert offsets.tolist() == [0, 10, 30, 35]


def test_locate_shard_basic():
    offsets = build_shard_offsets([10, 20, 5])
    # row 0 -> shard 0 local 0; row 9 -> shard 0 local 9; row 10 -> shard 1 local 0
    # row 29 -> shard 1 local 19; row 30 -> shard 2 local 0; row 34 -> shard 2 local 4
    idx = np.array([0, 9, 10, 29, 30, 34])
    shard_ids, local = locate_shard(idx, offsets)
    assert shard_ids.tolist() == [0, 0, 1, 1, 2, 2]
    assert local.tolist() == [0, 9, 0, 19, 0, 4]


def test_locate_shard_is_order_preserving_not_sorted():
    """Row resolution must preserve the CALLER's order, even when the input
    indices are not sorted (e.g. a shuffled probe set)."""
    offsets = build_shard_offsets([10, 20, 5])
    idx = np.array([34, 0, 29, 10])
    shard_ids, local = locate_shard(idx, offsets)
    assert shard_ids.tolist() == [2, 0, 1, 1]
    assert local.tolist() == [4, 0, 19, 0]
    # round-trip: offsets[shard]+local reconstructs the original idx, in order
    recon = offsets[shard_ids] + local
    assert recon.tolist() == idx.tolist()


def test_locate_shard_rejects_out_of_range():
    offsets = build_shard_offsets([10, 20, 5])
    with pytest.raises(ValueError, match="outside covered range"):
        locate_shard(np.array([0, 35]), offsets)
    with pytest.raises(ValueError, match="outside covered range"):
        locate_shard(np.array([-1, 5]), offsets)


# ---------------------------------------------------------------------
# drift / abort guard
# ---------------------------------------------------------------------

def test_assert_row_identity_passes_on_exact_match():
    ids = np.array([5, 3, 9, 1])
    assert assert_row_identity(ids, ids.copy()) is True


def test_assert_row_identity_fires_on_reorder():
    """Same SET of ids but shuffled order must be rejected — order matters
    because embeddings are written positionally against sample_indices."""
    expected = np.array([1, 2, 3, 4])
    resolved = np.array([1, 2, 4, 3])   # last two swapped
    with pytest.raises(ValueError, match="index drift"):
        assert_row_identity(resolved, expected)


def test_assert_row_identity_fires_on_content_mismatch():
    expected = np.array([1, 2, 3, 4])
    resolved = np.array([1, 2, 3, 99])
    with pytest.raises(ValueError, match="index drift at position 3"):
        assert_row_identity(resolved, expected)


def test_assert_row_identity_fires_on_count_drift():
    expected = np.array([1, 2, 3, 4])
    resolved = np.array([1, 2, 3])
    with pytest.raises(ValueError, match="row count drift"):
        assert_row_identity(resolved, expected)


def test_locate_shard_and_identity_guard_compose():
    """End-to-end (still synthetic): resolve a shuffled index set via
    locate_shard, reconstruct global ids, and feed through the identity
    guard — this is the exact sequence embed_prompted_200k.main() runs
    before touching the model or GPU."""
    offsets = build_shard_offsets([100, 100, 100])
    sample_indices = np.array([250, 10, 199, 0, 300 - 1])
    shard_ids, local = locate_shard(sample_indices, offsets)
    resolved = offsets[shard_ids] + local
    assert assert_row_identity(resolved, sample_indices) is True

    # now corrupt one resolved id (simulating a resolution bug) and confirm
    # the guard fires instead of silently proceeding
    corrupted = resolved.copy()
    corrupted[2] = 42
    with pytest.raises(ValueError, match="index drift"):
        assert_row_identity(corrupted, sample_indices)


# ---------------------------------------------------------------------
# prompting
# ---------------------------------------------------------------------

def test_apply_prompt_prefixes_every_row_in_order():
    texts = ["alpha", "beta", "gamma"]
    out = apply_prompt(texts)
    assert out == ["Document: alpha", "Document: beta", "Document: gamma"]
    assert all(t.startswith(PROMPT_PREFIX) for t in out)


def test_apply_prompt_custom_prefix():
    assert apply_prompt(["x"], prefix="Query: ") == ["Query: x"]


def test_apply_prompt_empty_list():
    assert apply_prompt([]) == []


# ---------------------------------------------------------------------
# norm stats
# ---------------------------------------------------------------------

def test_l2_norm_stats_unit_norm():
    rng = np.random.RandomState(0)
    X = rng.randn(50, 8).astype(np.float32)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    stats = l2_norm_stats(X)
    assert stats["is_unit_norm"] is True
    assert abs(stats["mean"] - 1.0) < 1e-5


def test_l2_norm_stats_flags_non_unit_norm():
    X = np.full((10, 4), 2.0, dtype=np.float32)  # norm = 4.0 per row
    stats = l2_norm_stats(X)
    assert stats["is_unit_norm"] is False
    assert stats["mean"] == pytest.approx(4.0)


# ---------------------------------------------------------------------
# manifest well-formedness
# ---------------------------------------------------------------------

def _sample_manifest(n_rows=4, dim=3, faithfulness_passed=True):
    row_ids = np.arange(n_rows, dtype=np.int64)
    return build_manifest(
        model_id="jinaai/jina-embeddings-v5-text-nano-retrieval",
        model_commit="deadbeef1234",
        prompt_prefix=PROMPT_PREFIX,
        n_rows=n_rows, dim=dim, dtype="float32",
        row_ids=row_ids,
        text_shards=["data-00000-of-00099.parquet"],
        text_shard_sha={"data-00000-of-00099.parquet": "abc123"},
        out_shards=["data-00000.npy"],
        norm_check={"mean": 1.0, "std": 0.0, "min": 1.0, "max": 1.0, "is_unit_norm": True},
        faithfulness={"n_probe": 8, "mean_cosine": 0.995, "min_cosine": 0.986,
                      "std_cosine": 0.004, "threshold": 0.98, "passed": faithfulness_passed},
        canary={"canary_rows": 2000, "canary_wall_s": 5.0, "rows_per_s": 400.0,
                "estimated_total_wall_s": 500.0, "actual_total_wall_s": 510.0},
    )


def test_build_manifest_has_all_required_keys():
    man = _sample_manifest()
    for k in REQUIRED_MANIFEST_KEYS:
        assert k in man, f"missing manifest key {k}"


def test_manifest_prompt_prefix_hex_roundtrips():
    man = _sample_manifest()
    assert bytes.fromhex(man["prompt_prefix_hex"]).decode("utf-8") == PROMPT_PREFIX


def test_validate_manifest_shape_accepts_wellformed():
    validate_manifest_shape(_sample_manifest())  # should not raise


def test_validate_manifest_shape_rejects_missing_key():
    man = _sample_manifest()
    del man["faithfulness"]
    with pytest.raises(ValueError, match="missing required keys"):
        validate_manifest_shape(man)


def test_validate_manifest_shape_rejects_wrong_schema():
    man = _sample_manifest()
    man["schema"] = "some_other_schema"
    with pytest.raises(ValueError, match="schema"):
        validate_manifest_shape(man)


def test_validate_manifest_shape_rejects_tampered_prompt_hex():
    man = _sample_manifest()
    man["prompt_prefix_hex"] = "00"
    with pytest.raises(ValueError, match="prompt_prefix_hex"):
        validate_manifest_shape(man)


def test_validate_manifest_shape_rejects_incomplete_faithfulness():
    man = _sample_manifest()
    man["faithfulness"] = {"mean_cosine": 0.99}  # missing n_probe/min_cosine/passed
    with pytest.raises(ValueError, match="faithfulness missing key"):
        validate_manifest_shape(man)


def test_manifest_records_faithfulness_failure_without_hiding_it():
    """A manifest is allowed to record a FAILED faithfulness check (e.g. under
    the unsafe bypass) — well-formedness must not silently paper over a
    failing measurement."""
    man = _sample_manifest(faithfulness_passed=False)
    validate_manifest_shape(man)  # still structurally valid
    assert man["faithfulness"]["passed"] is False


# ---------------------------------------------------------------------
# graph-construction parity with the unprompted artifact (REVIEW: currently
# FAILING — documents a real divergence found in review)
# ---------------------------------------------------------------------

def test_build_fuzzy_graph_matches_unprompted_artifact_construction():
    """The unprompted testbed's edges_k50_fuzzy.npz has a hard out-degree
    floor of k-1 = 49 (10,626 nodes at exactly 49, none below) — the
    signature of the STANDARD UMAP construction: self-INCLUSIVE knn_indices
    (self at column 0, distance 0) passed to fuzzy_simplicial_set, whose
    compute_membership_strengths zeroes the self-edge, leaving k-1 real
    out-edges per node before symmetrization. Verified against the installed
    umap 0.5.12: umap-internal kNN and self-inclusive precomputed kNN both
    floor at k-1; the self-EXCLUDED construction floors at k and shifts every
    edge weight (rho/sigma are calibrated over a different distance list;
    max per-edge weight delta 0.69 on a synthetic probe).

    build_prompted_graph.build_fuzzy_graph currently passes k self-EXCLUDED
    neighbours (exclude_self_ids=ids), so the prompted graph is built by a
    systematically different recipe than the unprompted graph it is compared
    against — an unfair-comparison confound. This test pins the artifact's
    construction (out-degree floor k-1) and FAILS against the current code.
    Fix: pass self-inclusive top-k (exclude_self_ids=None; self ranks first
    under cosine on normalized vectors) so both graphs share the recipe."""
    from experiments.build_prompted_graph import build_fuzzy_graph
    rng = np.random.RandomState(0)
    n, d, k = 400, 16, 25
    X = rng.randn(n, d).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    sources, targets, weights, info = build_fuzzy_graph(X, k=k, seed=42, device="cpu")
    out_deg = np.bincount(sources, minlength=n)
    assert out_deg.min() == k - 1, (
        f"fuzzy graph out-degree floor is {out_deg.min()}, but the unprompted "
        f"edges_k50_fuzzy.npz artifact floors at k-1 (self-inclusive kNN, "
        f"UMAP-standard). Prompted and unprompted graphs are built by "
        f"different recipes -> unfair prompted-vs-unprompted comparison.")
