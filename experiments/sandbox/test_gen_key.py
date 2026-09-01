"""Cache-invalidation tests for gen_key (5th review #3). Run: .venv/bin/python experiments/sandbox/test_gen_key.py"""
import sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_key import artifact_key, write_manifest, check_manifest, cached, proof_digest


def test_key_changes_on_config():
    base = {"kind": "armB-raw", "snapshot": "S3", "umap_kw": {"n_epochs": 500, "init": "spectral"},
            "seed": 0, "substrate_proof": "abc123"}
    k = artifact_key(base)
    assert artifact_key(base) == k, "key must be stable for identical config"
    assert artifact_key({**base, "seed": 43}) != k, "seed change must invalidate"
    assert artifact_key({**base, "umap_kw": {"n_epochs": 200, "init": "spectral"}}) != k, "umap kw change must invalidate"
    assert artifact_key({**base, "substrate_proof": "def456"}) != k, "substrate/proof digest change must invalidate"
    assert artifact_key({**base, "snapshot": "S4"}) != k, "snapshot change must invalidate"


def test_manifest_roundtrip_and_gate():
    with tempfile.TemporaryDirectory() as d:
        art = Path(d) / "coords-S3.npy"; art.write_bytes(b"fake-coords")
        k = artifact_key({"kind": "armB-raw", "seed": 0})
        assert not cached(art, k), "no manifest yet -> not cached (bare existence must NOT reuse)"
        write_manifest(art, k, {"note": "test"})
        assert check_manifest(art, k) and cached(art, k), "matching key -> cached"
        assert not cached(art, artifact_key({"kind": "armB-raw", "seed": 43})), "wrong key -> cache miss"
        assert not cached(Path(d) / "missing.npy", k), "absent artifact -> not cached"


def test_stale_proofs_incident():
    """The live incident: an invalid build wrote proofs, was replaced; the stale manifest must miss."""
    with tempfile.TemporaryDirectory() as d:
        art = Path(d) / "substrate.f32.npy"; art.write_bytes(b"x")
        proofs = Path(d) / "proofs.json"
        proofs.write_text('{"all_disjoint": true, "T0": "contiguous-INVALID"}')
        stale_key = artifact_key({"kind": "d768-T0", "substrate_proof": proof_digest(proofs)})
        write_manifest(art, stale_key, {"note": "invalid contiguous T0"})
        # corrected build changes the proofs -> new digest -> new key -> the stale manifest is rejected
        proofs.write_text('{"all_disjoint": true, "T0_equals_head_members": true}')
        corrected_key = artifact_key({"kind": "d768-T0", "substrate_proof": proof_digest(proofs)})
        assert corrected_key != stale_key, "changed proofs must change the key"
        assert not cached(art, corrected_key), "stale manifest must NOT satisfy the corrected key"


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"PASS {name}"); n += 1
    print(f"\n{n} tests passed")
