"""Generation-key manifests for evolution-benchmark artifacts (5th review #3).

Bare file-existence caching produced a live false-reuse this session: an INVALID contiguous-T0 D768 build
wrote a proofs JSON, was stopped, and the stale JSON was still read as if current. Fix: every cached
artifact (truth, head, armB raw coords, tranche) carries a sidecar `<artifact>.manifest.json` binding a
KEY = sha256 over {git commit, substrate/proof digest, full config incl. complete UMAP kwargs, seed}.
Reuse requires an EXACT key match — a changed commit/config/seed/substrate misses the cache and rebuilds.

Usage:
    key = artifact_key({"kind":"armB-raw","snapshot":"S3","umap_kw":UMAP_KW,"seed":0,
                        "substrate_proof": proof_digest(PROOFS_JSON)})
    if cached(path, key): reuse
    else: build...; write_manifest(path, key, {"note":...})
"""
import hashlib, json, subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def git_commit(repo: Path = REPO) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "nogit"


def file_digest(path) -> str:
    """Full sha256 of a small file (proofs JSON, config). Use this to bind to the PROOFS digest rather
    than the multi-GB substrate — a changed substrate is rebuilt -> new proofs -> new digest."""
    p = Path(path)
    if not p.exists():
        return "absent"
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def proof_digest(proofs_json) -> str:
    return file_digest(proofs_json)


def artifact_key(config: dict) -> str:
    """Stable 16-hex key over git commit + a canonical JSON of config (sorted keys, str fallback)."""
    payload = json.dumps({"_commit": git_commit(), **config}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _manifest_path(artifact_path) -> Path:
    return Path(str(artifact_path) + ".manifest.json")


def write_manifest(artifact_path, key: str, meta: dict | None = None) -> None:
    _manifest_path(artifact_path).write_text(json.dumps({"key": key, "commit": git_commit(),
                                                         **(meta or {})}, indent=1))


def check_manifest(artifact_path, key: str) -> bool:
    mp = _manifest_path(artifact_path)
    if not mp.exists():
        return False
    try:
        return json.loads(mp.read_text()).get("key") == key
    except Exception:
        return False


def cached(artifact_path, key: str) -> bool:
    """True iff the artifact exists AND its manifest key matches (the reuse gate). Never reuse on bare
    existence — a matching manifest is required."""
    return Path(artifact_path).exists() and check_manifest(artifact_path, key)
