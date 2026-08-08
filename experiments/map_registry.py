#!/usr/bin/env python3
"""Map registry: index every trained basemap with provenance, publish a browsable site.

Read-only, post-hoc tooling (see latent-labs/guides/plan-map-inspection.md).
Never a launch-path dependency; must work with roundwatch down.

  uv run python experiments/map_registry.py scan      # -> /data/latent-basemap/maps.json
  uv run python experiments/map_registry.py publish   # -> ~/.agent/basemap-maps/ (gsv.local:8800/basemap-maps/)

The scanner keys off receipt presence (queue.json / render-manifest.json),
never a fixed tree, because only rounds 0014+ share the modern layout.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

RUNS_DIR = Path("/data/latent-basemap/runs")
CHECKPOINT_DIR = Path("/data/checkpoints/pumap")
LEDGER_DIR = Path.home() / "code/latent-labs/basemap-100m"
REGISTRY_PATH = Path("/data/latent-basemap/maps.json")
HISTORY_DIR = Path("/data/latent-basemap/registry-history")
SITE_DIR = Path.home() / ".agent/basemap-maps"
SITE_URL = "http://gsv.local:8800/basemap-maps"

SCHEMA = "basemap-map-registry-v2"


# ---------------------------------------------------------------- ledger ----

def _front_matter(path: Path) -> dict:
    """Minimal YAML front-matter reader (flat `key: value` lines only)."""
    out: dict = {}
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return out
    m = re.match(r"\A---\n(.*?)\n---\n", text, re.S)
    if not m:
        return out
    for line in m.group(1).splitlines():
        kv = re.match(r"^([A-Za-z_][A-Za-z0-9_]*):\s*(.*)$", line)
        if not kv:
            continue
        key, raw = kv.group(1), kv.group(2).strip()
        out[key] = raw.strip('"')
    return out


def ledger_status() -> dict:
    """round_id -> {round, result, review} front-matter status strings + doc names."""
    rounds: dict = {}
    if not LEDGER_DIR.is_dir():
        return rounds
    for doc in sorted(LEDGER_DIR.glob("*.md")):
        m = re.match(r"(round|result|review)-(\d{4})-", doc.name)
        if not m:
            continue
        kind, rid = m.group(1), m.group(2)
        fm = _front_matter(doc)
        entry = rounds.setdefault(rid, {})
        # keep the newest doc of each kind (suffix -01 style reissues sort last)
        entry[kind] = {"doc": doc.name, "status": fm.get("status", "unknown")}
    return rounds


def evidence_status(rid: str, ledger: dict) -> str:
    entry = ledger.get(rid, {})
    if "review" in entry:
        return f"review:{entry['review']['status']}"
    if "result" in entry:
        return "result:pending-review"
    if "round" in entry:
        return f"round:{entry['round']['status']}"
    return "unregistered"


# ----------------------------------------------------------------- scan -----

def _load_json(path: Path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _relpath(p: Path) -> str:
    return f"gsv:{p}"


def _file_signature(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "kind": "file",
        "canonical_path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _projection_map_context(panel: dict, rid: str) -> tuple[str, dict | None, dict | None]:
    current = panel.get("map") if isinstance(panel.get("map"), dict) else {}
    label = current.get("label")
    model = current.get("model") if isinstance(current.get("model"), dict) else None
    coordinates = current.get("coordinate_receipt") \
        if isinstance(current.get("coordinate_receipt"), dict) else None
    if not label and isinstance(panel.get("r0019_model"), dict):
        label = "r0019"
        model = panel["r0019_model"]
        coordinates = panel.get("r0019_coordinate_receipt")
    return str(label or f"round{rid}"), model, coordinates


def scan_projection_maps(
    round_dir: Path,
    ledger: dict,
    *,
    queue_dir: Path | None = None,
) -> list[dict]:
    """Discover immutable OOD coordinate archives beside registered panels."""
    rid_m = re.match(r"round-(\d{4})", round_dir.name)
    rid = rid_m.group(1) if rid_m else round_dir.name
    queue_dir = queue_dir or round_dir / "queue"
    artifacts = queue_dir / "artifacts"
    panels = sorted({
        *artifacts.glob("**/universality-panel-v1.json"),
        *artifacts.glob("**/common-corpus-ood-panel-v1.json"),
    }) if artifacts.is_dir() else []
    entries: list[dict] = []
    display_names = {
        "dadabase": "Dadabase jokes",
        "trec-covid": "TREC-COVID",
        "code": "Common Corpus code",
        "science": "Common Corpus science",
        "latin": "Common Corpus Latin",
        "pol_Latn": "Held-out Polish",
    }
    queue = _load_json(queue_dir / "queue.json") or {}
    release_sha = queue.get("release_sha") or (queue.get("release") or {}).get("sha")
    for panel_path in panels:
        panel = _load_json(panel_path)
        if not isinstance(panel, dict) or not isinstance(panel.get("probes"), dict):
            continue
        map_label, model, coordinate_receipt = _projection_map_context(panel, rid)
        coordinate_path = Path(coordinate_receipt.get("canonical_path", "")) \
            if isinstance(coordinate_receipt, dict) else None
        base_dir = coordinate_path.parent if coordinate_path and coordinate_path.is_file() else None
        sample_path = base_dir.parent / "semantic-renders/sample-semantic-ids.npy" \
            if base_dir else None
        for probe_name, probe in sorted(panel["probes"].items()):
            if not isinstance(probe, dict) or probe.get("status") != "included":
                continue
            coordinate_info = probe.get("coordinates") \
                if isinstance(probe.get("coordinates"), dict) else None
            archive = Path(coordinate_info.get("canonical_path", "")) \
                if coordinate_info else panel_path.parent / f"{probe_name}-coordinates.npz"
            if not archive.is_file():
                continue
            signature = coordinate_info or _file_signature(archive)
            metrics = probe.get("probe") if isinstance(probe.get("probe"), dict) else {}
            control = probe.get("matched_control")
            if not isinstance(control, dict):
                control = probe.get("shape_matched_in_domain_control")
            control = control if isinstance(control, dict) else {}
            inputs = probe.get("inputs") if isinstance(probe.get("inputs"), dict) else {}
            render_candidates = sorted(artifacts.glob(f"**/{probe_name}-overlay.png"))
            renders = [
                {"path": _relpath(path), "bytes": path.stat().st_size}
                for path in render_candidates
            ]
            map_id = (
                f"round-{rid}-{_slug(map_label)}-{_slug(probe_name)}-projection"
            )
            entries.append({
                "map_id": map_id,
                "round_id": rid,
                "kind": "projection-map",
                "date": datetime.fromtimestamp(
                    panel_path.stat().st_mtime, tz=timezone.utc).isoformat(),
                "evidence_status": evidence_status(rid, ledger),
                "base_map": map_label,
                "base_model": model,
                "base_coordinates": {
                    "dir": _relpath(base_dir),
                    "receipt": coordinate_receipt,
                } if base_dir else None,
                "base_sample_ids": {
                    "path": _relpath(sample_path),
                } if sample_path and sample_path.is_file() else None,
                "projection": {
                    "probe": probe_name,
                    "display_name": display_names.get(
                        probe_name, probe_name.replace("-", " ").title()),
                    "coordinates": _relpath(archive),
                    "coordinate_signature": signature,
                    "panel": _relpath(panel_path),
                    "panel_sha256": _file_signature(panel_path)["sha256"],
                    "corpus_rows": metrics.get("corpus_rows"),
                    "query_rows": metrics.get("query_rows"),
                    "ffr": metrics.get("ffr"),
                    "control_ffr": control.get("ffr"),
                    "retention": probe.get("retention"),
                    "verdict": probe.get("verdict"),
                    "inputs": inputs,
                },
                "renders": renders,
                "release_sha": release_sha,
                "run_dir": _relpath(round_dir),
            })
    return entries


def scan_modern_round(
    round_dir: Path,
    ledger: dict,
    *,
    queue_dir: Path | None = None,
) -> list[dict]:
    """Rounds with a queue/ manifest (0014+ layout). One entry per trained map."""
    rid_m = re.match(r"round-(\d{4})", round_dir.name)
    rid = rid_m.group(1) if rid_m else round_dir.name
    queue_dir = queue_dir or round_dir / "queue"
    art = queue_dir / "artifacts"
    queue = _load_json(queue_dir / "queue.json") or {}
    entries = []

    receipt = _load_json(art / "train/train-receipt.json")
    if not receipt:
        return entries
    prof = receipt.get("performance_profile", {})
    base = prof.get("baseline_key", {})
    stats = receipt.get("train_stats", {})
    config = _load_json(art / "train/production-config.json") or {}
    model_cfg = (config.get("config") or {}).get("model", {})
    panel = _load_json(art / "panel/panel.json") or {}
    render_manifest = _load_json(art / "semantic-renders/render-manifest.json") or {}
    transform = _load_json(art / "coordinates/actual-transform.json") or {}

    model = receipt.get("model", {})
    png = art / "semantic-renders/seed42-map.png"
    pngs = sorted((art / "semantic-renders").glob("*.png")) if (art / "semantic-renders").is_dir() else []
    coords_dir = art / "coordinates"
    coord_chunks = sorted(coords_dir.glob("chunk-*/coordinates.npy")) if coords_dir.is_dir() else []

    train_done = _load_json(art / "train_seed42_30m.done.json") or {}
    finished = train_done.get("finished")
    if not finished:
        for dm in sorted(art.glob("*.done.json")):
            j = _load_json(dm) or {}
            finished = j.get("finished") or finished

    p = panel.get("panel", {})
    proj = panel.get("projection", {})
    purity = p.get("purity", {}) if isinstance(p.get("purity"), dict) else {}
    checks = panel.get("decision_checks", {})

    entries.append({
        "map_id": f"round-{rid}-seed{(config.get('config') or {}).get('seed', stats.get('seed', 42))}",
        "round_id": rid,
        "kind": "round-map",
        "date": finished,
        "evidence_status": evidence_status(rid, ledger),
        "n_rows": base.get("n"),
        "dims": [base.get("d"), base.get("n_components")],
        "architecture": model_cfg.get("architecture"),
        "hidden_dim": base.get("hidden_dim") or model_cfg.get("hidden_dimension"),
        "kernel": base.get("kernel"),
        "pipeline": base.get("pipeline"),
        "precision": "bf16" if stats.get("amp_dtype") == "bfloat16" else ("fp16" if base.get("use_amp") else "fp32"),
        "updates": stats.get("optimizer_steps_succeeded"),
        "updates_per_s": stats.get("updates_per_s") or prof.get("rate_median"),
        "model": {"path": _relpath(art / "train/model.pt"), "sha256": model.get("sha256"), "bytes": model.get("bytes")},
        "coordinates": {
            "dir": _relpath(coords_dir),
            "chunks": len(coord_chunks),
            "receipt_sha256": (render_manifest.get("coordinate_stream") or {}).get("sha256")
                               or _sha_of(transform),
        },
        "panel": {
            "path": _relpath(art / "panel/panel.json"),
            "ffr": p.get("ffr"),
            "density": p.get("density"),
            "purity_k256": purity.get("k256"),
            "purity_k1024": purity.get("k1024"),
            "proj_ffr": proj.get("proj_ffr"),
            "proj_knn_ffr": proj.get("proj_knn_regressor_ffr"),
            "decision_checks_all_pass": bool(checks) and all(bool(v) for v in checks.values()),
            "formula_version": p.get("formula_version"),
        },
        "renders": [{"path": _relpath(x), "bytes": x.stat().st_size} for x in pngs],
        "render_diagnostics": render_manifest.get("diagnostics"),
        "release_sha": ((queue.get("release") or {}).get("sha")
                        or queue.get("release_sha")
                        or (panel.get("panel", {}).get("provenance") or {}).get("code_commit")),
        "run_dir": _relpath(round_dir),
    })
    return entries


SLIM_CELL_KEYS = ("cells", "new_cells")
SLIM_CELL_REQUIRED = {"capability", "coordinates", "panel_metrics", "seed"}


def _slim_cells(doc) -> tuple[str | None, dict]:
    """Per-seed map cells inside a slim-protocol panel/comparison artifact.

    Matched structurally, not by schema name: every round after 0218 mints its
    own schema string, so keying off those would need a code change per round.
    """
    if not isinstance(doc, dict):
        return None, {}
    for key in SLIM_CELL_KEYS:
        cells = doc.get(key)
        if not isinstance(cells, dict) or not cells:
            continue
        if all(
            isinstance(cell, dict) and SLIM_CELL_REQUIRED <= set(cell)
            for cell in cells.values()
        ):
            return key, cells
    return None, {}


def scan_slim_panel_round(
    round_dir: Path,
    ledger: dict,
    *,
    queue_dir: Path | None = None,
) -> list[dict]:
    """Rounds under the slim v2 protocol, which score a family of maps at once.

    R0218 onward publish one artifact JSON per round carrying a per-seed cell
    map; each cell binds its own checkpoint, its sealed 2-D coordinates and its
    panel metrics.  ``scan_modern_round`` cannot see any of it — that scanner
    assumes ``artifacts/train/`` and exactly one map per round — so the whole
    MiniLM 2M seed family and its cuVS-graph siblings were invisible to both the
    registry and the compare gallery.

    The treatment that separates those siblings is the *graph* each map trained
    on, read from the map's own train receipt (``graph_capability``), never from
    the round's prose: R0223's pipeline stamps still carry an
    ``R0216-exact-…`` policy label by carryover, which would mislabel every
    cuVS cell if trusted.
    """
    rid_m = re.match(r"round-(\d{4})", round_dir.name)
    rid = rid_m.group(1) if rid_m else round_dir.name
    queue_dir = queue_dir or round_dir / "queue"
    art = queue_dir / "artifacts"
    if not art.is_dir():
        return []
    queue = _load_json(queue_dir / "queue.json") or {}

    finished = None
    for dm in sorted(art.glob("*.done.json")):
        finished = (_load_json(dm) or {}).get("finished") or finished

    entries: list[dict] = []
    for doc_path in sorted(art.glob("*/*.json")):
        doc = _load_json(doc_path)
        cell_key, cells = _slim_cells(doc)
        if not cell_key:
            continue
        for seed in sorted(cells, key=lambda s: int(s) if s.isdigit() else s):
            cell = cells[seed]
            capability = str(cell.get("capability") or f"seed{seed}")
            coords = cell.get("coordinates") if isinstance(cell.get("coordinates"), dict) else {}
            coords_path = Path(str(coords.get("canonical_path") or ""))
            if not coords_path.is_file():
                continue
            model = cell.get("model") if isinstance(cell.get("model"), dict) else {}
            receipt_ref = cell.get("train_receipt") if isinstance(cell.get("train_receipt"), dict) else {}
            receipt = _load_json(Path(str(receipt_ref.get("canonical_path") or ""))) or {}
            model_path = Path(str(model.get("canonical_path") or ""))
            config = _load_json(model_path.parent / "production-config.json") or {}
            model_cfg = (config.get("config") or {}).get("model", {})
            pm = cell.get("panel_metrics") if isinstance(cell.get("panel_metrics"), dict) else {}
            graph_capability = str(receipt.get("graph_capability") or "")
            rows = receipt.get("rows") or doc.get("rows")
            # The map belongs to the round that TRAINED it, which is usually not
            # the round that scored it (R0218 scored R0217's family). Evidence
            # status and the map page follow training; `scored_in_round` records
            # where these numbers came from.
            trained_m = re.search(r"/runs/round-(\d{4})/", str(model_path))
            trained_rid = trained_m.group(1) if trained_m else rid

            entries.append({
                "map_id": f"round-{trained_rid}-{capability}",
                "title": f"seed {seed}",
                "round_id": trained_rid,
                "scored_in_round": rid,
                "kind": "round-map",
                "page": f"round-{trained_rid}/{_slug(capability)}",
                "date": finished,
                "evidence_status": evidence_status(trained_rid, ledger),
                "n_rows": int(rows) if rows else None,
                "dims": [receipt.get("dimension"), model_cfg.get("output_dimension")],
                "seed": int(seed) if str(seed).isdigit() else None,
                "architecture": model_cfg.get("architecture"),
                "hidden_dim": model_cfg.get("hidden_dimension"),
                "kernel": model_cfg.get("low_dim_kernel"),
                "graph": {
                    "capability": graph_capability,
                    "treatment": "cuvs" if "cuvs" in graph_capability else "exact",
                    "sha256": ((receipt.get("exact_execution_receipt") or {})
                               .get("graph", {}).get("graph", {}).get("sha256")),
                    "directed_edges": receipt.get("directed_edges"),
                },
                "updates": receipt.get("optimizer_updates"),
                "updates_per_s": receipt.get("steady_updates_per_s"),
                "model": {
                    "path": _relpath(model_path),
                    "sha256": model.get("sha256"),
                    "bytes": model.get("bytes"),
                },
                "coordinates": {
                    "file": _relpath(coords_path),
                    "dir": _relpath(coords_path.parent),
                    "rows": cell.get("transform_rows_finite"),
                    "sha256": coords.get("sha256"),
                    "ordered_sha256": cell.get("coordinates_ordered_sha256"),
                },
                "panel": {
                    "path": _relpath(doc_path),
                    "ffr": pm.get("ffr"),
                    "density": pm.get("density_v2"),
                    "density_semantics": "density-v2",
                    "purity_k256": pm.get("purity_fidelity_k256"),
                    "purity_k1024": pm.get("purity_fidelity_k1024"),
                    "proj_ffr": None,
                    "corpus_ffr": cell.get("corpus_ffr"),
                    "formula_version": (cell.get("panel") or {}).get("formula_version"),
                },
                "renders": [],
                "release_sha": receipt.get("release_sha")
                               or doc.get("release_sha")
                               or (queue.get("release") or {}).get("sha"),
                "run_dir": _relpath(round_dir),
            })
    return entries


def scan_evaluation_round(
    round_dir: Path,
    ledger: dict,
    *,
    queue_dir: Path | None = None,
) -> list[dict]:
    """Discover a map evaluated in a successor round to its training round.

    R0034 stops at a reviewed model candidate and R0036 owns its transform and
    quality claims.  Requiring a synthetic ``artifacts/train`` directory here
    would misstate that provenance, so the registry consumes the sealed
    transform's external model reference directly.
    """
    rid_m = re.match(r"round-(\d{4})", round_dir.name)
    rid = rid_m.group(1) if rid_m else round_dir.name
    queue_dir = queue_dir or round_dir / "queue"
    art = queue_dir / "artifacts"
    transform_path = art / "coordinates/actual-transform.json"
    panel_path = art / "panel/panel.json"
    render_path = art / "semantic-renders/render-manifest.json"
    transform = _load_json(transform_path)
    panel = _load_json(panel_path)
    if (
        not isinstance(transform, dict)
        or transform.get("schema") != "round0036-transform-capability-v1"
        or not isinstance(panel, dict)
        or panel.get("schema") != "round0036-registered-panel-v1"
    ):
        return []
    queue = _load_json(queue_dir / "queue.json") or {}
    render = _load_json(render_path) or {}
    scientific = panel.get("panel") if isinstance(panel.get("panel"), dict) else {}
    projection = panel.get("projection") \
        if isinstance(panel.get("projection"), dict) else {}
    purity = scientific.get("purity") \
        if isinstance(scientific.get("purity"), dict) else {}
    model = transform.get("model") if isinstance(transform.get("model"), dict) else {}
    chunks = sorted((art / "coordinates").glob("chunk-*/coordinates.npy"))
    pngs = sorted((art / "semantic-renders").glob("*.png")) \
        if (art / "semantic-renders").is_dir() else []
    finished = None
    for marker in sorted(art.glob("*.done.json")):
        finished = (_load_json(marker) or {}).get("finished") or finished
    checks = panel.get("decision_checks") or {}
    accounting = transform.get("row_accounting") or {}
    selector_passed = bool(checks) and all(bool(value) for value in checks.values())
    return [{
        "map_id": f"round-{rid}-r0034-seed42-150m",
        "round_id": rid,
        "kind": "round-map",
        "date": finished,
        "evidence_status": evidence_status(rid, ledger),
        "n_rows": accounting.get("all_rows"),
        "scientific_rows": accounting.get("retained_representatives"),
        "dims": [384, 2],
        "architecture": "residual_bottleneck",
        "hidden_dim": 2048,
        "kernel": "legacy_lp",
        "pipeline": "R0034-host-int8-canonical/R0036-retained-evaluation",
        "precision": "fp32-transform",
        "scientific_status": (
            "same-domain-selector-pass"
            if selector_passed
            else "same-domain-selector-failed-diagnostic"
        ),
        "capability_candidate": selector_passed,
        "model": model,
        "coordinates": {
            "dir": _relpath(art / "coordinates"),
            "chunks": len(chunks),
            "receipt_sha256": _file_signature(transform_path)["sha256"],
        },
        "panel": {
            "path": _relpath(panel_path),
            "ffr": scientific.get("ffr"),
            "density": scientific.get("density"),
            "purity_k256": purity.get("k256"),
            "purity_k1024": purity.get("k1024"),
            "proj_ffr": projection.get("proj_ffr"),
            "proj_knn_ffr": projection.get("proj_knn_regressor_ffr"),
            "decision_checks_all_pass": selector_passed,
            "formula_version": scientific.get("formula_version"),
        },
        "renders": [
            {"path": _relpath(path), "bytes": path.stat().st_size}
            for path in pngs
        ],
        "render_diagnostics": render.get("diagnostics"),
        "release_sha": queue.get("release_sha") or (queue.get("release") or {}).get("sha"),
        "run_dir": _relpath(round_dir),
        "training_round": "0034",
    }]


def scan_scale_evaluation_round(
    round_dir: Path,
    ledger: dict,
    *,
    queue_dir: Path | None = None,
) -> list[dict]:
    """Discover registry-facing maps produced by scale evaluation rounds.

    R0064 intentionally evaluates three map/universe pairs, but only the
    matched 30M control and the full 60M treatment are base maps.  The
    60M-model-on-30M pair is a same-row diagnostic and remains available in
    the scale-comparison receipt without becoming a third product map. Newer
    rounds declare their base maps in an immutable, queue-local definition
    file so registry discovery does not require another round-specific branch.
    """
    rid_m = re.match(r"round-(\d{4})", round_dir.name)
    rid = rid_m.group(1) if rid_m else round_dir.name
    queue_dir = queue_dir or round_dir / "queue"
    art = queue_dir / "artifacts"
    queue = _load_json(queue_dir / "queue.json") or {}
    render_manifest = _load_json(
        art / "semantic-renders/render-manifest.json"
    ) or {}
    if rid == "0064":
        definitions = (
            {
                "key": "r0061-30m-on-30m",
                "label": "r0061-balanced-30m-seed42",
                "coordinates": art / "coordinates-r0061-30m",
                "panel": art / "panel-r0061-30m/panel.json",
                "render": art / "semantic-renders/r0061-30m-on-30m.png",
                "training_round": "0061",
                "panel_schema": "round0064-registered-panel-v1",
            },
            {
                "key": "r0063-60m-on-60m",
                "label": "r0063-balanced-60m-seed42",
                "coordinates": art / "coordinates-r0063-60m",
                "panel": art / "panel-r0063-60m/panel.json",
                "render": art / "semantic-renders/r0063-60m-on-60m.png",
                "training_round": "0063",
                "panel_schema": "round0064-registered-panel-v1",
            },
        )
    else:
        declared = _load_json(
            art / "semantic-renders/scale-map-definitions.json"
        )
        if (
            not isinstance(declared, dict)
            or declared.get("schema") != "scale-map-definitions-v1"
            or declared.get("round_id") != rid
            or not isinstance(declared.get("maps"), list)
        ):
            return []
        definitions = []
        for raw in declared["maps"]:
            if not isinstance(raw, dict):
                return []
            resolved: dict[str, object] = {}
            for field in ("coordinates", "panel", "render"):
                value = raw.get(field)
                relative = Path(value) if isinstance(value, str) else None
                if (
                    relative is None
                    or relative.is_absolute()
                    or ".." in relative.parts
                ):
                    return []
                resolved[field] = art / relative
            definitions.append({
                "key": raw.get("key"),
                "label": raw.get("label"),
                "training_round": raw.get("training_round"),
                "panel_schema": raw.get("panel_schema"),
                "density_semantics": raw.get("density_semantics"),
                **resolved,
            })
    entries: list[dict] = []
    for definition in definitions:
        transform_path = definition["coordinates"] / "actual-transform.json"
        transform = _load_json(transform_path)
        panel = _load_json(definition["panel"])
        if (
            not isinstance(transform, dict)
            or transform.get("schema")
            != "round0036-transform-capability-v1"
            or transform.get("map_key") != definition["key"]
            or not isinstance(panel, dict)
            or panel.get("schema") != definition["panel_schema"]
            or panel.get("map_key") != definition["key"]
        ):
            continue
        scientific = panel.get("panel") or {}
        projection = panel.get("projection") or {}
        purity = scientific.get("purity") or {}
        accounting = transform.get("row_accounting") or {}
        model = transform.get("model") or {}
        chunks = sorted(
            definition["coordinates"].glob("chunk-*/coordinates.npy")
        )
        renders = (
            [{
                "path": _relpath(definition["render"]),
                "bytes": definition["render"].stat().st_size,
            }]
            if definition["render"].is_file()
            else []
        )
        checks = panel.get("decision_checks") or {}
        if definition.get("density_semantics") in {
            "representative-relative-v1",
            "density-v2-fixed-floor-plus-legacy-diagnostic",
        }:
            selector_checks = {
                key: value
                for key, value in checks.items()
                if key != "density_at_least_0_60"
            }
            selector_passed = bool(selector_checks) and all(
                bool(value) for value in selector_checks.values()
            )
            selector_label = "representative-non-density-selector-pass"
            selector_failed_label = (
                "representative-non-density-selector-failed-diagnostic"
            )
        else:
            # Preserve the registered decision on older panels. Some accepted
            # scale panels predate the durable decision_checks object even
            # though their absolute_selector_passed field is present.
            selector_passed = bool(panel.get("absolute_selector_passed"))
            selector_label = "same-domain-selector-pass"
            selector_failed_label = "same-domain-selector-failed-diagnostic"
        entries.append({
            "map_id": f"round-{rid}-{_slug(definition['label'])}",
            "round_id": rid,
            "kind": "round-map",
            "map_label": definition["label"],
            "date": datetime.fromtimestamp(
                definition["panel"].stat().st_mtime,
                tz=timezone.utc,
            ).isoformat(),
            "evidence_status": evidence_status(rid, ledger),
            "n_rows": accounting.get("all_rows"),
            "scientific_rows": accounting.get("retained_representatives"),
            "dims": [384, 2],
            "architecture": "residual_bottleneck",
            "hidden_dim": 2048,
            "kernel": "legacy_lp",
            "pipeline": (
                f"R{definition['training_round']}-host-int8-canonical/"
                f"R{rid}-representative-evaluation"
            ),
            "precision": "fp32-transform",
            "scientific_status": (
                selector_label
                if selector_passed
                else selector_failed_label
            ),
            "capability_candidate": selector_passed,
            "density_semantics": definition.get("density_semantics"),
            "model": model,
            "coordinates": {
                "dir": _relpath(definition["coordinates"]),
                "chunks": len(chunks),
                "receipt_sha256": _file_signature(transform_path)["sha256"],
            },
            "panel": {
                "path": _relpath(definition["panel"]),
                "ffr": scientific.get("ffr"),
                "density": scientific.get("density"),
                "purity_k256": purity.get("k256"),
                "purity_k1024": purity.get("k1024"),
                "proj_ffr": projection.get("proj_ffr"),
                "proj_knn_ffr": projection.get(
                    "proj_knn_regressor_ffr"
                ),
                "decision_checks_all_pass": selector_passed,
                "raw_decision_checks_all_pass": (
                    bool(checks)
                    and all(bool(value) for value in checks.values())
                ),
                "legacy_absolute_selector_passed": bool(
                    panel.get("absolute_selector_passed")
                ),
                "formula_version": scientific.get("formula_version"),
            },
            "renders": renders,
            "render_diagnostics": (
                (render_manifest.get("renders") or {}).get(
                    definition["key"]
                )
            ),
            "release_sha": queue.get("release_sha")
            or (queue.get("release") or {}).get("sha"),
            "run_dir": _relpath(round_dir),
            "training_round": definition["training_round"],
        })
    return entries


def scan_round0108_atlas(
    round_dir: Path,
    ledger: dict,
    *,
    queue_dir: Path | None = None,
) -> list[dict]:
    """Discover the diverse-Jina atlas evaluated by R0108.

    The map was trained in R0107 and transformed/evaluated in R0108.  Its
    compact retained order and Jina-specific density calibration are not the
    MiniLM scale-evaluation schema, so consume its explicit immutable
    definition instead of pretending it is an R0036 map.
    """
    queue_dir = queue_dir or round_dir / "queue"
    artifacts = queue_dir / "artifacts"
    definition_path = artifacts / "semantic-renders/map-definition.json"
    definition = _load_json(definition_path)
    transform_path = artifacts / "coordinates/actual-transform.json"
    core_path = artifacts / "core-geometry/core-geometry.json"
    decision_path = artifacts / "decision/atlas-decision.json"
    transform = _load_json(transform_path)
    core = _load_json(core_path)
    decision = _load_json(decision_path)
    if (
        not isinstance(definition, dict)
        or definition.get("schema") != "round0108-map-definition-v1"
        or definition.get("round_id") != "0108"
        or not isinstance(transform, dict)
        or transform.get("round_id") != "0108"
        or transform.get("map_key") != definition.get("map_key")
        or not isinstance(core, dict)
        or core.get("schema") != "round0108-diverse-jina-core-geometry-v1"
        or not isinstance(decision, dict)
        or decision.get("schema") != "round0108-diverse-jina-atlas-decision-v1"
        or definition.get("embedding_prompt") != "raw"
        or definition.get("prompt_applied") is not False
        or definition.get("production_document_prompt_transfer_resolved")
        is not False
        or definition.get("production_ready") is not False
    ):
        return []
    queue = _load_json(queue_dir / "queue.json") or {}
    global_metrics = (core.get("metrics") or {}).get("global") or {}
    density = (core.get("metrics") or {}).get("density_v2") or {}
    accounting = transform.get("row_accounting") or {}
    accepted = decision.get("atlas_quality_capability_released") is True
    coordinate_chunks = sorted(
        (artifacts / "coordinates").glob("chunk-*/coordinates.npy")
    )
    return [{
        "map_id": "round-0108-r0107-diverse-jina-25m-seed42",
        "round_id": "0108",
        "kind": "round-map",
        "map_label": definition.get("map_label"),
        "date": datetime.fromtimestamp(
            decision_path.stat().st_mtime, tz=timezone.utc
        ).isoformat(),
        "evidence_status": evidence_status("0108", ledger),
        "n_rows": accounting.get("all_rows"),
        "scientific_rows": accounting.get("retained_representatives"),
        "dims": [768, 2],
        "architecture": "residual_bottleneck",
        "hidden_dim": 2048,
        "kernel": "legacy_lp",
        "pipeline": "R0107 weighted-host-int8/R0108 retained evaluation",
        "precision": "fp32-transform",
        "embedding_prompt": definition.get("embedding_prompt"),
        "prompt_applied": definition.get("prompt_applied"),
        "production_document_prompt_transfer_resolved": definition.get(
            "production_document_prompt_transfer_resolved"
        ),
        "production_ready": definition.get("production_ready") is True,
        "scientific_status": (
            "core-and-polish-ood-pass"
            if accepted else "failed-with-registered-diagnostics"
        ),
        "capability_candidate": accepted,
        "density_semantics": "jina-density-v2-two-seed-calibrated",
        "model": transform.get("model"),
        "coordinates": {
            "dir": _relpath(artifacts / "coordinates"),
            "chunks": len(coordinate_chunks),
            "receipt_sha256": _file_signature(transform_path)["sha256"],
        },
        "panel": {
            "path": _relpath(core_path),
            "ffr": global_metrics.get("ffr"),
            "density": density.get("correlation"),
            "purity_k256": None,
            "purity_k1024": None,
            "proj_ffr": None,
            "decision_checks_all_pass": accepted,
            "formula_version": "density-v2-jina-calibrated",
        },
        "renders": [],
        "render_diagnostics": core.get("geometry_diagnostics"),
        "release_sha": queue.get("release_sha"),
        "run_dir": _relpath(round_dir),
        "training_round": "0107",
    }]


def _sha_of(obj) -> str | None:
    if isinstance(obj, dict):
        return obj.get("sha256") or obj.get("identity_sha256")
    return None


def scan_legacy_renders(round_dir: Path, ledger: dict) -> list[dict]:
    """Rounds with a top-level renders/ dir (round-0001 style) but no queue artifacts."""
    rid_m = re.match(r"round-(\d{4})", round_dir.name)
    rid = rid_m.group(1) if rid_m else round_dir.name
    renders_dir = round_dir / "renders"
    manifest = _load_json(renders_dir / "render-manifest.json")
    pngs = sorted(renders_dir.glob("*.png"))
    if not pngs:
        return []
    return [{
        "map_id": f"round-{rid}-legacy-renders",
        "round_id": rid,
        "kind": "legacy-renders",
        "date": datetime.fromtimestamp(pngs[0].stat().st_mtime, tz=timezone.utc).isoformat(),
        "evidence_status": evidence_status(rid, ledger),
        "renders": [{"path": _relpath(x), "bytes": x.stat().st_size} for x in pngs],
        "render_manifest": bool(manifest),
        "run_dir": _relpath(round_dir),
    }]


def scan_checkpoints() -> list[dict]:
    """Pre-round checkpoints in /data/checkpoints/pumap (best-effort, no metrics)."""
    if not CHECKPOINT_DIR.is_dir():
        return []
    out = []
    for pt in sorted(CHECKPOINT_DIR.glob("*.pt")):
        st = pt.stat()
        out.append({
            "map_id": f"checkpoint-{pt.stem}",
            "kind": "pre-round-checkpoint",
            "date": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
            "evidence_status": "pre-protocol",
            "model": {"path": _relpath(pt), "bytes": st.st_size},
        })
    return out


def _latest_queue_dir(round_dir: Path) -> Path | None:
    """Select the newest immutable slim-runner attempt for one round.

    ``queue`` is attempt 1. Setup corrections preserve it and materialize
    ``queue-attempt-N`` siblings; scanning only the original directory makes
    successful retry artifacts invisible to both the registry and the local
    explorer. The highest numbered materialized attempt is the authoritative
    artifact root while every older attempt remains untouched on disk.

    Retries have been named ``queue-correction-N`` since R0216, which is the
    naming every MiniLM 2M round used — including every cuVS-graph map. Only
    the two bare numbered forms count: a round's abandoned attempts carry a
    descriptive suffix (``queue-attempt-1-unrunnable-metadata``) and must stay
    invisible. No round on disk mixes the two families, so one numeric rank
    across both is unambiguous.
    """
    candidates: list[tuple[int, Path]] = []
    canonical = round_dir / "queue"
    if (canonical / "artifacts").is_dir():
        candidates.append((1, canonical))
    for candidate in round_dir.glob("queue-*"):
        match = re.fullmatch(r"queue-(?:attempt|correction)-(\d+)", candidate.name)
        if match and (candidate / "artifacts").is_dir():
            candidates.append((int(match.group(1)), candidate))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def scan() -> dict:
    ledger = ledger_status()
    maps: list[dict] = []
    if RUNS_DIR.is_dir():
        for round_dir in sorted(RUNS_DIR.glob("round-*")):
            if re.fullmatch(r"round-\d{4}", round_dir.name) is None:
                continue
            queue_dir = _latest_queue_dir(round_dir)
            if queue_dir is not None:
                maps += scan_modern_round(
                    round_dir, ledger, queue_dir=queue_dir
                )
                maps += scan_evaluation_round(
                    round_dir, ledger, queue_dir=queue_dir
                )
                maps += scan_scale_evaluation_round(
                    round_dir, ledger, queue_dir=queue_dir
                )
                maps += scan_round0108_atlas(
                    round_dir, ledger, queue_dir=queue_dir
                )
                maps += scan_slim_panel_round(
                    round_dir, ledger, queue_dir=queue_dir
                )
                maps += scan_projection_maps(
                    round_dir, ledger, queue_dir=queue_dir
                )
            elif (round_dir / "renders").is_dir():
                maps += scan_legacy_renders(round_dir, ledger)
    maps += scan_checkpoints()
    # A map re-scored by a later round (R0221's cells reappear in R0222) would
    # otherwise be indexed twice under one id. Keep the first sighting, which is
    # the earliest scoring round, and record the others.
    seen: dict[str, dict] = {}
    deduped: list[dict] = []
    for m in maps:
        prior = seen.get(m["map_id"])
        if prior is None:
            seen[m["map_id"]] = m
            deduped.append(m)
        elif m.get("scored_in_round"):
            prior.setdefault("also_scored_in", []).append(m["scored_in_round"])
    maps = deduped
    return {
        "schema": SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "counts": {
            "round_maps": sum(1 for m in maps if m["kind"] == "round-map"),
            "projection_maps": sum(1 for m in maps if m["kind"] == "projection-map"),
            "legacy_render_sets": sum(1 for m in maps if m["kind"] == "legacy-renders"),
            "pre_round_checkpoints": sum(1 for m in maps if m["kind"] == "pre-round-checkpoint"),
        },
        "maps": maps,
    }


def _content_sha(reg: dict) -> str:
    """Hash the inventory content, ignoring the generation timestamp, so a
    rescan that finds nothing new does not mint a new snapshot."""
    stable = {k: v for k, v in reg.items()
              if k not in ("generated_utc", "content_sha256", "mutable_view_note")}
    return hashlib.sha256(
        json.dumps(stable, sort_keys=True).encode()).hexdigest()


def write_registry(reg: dict) -> Path | None:
    """Write the mutable view, plus an immutable content-addressed snapshot
    under registry-history/ when the inventory actually changed.

    maps.json is a regenerated VIEW (protocol v2.1): rounds and reviews bind
    their own immutable snapshot artifacts, never this file. The history dir
    makes any later divergence diagnosable without blocking a review.
    """
    reg = dict(reg)
    sha = _content_sha(reg)
    reg["content_sha256"] = sha
    reg["mutable_view_note"] = (
        "regenerated view; rounds/reviews bind immutable snapshots, not this file")
    REGISTRY_PATH.write_text(json.dumps(reg, indent=1))
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    existing = sorted(HISTORY_DIR.glob("maps-*.json"))
    if existing:
        last = _load_json(existing[-1])
        if last is not None and last.get("content_sha256") == sha:
            return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snapshot = HISTORY_DIR / f"maps-{stamp}-{sha[:12]}.json"
    snapshot.write_text(json.dumps(reg, indent=1))
    return snapshot


# --------------------------------------------------------------- publish ----

CSS = """
:root { color-scheme: light dark; --fg:#1a1d21; --bg:#fff; --muted:#667; --line:#e2e5ea;
        --card:#f6f7f9; --ok:#0a7d33; --warn:#a15c00; --bad:#b3261e; }
@media (prefers-color-scheme: dark) {
  :root { --fg:#e6e8eb; --bg:#121417; --muted:#9aa1ab; --line:#2a2f36; --card:#1b1f24;
          --ok:#4ccb7a; --warn:#e0a34e; --bad:#e5776f; } }
* { box-sizing: border-box; }
body { font: 15px/1.5 system-ui, sans-serif; color: var(--fg); background: var(--bg);
       margin: 0 auto; max-width: 1100px; padding: 24px 20px 80px; }
h1 { font-size: 1.5rem; } h2 { font-size: 1.15rem; margin-top: 2rem; }
a { color: inherit; } small, .muted { color: var(--muted); }
table { border-collapse: collapse; width: 100%; font-size: 13.5px; }
th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--line); white-space: nowrap; }
th { position: sticky; top: 0; background: var(--bg); }
.num { text-align: right; font-variant-numeric: tabular-nums; }
.scroll { overflow-x: auto; }
.badge { padding: 1px 8px; border-radius: 9px; font-size: 12px; background: var(--card); }
.accepted { color: var(--ok); } .partial, .pending { color: var(--warn); } .rejected { color: var(--bad); }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px; margin: 12px 0; }
img.render { max-width: 100%; border: 1px solid var(--line); border-radius: 6px; background: #fff; }
code { font-size: 12.5px; background: var(--card); padding: 1px 5px; border-radius: 4px; }
dl { display: grid; grid-template-columns: max-content 1fr; gap: 3px 14px; margin: 8px 0; }
dt { color: var(--muted); } dd { margin: 0; overflow-wrap: anywhere; }
.cardgrid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
            gap: 14px; margin: 12px 0 8px; }
.mapcard { background: var(--card); border: 1px solid var(--line); border-radius: 10px;
           padding: 12px 13px; display: flex; flex-direction: column; gap: 8px; }
.mapcard .thumb { width: 100%; aspect-ratio: 1/1; object-fit: cover; border-radius: 7px;
                  border: 1px solid var(--line); background: #fff; }
.mapcard h3 { font-size: 0.98rem; margin: 0; line-height: 1.25; overflow-wrap: anywhere; }
.mapcard .meta { color: var(--muted); font-size: 12px; }
.chips { display: flex; flex-wrap: wrap; gap: 5px; }
.chip { padding: 1px 8px; border-radius: 9px; font-size: 12px; background: var(--bg);
        border: 1px solid var(--line); font-variant-numeric: tabular-nums; }
.chip.ok { color: var(--ok); border-color: var(--ok); }
.chip.bad { color: var(--bad); border-color: var(--bad); }
.viewerbtn { margin-top: auto; display: inline-block; text-align: center; text-decoration: none;
             padding: 6px 10px; border-radius: 8px; border: 1px solid var(--line);
             background: var(--bg); font-size: 13px; font-weight: 600; }
.viewerbtn:hover { border-color: var(--ok); }
.legacybtn { display: inline-block; text-align: center; margin-top: 4px; font-size: 12px;
             color: var(--muted); text-decoration: none; }
.legacybtn:hover { text-decoration: underline; }
"""


def _badge(status: str) -> str:
    cls = "muted"
    if "accepted" in status: cls = "accepted"
    elif "partial" in status or "pending" in status: cls = "partial"
    elif "rejected" in status: cls = "rejected"
    return f'<span class="badge {cls}">{html.escape(status)}</span>'


def _fmt(v, digits=4):
    if v is None: return "—"
    if isinstance(v, float): return f"{v:.{digits}f}"
    if isinstance(v, int) and v >= 1_000_000: return f"{v/1e6:.0f}M"
    return html.escape(str(v))


DENSITY_FLOOR = 0.60  # density_v2 pass threshold (density_at_least_0_60)


def _viewer_card(built: dict, entry: dict | None) -> str:
    """Render one map card: thumbnail, title, rows, evidence badge, metric chips."""
    panel = (entry or {}).get("panel", {}) if entry else {}
    ffr = panel.get("ffr")
    density = panel.get("density")
    rows = built.get("rows_total")
    checks_pass = panel.get("decision_checks_all_pass")

    ffr_cls = "ok" if checks_pass else ""
    ffr_chip = f'<span class="chip {ffr_cls}">FFR {_fmt(ffr)}</span>'
    if density is None:
        dens_chip = '<span class="chip">density_v2 —</span>'
    else:
        dens_ok = density >= DENSITY_FLOOR
        mark = "✓" if dens_ok else "✗"
        dens_chip = (f'<span class="chip {"ok" if dens_ok else "bad"}">'
                     f'density_v2 {_fmt(density)} {mark}</span>')

    thumb = built.get("thumb_rel")
    thumb_tag = (f'<img class="thumb" src="{html.escape(thumb)}" alt="density thumbnail" '
                 f'loading="lazy">' if thumb else "")
    rows_line = f'{_fmt(rows)} rows' if rows else ""
    evidence = built.get("evidence_status") or (entry or {}).get("evidence_status") or ""
    # Primary link is the React app (deployed to <site>/app/); the vanilla
    # viewer page stays reachable as a small secondary "legacy viewer" link.
    app_href = f'app/index.html#/map/{html.escape(built["map_id"])}'
    legacy = ""
    if built.get("viewer_rel"):
        legacy = (f'<a class="legacybtn" href="{html.escape(built["viewer_rel"])}">'
                  f'legacy viewer</a>')
    return (
        '<div class="mapcard">'
        f'{thumb_tag}'
        f'<h3>{html.escape(built.get("title") or built.get("map_id"))}</h3>'
        f'<div class="meta">{html.escape(rows_line)}</div>'
        f'<div class="chips">{_badge(evidence)}{ffr_chip}{dens_chip}</div>'
        f'<a class="viewerbtn" href="{app_href}">open viewer →</a>'
        f'{legacy}'
        '</div>'
    )


def _inject_viewer_cards(site_dir: Path, registry: dict, built: list[dict]) -> None:
    """Splice the viewer card grid into index.html between the section markers."""
    index_path = site_dir / "index.html"
    if not index_path.is_file() or not built:
        return
    by_id = {m.get("map_id"): m for m in registry.get("maps", [])}
    order = {m.get("map_id"): i for i, m in enumerate(
        sorted((m for m in registry.get("maps", []) if m.get("kind") == "round-map"),
               key=lambda x: x.get("date") or "", reverse=True))}
    cards = "".join(
        _viewer_card(b, by_id.get(b["map_id"]))
        for b in sorted(built, key=lambda b: order.get(b["map_id"], 1 << 30))
    )
    block = ('<!-- viewer-cards:start --><h2>Interactive maps</h2>'
             '<p class="muted">Binned density viewers with sampled tooltips, '
             'corpus/language overlays, and live quality-metric exemplars.</p>'
             f'<div class="cardgrid">{cards}</div><!-- viewer-cards:end -->')
    body = index_path.read_text()
    body = re.sub(r"<!-- viewer-cards:start -->.*?<!-- viewer-cards:end -->",
                  block, body, flags=re.DOTALL)
    index_path.write_text(body)


def _page_slug(m: dict) -> str:
    """Site-relative directory for a map page.

    One page per round was fine while a round trained one map; the slim v2
    family rounds train four at a time, so those nest a per-map directory and
    the round dir would otherwise be written four times, last one winning.
    """
    return str(m.get("page") or f'round-{m["round_id"]}')


def publish(registry: dict) -> None:
    SITE_DIR.mkdir(parents=True, exist_ok=True)
    round_maps = [m for m in registry["maps"] if m["kind"] == "round-map"]
    projections = [m for m in registry["maps"] if m["kind"] == "projection-map"]
    legacy = [m for m in registry["maps"] if m["kind"] == "legacy-renders"]
    checkpoints = [m for m in registry["maps"] if m["kind"] == "pre-round-checkpoint"]

    rows = []
    for m in sorted(round_maps, key=lambda x: x.get("date") or "", reverse=True):
        p = m["panel"]
        page = _page_slug(m) + "/index.html"
        rows.append(
            f'<tr><td><a href="{page}">{html.escape(m["map_id"])}</a></td>'
            f'<td>{(m.get("date") or "")[:10]}</td>'
            f'<td class="num">{_fmt(m.get("n_rows"))}</td>'
            f'<td>h{m.get("hidden_dim")} {html.escape(str(m.get("architecture") or ""))}</td>'
            f'<td class="num">{_fmt(p.get("ffr"))}</td>'
            f'<td class="num">{_fmt(p.get("density"))}</td>'
            f'<td class="num">{_fmt(p.get("purity_k1024"))}</td>'
            f'<td class="num">{_fmt(p.get("proj_ffr"))}</td>'
            f'<td>{_badge(m["evidence_status"])}</td></tr>'
        )
    legacy_rows = [
        f'<tr><td><a href="round-{m["round_id"]}/index.html">{html.escape(m["map_id"])}</a></td>'
        f'<td>{(m.get("date") or "")[:10]}</td><td class="num">{len(m["renders"])} renders</td>'
        f'<td>{_badge(m["evidence_status"])}</td></tr>'
        for m in legacy
    ]
    projection_rows = []
    for m in sorted(projections, key=lambda x: x.get("date") or "", reverse=True):
        p = m["projection"]
        page = f'projections/{m["map_id"]}/index.html'
        # A projection gains a React-app viewer when its coordinate npz exists
        # (the same condition map_viewer uses to build viewer/<map_id>/data/).
        npz_path = Path((p.get("coordinates") or "").removeprefix("gsv:"))
        app_link = ""
        if npz_path.is_file():
            app_link = (f' · <a href="app/index.html#/map/{html.escape(m["map_id"])}">'
                        f'app</a>')
        projection_rows.append(
            f'<tr><td><a href="{page}">{html.escape(m["map_id"])}</a>{app_link}</td>'
            f'<td>{html.escape(str(m.get("base_map") or ""))}</td>'
            f'<td class="num">{_fmt(p.get("corpus_rows"))}</td>'
            f'<td class="num">{_fmt(p.get("ffr"))}</td>'
            f'<td class="num">{_fmt(p.get("control_ffr"))}</td>'
            f'<td class="num">{_fmt(p.get("retention"))}</td>'
            f'<td>{html.escape(str(p.get("verdict") or "—"))}</td>'
            f'<td>{_badge(m["evidence_status"])}</td></tr>'
        )
    ckpt_rows = [
        f'<tr><td>{html.escape(m["map_id"])}</td><td>{(m.get("date") or "")[:10]}</td>'
        f'<td class="num">{m["model"]["bytes"]/1e6:.0f} MB</td>'
        f'<td><code>{html.escape(m["model"]["path"])}</code></td></tr>'
        for m in checkpoints
    ]

    index = f"""<!doctype html><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>basemap maps</title><style>{CSS}</style>
<h1>Basemap map registry</h1>
<p class="muted">Generated {registry["generated_utc"][:19]}Z from
<code>/data/latent-basemap/maps.json</code> · {registry["counts"]["round_maps"]} round maps ·
{registry["counts"].get("projection_maps", 0)} projection maps ·
{registry["counts"]["legacy_render_sets"]} legacy render sets ·
{registry["counts"]["pre_round_checkpoints"]} pre-protocol checkpoints ·
<a href="../basemap-gallery/">old gallery (2026-07-01)</a> ·
<a href="http://gsv.local:8710/">roundwatch</a></p>
<!-- viewer-cards:start --><!-- viewer-cards:end -->
<h2>Projection maps</h2>
<p class="muted">Foreign-domain corpora and held-out queries projected through a registered basemap. Browser plots are deterministic samples; metrics use the complete registered panel.</p>
<div class="scroll"><table>
<tr><th>projection</th><th>base map</th><th>probe N</th><th>probe ffr</th><th>control ffr</th><th>retention</th><th>verdict</th><th>evidence</th></tr>
{''.join(projection_rows) or '<tr><td colspan=8 class="muted">none</td></tr>'}
</table></div>
<h2>Round maps</h2>
<div class="scroll"><table>
<tr><th>map</th><th>date</th><th>N</th><th>net</th><th>ffr</th><th>density</th>
<th>purity k1024</th><th>proj ffr</th><th>evidence</th></tr>
{''.join(rows) or '<tr><td colspan=9 class="muted">none</td></tr>'}
</table></div>
<h2>Legacy render sets</h2>
<div class="scroll"><table><tr><th>set</th><th>date</th><th>contents</th><th>evidence</th></tr>
{''.join(legacy_rows) or '<tr><td colspan=4 class="muted">none</td></tr>'}</table></div>
<h2>Pre-protocol checkpoints</h2>
<div class="scroll"><table><tr><th>checkpoint</th><th>date</th><th>size</th><th>path</th></tr>
{''.join(ckpt_rows) or '<tr><td colspan=4 class="muted">none</td></tr>'}</table></div>
"""
    compare_links = ""
    try:
        from gallery_v2 import build_compare_groups
        groups = build_compare_groups(registry, SITE_DIR)
        if groups:
            items = " · ".join(
                f'<a href="compare/{g["slug"]}/index.html">{g["n_rows"]:,} rows ({g["panels"]} maps)</a>'
                for g in groups)
            compare_links = f'<h2>Interactive comparison</h2><p>Linked small-multiples on a common 20k sample: {items}</p>'
    except Exception as e:  # viewer generation is best-effort; the index must still publish
        compare_links = f'<p class="muted">compare viewer generation failed: {html.escape(str(e))}</p>'
    index = index.replace("<h2>Round maps</h2>", compare_links + "\n<h2>Round maps</h2>", 1)

    (SITE_DIR / "index.html").write_text(index)

    for m in round_maps + legacy:
        slug = _page_slug(m)
        page_dir = SITE_DIR / slug
        page_dir.mkdir(parents=True, exist_ok=True)
        up = "../" * len(Path(slug).parts)
        img_tags = []
        for r in m.get("renders", []):
            src = Path(r["path"].removeprefix("gsv:"))
            if src.is_file():
                dst = page_dir / src.name
                if not dst.exists() or dst.stat().st_size != src.stat().st_size:
                    shutil.copy2(src, dst)
                img_tags.append(f'<p><img class="render" src="{src.name}" alt="{src.name}">'
                                f'<br><small>{src.name}</small></p>')
        panel = m.get("panel", {})
        dl_items = []
        for label, val in [
            ("evidence", m["evidence_status"]), ("date", m.get("date")),
            ("scored in round", m.get("scored_in_round")
             if m.get("scored_in_round") not in (None, m.get("round_id")) else None),
            ("also scored in", ", ".join(m.get("also_scored_in", [])) or None),
            ("seed", m.get("seed")),
            ("graph", (m.get("graph") or {}).get("capability")),
            ("graph sha256", (m.get("graph") or {}).get("sha256")),
            # Exact and cuVS graphs differ by 0.03% of their edge count, so the
            # abbreviating formatter would print both as "48M".
            ("graph directed edges", f'{edges:,}' if (
                edges := (m.get("graph") or {}).get("directed_edges")) else None),
            ("N rows", m.get("n_rows")), ("architecture",
             f'{m.get("architecture")} h{m.get("hidden_dim")} → {m.get("dims")}' if m.get("architecture") else None),
            ("kernel", m.get("kernel")), ("pipeline", m.get("pipeline")),
            ("precision", m.get("precision")), ("updates", m.get("updates")),
            ("updates/s", m.get("updates_per_s")), ("release", m.get("release_sha")),
            ("model", (m.get("model") or {}).get("path")),
            ("model sha256", (m.get("model") or {}).get("sha256")),
            ("coordinates", (m.get("coordinates") or {}).get("dir")),
            ("coordinates file", (m.get("coordinates") or {}).get("file")),
            ("panel file", panel.get("path")), ("panel version", panel.get("formula_version")),
            ("density semantics", panel.get("density_semantics")),
            ("run dir", m.get("run_dir")),
        ]:
            if val is not None:
                dl_items.append(f"<dt>{label}</dt><dd>{_fmt(val) if isinstance(val,(int,float)) else html.escape(str(val))}</dd>")
        metrics = ""
        if panel.get("ffr") is not None:
            metrics = (f'<div class="card"><b>Panel</b><dl>'
                       f'<dt>ffr@0.1%</dt><dd>{_fmt(panel.get("ffr"))}</dd>'
                       f'<dt>density</dt><dd>{_fmt(panel.get("density"))}</dd>'
                       f'<dt>purity k256 / k1024</dt><dd>{_fmt(panel.get("purity_k256"))} / {_fmt(panel.get("purity_k1024"))}</dd>'
                       f'<dt>proj ffr (vs kNN reg)</dt><dd>{_fmt(panel.get("proj_ffr"))} (vs {_fmt(panel.get("proj_knn_ffr"))})</dd>'
                       f'<dt>all decision checks</dt><dd>{"PASS" if panel.get("decision_checks_all_pass") else "see panel.json"}</dd>'
                       f'</dl></div>')
        page = f"""<!doctype html><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(m["map_id"])}</title><style>{CSS}</style>
<p><a href="{up}index.html">← all maps</a></p>
<h1>{html.escape(m["map_id"])}</h1>
{f'<p class="muted">{html.escape(str(m.get("title")))}</p>' if m.get("title") else ''}
{metrics}
<div class="card"><b>Provenance</b><dl>{''.join(dl_items)}</dl></div>
<h2>Renders</h2>
{''.join(img_tags) or '<p class="muted">no renders on disk</p>'}
"""
        (page_dir / "index.html").write_text(page)

    projection_count = 0
    try:
        try:
            from experiments.projection_gallery import build_projection_explorers
        except ModuleNotFoundError:  # direct script execution
            from projection_gallery import build_projection_explorers
        projection_count = len(build_projection_explorers(registry, SITE_DIR))
    except Exception as exc:
        print(f"projection explorer generation failed: {exc}")

    viewer_count = 0
    try:
        try:
            from experiments.map_viewer import build_map_viewers
        except ModuleNotFoundError:  # direct script execution
            from map_viewer import build_map_viewers
        built = build_map_viewers(registry, SITE_DIR)
        _inject_viewer_cards(SITE_DIR, registry, built)
        viewer_count = len(built)
    except Exception as exc:  # viewer build is best-effort; index must still stand
        print(f"map viewer generation failed: {exc}")

    print(
        f"published {len(round_maps)+len(legacy)} map pages, "
        f"{projection_count} projection explorers and "
        f"{viewer_count} interactive viewers -> {SITE_DIR}  ({SITE_URL}/)"
    )


# ------------------------------------------------------------------ main ----

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["scan", "publish"])
    args = ap.parse_args()
    if args.command == "scan":
        reg = scan()
        snapshot = write_registry(reg)
        print(f"wrote {REGISTRY_PATH}: {reg['counts']}"
              + (f"; snapshot {snapshot}" if snapshot else "; inventory unchanged"))
    else:
        reg = _load_json(REGISTRY_PATH)
        if reg is None or reg.get("schema") != SCHEMA:
            reg = scan()
            write_registry(reg)
            print(f"(re)scanned -> {REGISTRY_PATH}")
        publish(reg)


if __name__ == "__main__":
    main()
