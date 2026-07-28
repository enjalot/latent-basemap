"""Run the read-only diverse jina inventory and duplicate census."""
from __future__ import annotations

import os
import resource
import time
from typing import Any

from basemap.artifact_identity import (
    canonical_json,
    expected_input_signature,
    sha256_bytes,
)
from basemap.output_safety import (
    atomic_save_new_npz,
    atomic_write_new_json,
    create_fresh_directory,
)
from basemap.round0087_inventory import (
    CATALOG_PATH,
    ROUND_ID,
    TARGET_ROWS,
    Round0087Error,
    build_selection,
    duplicate_census,
    inventory_datasets,
    reconcile_catalog,
)


def run_inventory(
    active: dict[str, Any],
    job: dict[str, Any],
) -> dict[str, Any]:
    output = create_fresh_directory(
        job["outputs"][0],
        label="R0087 diverse jina inventory",
    )
    started = time.monotonic()
    inventory = inventory_datasets()
    selection = build_selection(inventory)
    catalog = reconcile_catalog(inventory, catalog_path=CATALOG_PATH)
    census = duplicate_census(selection)
    eligibility_path = os.path.join(
        output, "jina-diverse-25m-eligibility-v1.npz"
    )
    atomic_save_new_npz(
        eligibility_path,
        immutable=True,
        compressed=False,
        **census["arrays"],
    )
    eligibility = expected_input_signature(eligibility_path)
    capability_ready = (
        selection["complete"] is True
        and selection["selected_rows"] == TARGET_ROWS
        and census["summary"]["row_count"] == TARGET_ROWS
        and census["summary"]["retained_row_count"] > 0
    )
    body = {
        "schema": "jina-diverse-25m-inventory-v1",
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "embedding_prompt": "raw",
        "inventory": inventory,
        "selection": selection,
        "catalog_reconciliation": catalog,
        "duplicate_control": {
            "definition": (
                "complete raw fp16 768d row-byte equality after a "
                "two-u64 fingerprint; every collision byte-verified"
            ),
            "representative_policy": (
                "lexicographically first dataset/shard/row in registered "
                "global selection order"
            ),
            "zero_policy": "exclude numeric positive-or-negative zero rows",
            "nonfinite_policy": "exclude and report",
            "summary": census["summary"],
            "eligibility": eligibility,
        },
        "gap_list": selection["gaps"],
        "capability_ready": capability_ready,
        "capability": (
            "jina-diverse-25m-inventory-v1"
            if capability_ready else None
        ),
        "authorizes": (
            "drafting only: 25M substrate/search/graph/train/evaluate rounds"
            if capability_ready
            else "no downstream 25M round; resolve recorded gaps"
        ),
        "no_embedding_performed": True,
        "training_performed": False,
        "optimizer_updates": 0,
        "gpu_used": False,
        "performance": {
            "wall_seconds": time.monotonic() - started,
            "peak_rss_gib": (
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / (1024 ** 2)
            ),
        },
    }
    manifest = {
        **body,
        "identity_sha256": sha256_bytes(canonical_json(body)),
    }
    path = os.path.join(output, "jina-diverse-25m-inventory-v1.json")
    atomic_write_new_json(path, manifest, immutable=True)
    return {**manifest, "receipt": expected_input_signature(path)}


def run_job(
    active: dict[str, Any],
    job: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if active.get("manifest", {}).get("round_id") != ROUND_ID or job is None:
        raise Round0087Error("R0087 handler requires its exact round/job")
    if job.get("action") != "inventory":
        raise Round0087Error("R0087 accepts only the inventory action")
    return run_inventory(active, job)
