"""Execute R0240 — build and qualify the 100,000,000-row k15 fuzzy graph.

Two nodes, one queue, deliberately narrow. Nothing is assembled: R0238's
substrate, uniform truth probe and `c = 400` reachability ceiling are inherited
BY HASH and every registered literal about them is asserted before the first
cell runs.

* `ladder_100000k` (GPU) re-measures the five-seed imbalance grid at
  `N = 100,000,000`, `c = 400`, `s = 8` on this substrate — because
  review-0238-01/F5 showed the seed ranking INVERTS across a doubling, so no
  earlier grid predicts this one — re-derives `c` from that measurement under
  the unchanged `1.164884` margin, and builds the selected cell under
  signal-free cooperative supervision with the block-layer, `/proc/<pid>/io`
  and `mincore` instruments attached.
* `qualify_100000k` (GPU) scores the emitted graph against the inherited
  builder-independent truth on the uniform probe, applies the R0215 degree-zero
  tripwire over every row in BOTH directions, adjudicates any `min = 0.0` row
  individually, symmetrises through R0216's fuzzy law and publishes the per-rung
  re-derivation, the per-seed movement table and the measured tolerance.

**The science code is R0238's, called unchanged.** This module imports
`experiments.round0238_nodes` read-only and adds three things around it: the
fail-closed inheritance verification, and one sealed R0240 attestation per node
carrying the headline numbers and binding R0238's receipt by sha256. The
rationale, and the disclosed consequence that those receipts carry
`round_id: "0238"`, are in `basemap/round0240_rung5.py`'s module docstring.

No signal is delivered to any process on any path, and no round0215-0239 file
is modified.
"""
from __future__ import annotations

import os
import time
from collections.abc import Mapping
from typing import Any

from basemap import round0113_prompt_contrast as prompt_contract
from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import atomic_write_new_json
from basemap.round0238_rung5 import json_safe
from basemap.round0240_rung5 import (
    LADDER_ATTEST_FILE,
    QUALIFY_ATTEST_FILE,
    ROUND_ID,
    ROWS,
    Round0240Error,
    ladder_attestation,
    qualification_attestation,
    verify_inherited_reachability,
    verify_inherited_substrate,
    verify_inherited_truth,
)
from experiments.round0238_nodes import (
    LADDER_ACTION,
    QUALIFY_ACTION,
    run_ladder as _r0238_run_ladder,
    run_qualify as _r0238_run_qualify,
)
from basemap.round0253_stop_hooks import install_stop_hooks

#: The two actions this round authorizes. Deliberately the same identifiers
#: R0238 used, because the handler it delegates to dispatches on them.
LADDER_ACTION_0240 = LADDER_ACTION
QUALIFY_ACTION_0240 = QUALIFY_ACTION


def _sealed(job: Mapping[str, Any], key: str, *, label: str) -> tuple[str, dict[str, Any]]:
    """Read one hash-bound inherited manifest at its FULL signature."""
    reference = dict(job[key])
    if not reference.get("sha256"):
        raise Round0240Error(
            f"{label} must be bound at a full sha256 signature; R0240 inherits "
            "R0238's sealed artifacts across queues and never intra-queue"
        )
    path = prompt_contract.verify_signature(reference, label=label)
    return path, prompt_contract.read_sealed(path, label=label)


def verify_inheritance(job: Mapping[str, Any]) -> dict[str, Any]:
    """Assert every registered literal about the inherited artifacts. Fail-closed.

    Runs before any CUDA context exists. If any hash, composition figure,
    coverage figure, reserve figure or ceiling differs from what this round
    registered, it raises and the node stops with no GPU spent.
    """
    substrate_path, substrate = _sealed(
        job, "substrate_manifest", label="R0240 inherited substrate manifest"
    )
    truth_path, truth = _sealed(
        job, "truth_reference", label="R0240 inherited truth probe"
    )
    reach_path, reachability = _sealed(
        job, "reachability_reference", label="R0240 inherited reachability"
    )
    return {
        "substrate": {
            "source": expected_input_signature(substrate_path),
            **verify_inherited_substrate(substrate),
        },
        "truth": {
            "source": expected_input_signature(truth_path),
            **verify_inherited_truth(truth),
        },
        "reachability": {
            "source": expected_input_signature(reach_path),
            **verify_inherited_reachability(reachability),
        },
    }


def _seal_attestation(
    *,
    body: Mapping[str, Any],
    output: str,
    filename: str,
    inheritance: Mapping[str, Any],
    receipt_path: str,
    started: float,
) -> None:
    receipt = prompt_contract.seal(json_safe({
        **dict(body),
        "inheritance": dict(inheritance),
        "implementation_receipt": expected_input_signature(receipt_path),
        "attestation_wall_s": time.monotonic() - started,
    }))
    atomic_write_new_json(
        os.path.join(output, filename), receipt, immutable=True
    )


def run_ladder(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    """Verify the inheritance, run R0238's ladder unchanged, attest the result."""
    install_stop_hooks(label="R0253 round0240_nodes.run_ladder")
    manifest = active["manifest"]
    inheritance = verify_inheritance(job)
    started = time.monotonic()
    _r0238_run_ladder(active, job)
    output = str(job["outputs"][0])
    receipt_path = os.path.join(output, "build-ladder.json")
    if not os.path.exists(receipt_path):
        raise Round0240Error(f"R0240 ladder produced no receipt at {receipt_path}")
    ladder = prompt_contract.read_sealed(receipt_path, label="R0240 build ladder")
    if int(ladder.get("rows", -1)) != ROWS:
        raise Round0240Error("R0240 ladder receipt is not the 100,000,000 rung")
    _seal_attestation(
        body=ladder_attestation(
            ladder=ladder, release_sha=str(manifest["release_sha"])
        ),
        output=output,
        filename=LADDER_ATTEST_FILE,
        inheritance=inheritance,
        receipt_path=receipt_path,
        started=started,
    )


def run_qualify(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    """Verify the inheritance, run R0238's qualification unchanged, attest it."""
    install_stop_hooks(label="R0253 round0240_nodes.run_qualify")
    manifest = active["manifest"]
    inheritance = verify_inheritance(job)
    started = time.monotonic()
    _r0238_run_qualify(active, job)
    output = str(job["outputs"][0])
    receipt_path = os.path.join(output, "qualified-graph.json")
    if not os.path.exists(receipt_path):
        raise Round0240Error(
            f"R0240 qualification produced no receipt at {receipt_path}"
        )
    qualified = prompt_contract.read_sealed(
        receipt_path, label="R0240 qualified graph"
    )
    if int(qualified.get("rows", -1)) != ROWS:
        raise Round0240Error("R0240 graph receipt is not the 100,000,000 rung")
    _seal_attestation(
        body=qualification_attestation(
            qualified=qualified, release_sha=str(manifest["release_sha"])
        ),
        output=output,
        filename=QUALIFY_ATTEST_FILE,
        inheritance=inheritance,
        receipt_path=receipt_path,
        started=started,
    )


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    action = str(job.get("action") or "")
    if action == LADDER_ACTION_0240:
        run_ladder(active, job)
    elif action == QUALIFY_ACTION_0240:
        run_qualify(active, job)
    else:
        raise Round0240Error(f"R{ROUND_ID} does not authorize action {action!r}")


__all__ = [
    "LADDER_ACTION_0240",
    "QUALIFY_ACTION_0240",
    "run_job",
    "run_ladder",
    "run_qualify",
    "verify_inheritance",
]
