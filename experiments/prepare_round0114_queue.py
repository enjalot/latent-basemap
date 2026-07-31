#!/usr/bin/env python3
"""Prepare the CPU-only R0112 native-8192 substrate recovery queue."""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import (
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from basemap.round0112_prompt_substrate import (
    ELIGIBILITY_PATH,
    MODEL_ROOT,
    first2m_layout,
    model_member_signatures,
)
from basemap.round0114_prompt_recovery import (
    CAPABILITY,
    ROUND_ID,
    SOURCE_FAILED_PATH,
    SOURCE_TERMINAL_PATH,
    source_chunk_path,
    validate_source_failure,
    validate_source_terminal,
)
from experiments.prepare_round0020_0022_queues import (
    LAB_ROOT,
    _base_manifest,
    _dedupe,
)


ROUND_ROOT = "/data/latent-basemap/runs/round-0114"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-cpu-run"
ROUND_FILE_GLOB = os.path.join(LAB_ROOT, "round-0114-*.md")


def _status(path: str) -> str | None:
    with open(path, encoding="utf-8") as handle:
        text = handle.read(4096)
    match = re.search(r"(?m)^status:\s*[\"']?([^\s\"']+)", text)
    return match.group(1) if match else None


def _issued_round() -> str:
    paths = [
        path
        for path in sorted(glob.glob(ROUND_FILE_GLOB))
        if _status(path) == "issued"
    ]
    if len(paths) != 1:
        raise RuntimeError(f"R0114 requires one issued round; found {len(paths)}")
    return paths[0]


def _read_json(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def prepare_round0114(
    *,
    release_sha: str,
    queue_root: str = os.path.join(ROUND_ROOT, "queue"),
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0114 release SHA must be one full commit")
    round_file = _issued_round()
    validate_source_terminal(_read_json(SOURCE_TERMINAL_PATH))
    validate_source_failure(_read_json(SOURCE_FAILED_PATH))

    chunk_inputs = [
        expected_input_signature(source_chunk_path(arm, chunk))
        for arm in ("raw", "document")
        for chunk in range(80)
    ]
    text_inputs = [
        expected_input_signature(str(item["text_path"]))
        for item in first2m_layout()
    ]
    model_inputs = [
        {
            key: value
            for key, value in member.items()
            if key != "model_relative_path"
        }
        for member in model_member_signatures()
    ]
    inputs = _dedupe(
        [
            expected_input_signature(round_file),
            expected_input_signature(SOURCE_TERMINAL_PATH),
            expected_input_signature(SOURCE_FAILED_PATH),
            expected_input_signature(ELIGIBILITY_PATH),
            *chunk_inputs,
            *text_inputs,
            *model_inputs,
        ]
    )
    queue_root = create_fresh_directory(
        queue_root,
        label="R0114 native-8192 substrate recovery queue",
    )
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    output = os.path.join(
        artifacts,
        "jina-fineweb-2m-dual-prompt-native8192-substrate",
    )
    queue = _base_manifest(
        round_id=ROUND_ID,
        release_sha=release_sha,
        round_file=round_file,
        queue_root=queue_root,
        gpu_hours_cap=0.0,
        execution_authority="autonomous-cpu",
        gpu=False,
    )
    queue["schema"] = "round0114-native8192-recovery-queue-v1"
    queue["repo_root"] = RELEASE_ROOT
    queue["queue_class"] = "cpu-io-heavy"
    queue["capability_dependencies"] = []
    queue["capabilities_produced"] = [CAPABILITY]
    queue["training_performed"] = False
    queue["scientific_contract"] = {
        "source_round": "0112",
        "source_verdict_preserved": "failed",
        "source_outputs_reused_without_mutation": True,
        "fresh_native_max_seq_length": 8192,
        "historical_max_seq_length": 512,
        "row_identity_sample": {
            "rows": 256,
            "same_row_must_be_top1_within_radius": 16,
            "all_rows_mean_cosine_floor": 0.98,
            "at_most_512_tokens_mean_cosine_floor": 0.98,
            "at_most_512_tokens_minimum_cosine_floor": 0.95,
        },
        "no_embedding_graph_training_or_evaluation": True,
    }
    node = "recover_native8192_substrate"
    queue["jobs"] = [
        {
            "id": node,
            "action": node,
            "handler_module": "experiments.round0114_nodes",
            "handler_callable": "run_job",
            "deps": [],
            "outputs": [output],
            "done_marker": os.path.join(artifacts, f"{node}.done.json"),
            "expected_inputs": inputs,
            "p90_wall_s": 1_200.0,
            "node_policy": {
                "gpu_required": False,
                "training_performed": False,
            },
        }
    ]
    queue["p90_gpu_seconds"] = {"total": 0.0}
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, queue, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument(
        "--queue-root",
        default=os.path.join(ROUND_ROOT, "queue"),
    )
    args = parser.parse_args(argv)
    print(
        json.dumps(
            {
                "queue_manifest": prepare_round0114(
                    release_sha=args.release_sha,
                    queue_root=args.queue_root,
                )
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
