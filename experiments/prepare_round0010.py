#!/usr/bin/env python3
"""Round 0010 CPU-only input-pack driver.

Every production command requires CUDA to be hidden before Python starts and
routes all runtime output to /data/latent-basemap/runs/round-0010.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from basemap.minilm_input_pack import (
    DEFAULT_GRAPH,
    PackError,
    assemble_input_pack,
    build_source_inventory,
    convert_and_verify_endpoints,
    correct_endpoint_validation_receipt,
    materialize_fp16,
    reopen_input_pack,
    run_fixture_suite,
    seal_verification_logs,
)


ROOT = Path("/data/latent-basemap/runs/round-0010")
RECEIPTS = ROOT / "receipts"


def require_cpu_only_environment() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise PackError("Round 0010 requires CUDA_VISIBLE_DEVICES to be exactly empty")
    if "torch" in sys.modules:
        raise PackError("Torch was imported before the Round 0010 driver started")
    pycache = os.environ.get("PYTHONPYCACHEPREFIX")
    if not pycache:
        raise PackError("PYTHONPYCACHEPREFIX must point inside the Round 0010 /data root")
    try:
        Path(pycache).resolve().relative_to(ROOT)
    except ValueError as error:
        raise PackError("PYTHONPYCACHEPREFIX lies outside the Round 0010 root") from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory = subparsers.add_parser("inventory")
    inventory.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    inventory.add_argument("--rows-per-output-shard", type=int, default=1_000_000)

    materialize = subparsers.add_parser("materialize")
    materialize.add_argument(
        "--inventory", type=Path, default=RECEIPTS / "source-inventory.json"
    )
    materialize.add_argument("--rows-per-output-shard", type=int, default=1_000_000)
    materialize.add_argument("--block-rows", type=int, default=32_768)

    endpoints = subparsers.add_parser("endpoints")
    endpoints.add_argument(
        "--inventory", type=Path, default=RECEIPTS / "source-inventory.json"
    )

    correction = subparsers.add_parser("correct-endpoints")
    correction.add_argument(
        "--original", type=Path, default=RECEIPTS / "endpoints-reopen.json"
    )
    endpoints.add_argument(
        "--materialization",
        type=Path,
        default=RECEIPTS / "materialization-reopen.json",
    )

    subparsers.add_parser("fixtures")

    assemble = subparsers.add_parser("assemble")
    assemble.add_argument(
        "--inventory", type=Path, default=RECEIPTS / "source-inventory.json"
    )
    assemble.add_argument(
        "--materialization",
        type=Path,
        default=RECEIPTS / "materialization-reopen.json",
    )
    assemble.add_argument(
        "--endpoints",
        type=Path,
        default=RECEIPTS / "endpoints-reopen-correction-1.json",
    )
    assemble.add_argument(
        "--fixtures", type=Path, default=RECEIPTS / "fixture-receipt.json"
    )
    assemble.add_argument("--reopen", action="store_true")

    reopen = subparsers.add_parser("reopen")
    reopen.add_argument(
        "--manifest", type=Path, default=ROOT / "30m-input-pack-v1.json"
    )

    verification = subparsers.add_parser("seal-verification")
    verification.add_argument("--focused-log", type=Path, required=True)
    verification.add_argument("--focused-command", required=True)
    verification.add_argument("--suite-log", type=Path, required=True)
    verification.add_argument("--suite-command", required=True)

    subparsers.add_parser("status")
    return parser


def main() -> int:
    require_cpu_only_environment()
    args = build_parser().parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    if args.command == "inventory":
        result = build_source_inventory(
            ROOT,
            graph_path=args.graph,
            rows_per_output_shard=args.rows_per_output_shard,
        )
    elif args.command == "materialize":
        result = materialize_fp16(
            ROOT,
            args.inventory,
            rows_per_output_shard=args.rows_per_output_shard,
            block_rows=args.block_rows,
        )
    elif args.command == "endpoints":
        result = convert_and_verify_endpoints(
            ROOT, args.inventory, args.materialization
        )
    elif args.command == "correct-endpoints":
        result = correct_endpoint_validation_receipt(ROOT, args.original)
    elif args.command == "fixtures":
        result = run_fixture_suite(ROOT)
    elif args.command == "assemble":
        result = assemble_input_pack(
            ROOT,
            inventory_path=args.inventory,
            materialization_path=args.materialization,
            endpoints_path=args.endpoints,
            fixtures_path=args.fixtures,
        )
        if args.reopen:
            result = {
                "manifest": result,
                "reopen": reopen_input_pack(ROOT, ROOT / "30m-input-pack-v1.json"),
            }
    elif args.command == "reopen":
        result = reopen_input_pack(ROOT, args.manifest)
    elif args.command == "seal-verification":
        result = seal_verification_logs(
            ROOT,
            [
                {
                    "name": "focused-adversarial",
                    "command": args.focused_command,
                    "path": str(args.focused_log),
                },
                {
                    "name": "fresh-relevant-suite",
                    "command": args.suite_command,
                    "path": str(args.suite_log),
                },
            ],
        )
    elif args.command == "status":
        files = sorted(
            str(path.relative_to(ROOT))
            for path in ROOT.rglob("*")
            if path.is_file()
        )
        result = {"root": str(ROOT), "files": files}
    else:  # pragma: no cover - argparse enforces the enum
        raise AssertionError(args.command)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PackError as error:
        print(f"ROUND0010_FAIL_CLOSED: {error}", file=sys.stderr)
        raise SystemExit(2) from error
