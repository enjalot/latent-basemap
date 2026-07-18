"""Capture the immutable Round 0016 empty-GPU decision."""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.round0016_service import write_exclusive_decision


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment-manifest", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    value = write_exclusive_decision(
        args.out,
        environment_manifest=os.path.realpath(args.environment_manifest))
    print(json.dumps({
        "path": os.path.realpath(args.out),
        "policy": value["policy"],
        "allowed_processes": value["allowed_processes"],
        "compute_processes": len(value["gpu"]["compute_app_records"]),
        "identity_sha256": value["identity_sha256"],
        "job_cap_mib": value["memory_reservation"]["job_cap_mib"],
        "required_free_mib": value["memory_reservation"]["required_free_mib"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
