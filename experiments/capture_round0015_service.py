"""Capture the one immutable Round 0015 exact-service decision."""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.round0015_service import write_service_decision


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", required=True, type=int)
    parser.add_argument("--environment-manifest", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    value = write_service_decision(
        args.out, pid=args.pid,
        environment_manifest=os.path.realpath(args.environment_manifest))
    print(json.dumps({
        "path": os.path.realpath(args.out),
        "policy": value["policy"],
        "pid": value["allowed_processes"][0]["pid"],
        "identity_sha256": value["identity_sha256"],
        "combined_cap_mib": value["memory_reservation"]["combined_cap_mib"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
