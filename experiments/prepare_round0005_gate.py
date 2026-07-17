"""Prepare the exact published Round 0005 queue with canonical Roundwatch."""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.gate_preparation import prepare_gate


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("queue_manifest")
    args = parser.parse_args(argv)
    receipt = prepare_gate(args.queue_manifest)
    print(json.dumps({"prepared": True, "gate_id": receipt["gate"]["id"],
                      "manifest_sha256": receipt["manifest_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
