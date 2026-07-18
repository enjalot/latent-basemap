"""CLI for Round 0017 immutable path/hash-reference staging."""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.round0017_staging import ROUND_ROOT, stage_input_references


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round-root", default=ROUND_ROOT)
    parser.add_argument("--release-root", required=True)
    parser.add_argument("--release-sha", required=True)
    args = parser.parse_args(argv)
    result = stage_input_references(
        round_root=args.round_root,
        release_root=os.path.realpath(args.release_root),
        release_sha=args.release_sha)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
