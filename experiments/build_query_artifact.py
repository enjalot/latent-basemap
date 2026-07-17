"""Build an explicit, content-bound held-out query artifact (CPU only)."""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from basemap.query_artifact import (build_query_artifact, round0005_query_convention,
                                    validate_round0005_query_convention)
from basemap.round0005_staging import (
    ROUND0005_DIMENSIONS, ROUND0005_QUERY_ROWS, ROUND0005_QUERY_SEED,
)
from basemap.output_safety import atomic_write_new_json, refuse_existing


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--testbed", required=True)
    parser.add_argument("--testbed-seal", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dim", type=int, default=ROUND0005_DIMENSIONS)
    parser.add_argument("--n-holdout", type=int, default=ROUND0005_QUERY_ROWS)
    parser.add_argument("--seed", type=int, default=ROUND0005_QUERY_SEED)
    parser.add_argument("--convention",
                        help="optional expectation; must equal the built-in Round 0005 contract")
    parser.add_argument("--expectation-out")
    args = parser.parse_args(argv)
    convention = round0005_query_convention()
    if args.convention:
        with open(args.convention, encoding="utf-8") as handle:
            validate_round0005_query_convention(json.load(handle))
    report = build_query_artifact(
        testbed=args.testbed, source=args.source, out_dir=args.out_dir, dim=args.dim,
        n_holdout=args.n_holdout, seed=args.seed, convention=convention,
        testbed_seal_path=args.testbed_seal, production_contract=True)
    if args.expectation_out:
        if not os.path.realpath(args.expectation_out).startswith("/data/"):
            raise ValueError("expectation output must live under /data")
        refuse_existing(args.expectation_out, label="query expectation")
        atomic_write_new_json(args.expectation_out, convention, immutable=True)
    print(json.dumps({
        "manifest": report["manifest_path"],
        "manifest_sha256": report["manifest_sha256"],
        "identity_sha256": report["identity_sha256"],
        "shape": report["shape"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
