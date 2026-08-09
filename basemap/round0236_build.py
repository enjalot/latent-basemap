#!/usr/bin/env python3
"""R0236 — the reviewed `cluster-spill-nnd` build, reached without a diff.

    cuml_py basemap/round0236_build.py --config c.json --out d

The machine-safety path in this program has cost two GPU wedges to get right, so
it is not copied here. This module **calls** `basemap/round0235_build.py`, which
in turn rebinds exactly one named constant on `basemap/round0233_build.py` and
delegates to it. Both wrappers were verified by independent review
(review-0235-01: `assert_memmap_for_cuvs` on the substrate view and on every
intermediate spill file, `FlagWatchdog._trip` containing no `os.kill`,
`terminate()` or `kill()` path, and the OOM-as-measurement discipline).

So the executable safety path at this rung is byte-identical to the one review
verified at 12.5M, and the only R0236-specific code in the child process is the
fail-closed assertion below that R0235's own registered constant has not moved.

This module installs no signal handler, sends no signal, and touches no name in
`basemap.round0233_build` other than through R0235's registered rebind.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import basemap.round0235_build as r0235_build  # noqa: E402
from basemap.round0236_rung3 import Round0236Error  # noqa: E402


#: The value R0235 registered its rebind against. If R0233's constant ever moves,
#: R0235's wrapper is no longer a one-line delta against a reviewed path and this
#: script must fail closed before any CUDA context exists.
EXPECTED_R0235_EXPECTED_CAPACITY_ROWS = 5_204_724


def assert_reviewed_build_path() -> dict[str, int]:
    """Fail closed unless R0235's wrapper is the one review verified."""
    observed = int(r0235_build.EXPECTED_R0233_CAPACITY_ROWS)
    if observed != EXPECTED_R0235_EXPECTED_CAPACITY_ROWS:
        raise Round0236Error(
            f"R0235's build wrapper expects an R0233 capacity of {observed}, not "
            f"the {EXPECTED_R0235_EXPECTED_CAPACITY_ROWS} this delegation was "
            "registered against; the reviewed build path has moved"
        )
    return {"r0235_expected_r0233_capacity_rows": observed}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args, _rest = parser.parse_known_args(argv)
    assert_reviewed_build_path()
    with open(args.config, encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("cluster_capacity_rows") is None:
        raise Round0236Error(
            "R0236 cell config must carry cluster_capacity_rows; the capacity is "
            "derived from the fitted device law in the ladder node so that the "
            "launch guard and the per-rung derivation cannot disagree "
            "(review-0233-01 D1)"
        )
    return r0235_build.main(["--config", args.config, "--out", args.out])


if __name__ == "__main__":
    raise SystemExit(main())
