#!/usr/bin/env python3
"""R0237 — the reviewed `cluster-spill-nnd` build, reached without a diff.

    cuml_py basemap/round0237_build.py --config c.json --out d

The machine-safety path in this program has cost two GPU wedges to get right, so
it is not copied here. This module **calls** `basemap/round0236_build.py`, which
calls `basemap/round0235_build.py`, which rebinds exactly one named constant on
`basemap/round0233_build.py` and delegates to it. Every wrapper in that chain is
fail-closed and was verified by independent review (review-0235-01:
`assert_memmap_for_cuvs` on the substrate view and on every intermediate spill
file, `FlagWatchdog._trip` containing no `os.kill`, `terminate()` or `kill()`
path, and the OOM-as-measurement discipline).

So the executable safety path at this rung is byte-identical to the one review
verified at 12.5M and R0236 ran at 25M, and the only R0237-specific code in the
child process is the fail-closed assertion below that R0236's own registered
constant has not moved.

This module installs no signal handler, sends no signal, and touches no name in
`basemap.round0233_build` other than through the registered chain of rebinds.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import basemap.round0236_build as r0236_build  # noqa: E402
from basemap.round0237_rung4 import Round0237Error  # noqa: E402


#: The value R0236 registered its delegation against. If R0235's constant ever
#: moves, R0236's wrapper is no longer a checked delegation onto a reviewed path
#: and this script must fail closed before any CUDA context exists.
EXPECTED_R0236_EXPECTED_CAPACITY_ROWS = 5_204_724


def assert_reviewed_build_path() -> dict[str, int]:
    """Fail closed unless R0236's wrapper is the one that ran at 25M."""
    observed = int(r0236_build.EXPECTED_R0235_EXPECTED_CAPACITY_ROWS)
    if observed != EXPECTED_R0236_EXPECTED_CAPACITY_ROWS:
        raise Round0237Error(
            f"R0236's build wrapper expects an R0235 capacity of {observed}, not "
            f"the {EXPECTED_R0236_EXPECTED_CAPACITY_ROWS} this delegation was "
            "registered against; the reviewed build path has moved"
        )
    checked = r0236_build.assert_reviewed_build_path()
    return {
        "r0236_expected_r0235_capacity_rows": observed,
        **{str(key): int(value) for key, value in checked.items()},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args, _rest = parser.parse_known_args(argv)
    assert_reviewed_build_path()
    with open(args.config, encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("cluster_capacity_rows") is None:
        raise Round0237Error(
            "R0237 cell config must carry cluster_capacity_rows; the capacity is "
            "derived from the fitted device law in the ladder node so that the "
            "launch guard and the per-rung derivation cannot disagree "
            "(review-0233-01 D1)"
        )
    return r0236_build.main(["--config", args.config, "--out", args.out])


if __name__ == "__main__":
    raise SystemExit(main())
