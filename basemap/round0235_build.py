#!/usr/bin/env python3
"""R0235 — R0233's `cluster-spill-nnd` build, with ONE named constant rebound.

    cuml_py basemap/round0235_build.py --config c.json --out d

R0233's build script is the reviewed one: review-0233-01 verified its memmap
precondition on the substrate view *and* on every intermediate spill file, its
signal-free cooperative abort (`FlagWatchdog._trip` contains no `os.kill`,
`terminate()` or `kill()` path), and its OOM-as-measurement discipline. Copying
400 lines of it into a new file to change one number would put the machine-safety
path behind a diff, so this module **calls** it instead and rebinds exactly one
module global before delegating:

    `CLUSTER_CAPACITY_ROWS` — the largest realised cluster the child will accept
    before refusing itself after assignment.

That constant is a *guard*, and revising it is the whole subject of
review-0233-01 D1. R0233 pinned it at `5,204,724` rows, derived from a
deliberately steep two-point bound (`3400 B` per max-cluster row) chosen because
the `gd = 64` device law was unknown. The law has since been measured across a
`21.44x` span and independently reproduced, so R0235 derives the capacity from
the fitted law plus explicit margins in its ladder node and passes it in the
cell's `config.json`, where it is hash-bound like every other cell parameter and
is written into the build receipt as `cluster_capacity_rows`.

Nothing else is changed. This module installs no signal handler, sends no signal,
and does not touch any other name in `basemap.round0233_build`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import basemap.round0233_build as r0233_build  # noqa: E402
from basemap.round0233_substrate import (  # noqa: E402
    CLUSTER_CAPACITY_ROWS as R0233_CAPACITY,
)
from basemap.round0235_rung2 import Round0235Error  # noqa: E402


#: The value R0233 shipped. If this ever moves, the rebind below is no longer a
#: one-line delta against a reviewed path and this script fails closed.
EXPECTED_R0233_CAPACITY_ROWS = 5_204_724


def rebind_capacity(rows: int) -> int:
    """Rebind the one guard constant, fail-closed, and report the delta."""
    if int(R0233_CAPACITY) != EXPECTED_R0233_CAPACITY_ROWS:
        raise Round0235Error(
            f"R0233 cluster capacity is {int(R0233_CAPACITY)}, not the "
            f"{EXPECTED_R0233_CAPACITY_ROWS} this rebind was registered against"
        )
    if int(r0233_build.CLUSTER_CAPACITY_ROWS) != EXPECTED_R0233_CAPACITY_ROWS:
        raise Round0235Error(
            "R0233 build script no longer holds the registered capacity constant"
        )
    rows = int(rows)
    if rows <= 0:
        raise Round0235Error(f"R0235 cluster capacity {rows} is not positive")
    r0233_build.CLUSTER_CAPACITY_ROWS = rows
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args, _rest = parser.parse_known_args(argv)
    with open(args.config, encoding="utf-8") as handle:
        config = json.load(handle)
    capacity = config.get("cluster_capacity_rows")
    if capacity is None:
        raise Round0235Error(
            "R0235 cell config must carry cluster_capacity_rows; the capacity is "
            "derived from the fitted device law in the ladder node so that the "
            "guard and the per-rung derivation cannot disagree (D1)"
        )
    rebind_capacity(int(capacity))
    return r0233_build.main(["--config", args.config, "--out", args.out])


if __name__ == "__main__":
    raise SystemExit(main())
