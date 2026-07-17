#!/usr/bin/env python3
"""CUDA-hidden Round 0012 input-pack correction/qualification entry point."""

from __future__ import annotations

import argparse

from basemap.minilm_input_pack_round0012 import (
    DEFAULT_BLOCK_ROWS,
    ROUND_ROOT,
    UPSTREAM_ROOT,
    assemble_round0012_capability,
    full_read_only_reopen,
    record_graph_provenance,
    reopen_round0012_capability,
    require_cuda_hidden,
    run_round0012_fixtures,
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    subcommands = value.add_subparsers(dest="command", required=True)

    fixtures = subcommands.add_parser("fixtures")
    fixtures.add_argument("--root", default=str(ROUND_ROOT))

    provenance = subcommands.add_parser("graph-provenance")
    provenance.add_argument("--root", default=str(ROUND_ROOT))
    provenance.add_argument("--upstream-root", default=str(UPSTREAM_ROOT))

    reopen = subcommands.add_parser("full-reopen")
    reopen.add_argument("--root", default=str(ROUND_ROOT))
    reopen.add_argument("--upstream-root", default=str(UPSTREAM_ROOT))
    reopen.add_argument(
        "--progress-path", default=str(ROUND_ROOT / "management" / "progress.json")
    )
    reopen.add_argument(
        "--log-path", default=str(ROUND_ROOT / "logs" / "full-read-only-reopen.log")
    )
    reopen.add_argument("--block-rows", type=int, default=DEFAULT_BLOCK_ROWS)
    reopen.add_argument("--heavy-io-authorized", action="store_true")

    assemble = subcommands.add_parser("assemble")
    assemble.add_argument("--root", default=str(ROUND_ROOT))
    assemble.add_argument("--upstream-root", default=str(UPSTREAM_ROOT))

    capability = subcommands.add_parser("reopen-capability")
    capability.add_argument("--root", default=str(ROUND_ROOT))
    capability.add_argument("--upstream-root", default=str(UPSTREAM_ROOT))
    return value


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    require_cuda_hidden()
    if arguments.command == "fixtures":
        run_round0012_fixtures(arguments.root)
    elif arguments.command == "graph-provenance":
        record_graph_provenance(arguments.root, arguments.upstream_root)
    elif arguments.command == "full-reopen":
        full_read_only_reopen(
            arguments.root,
            arguments.upstream_root,
            progress_path=arguments.progress_path,
            log_path=arguments.log_path,
            block_rows=arguments.block_rows,
            heavy_io_authorized=arguments.heavy_io_authorized,
        )
    elif arguments.command == "assemble":
        assemble_round0012_capability(arguments.root, arguments.upstream_root)
    elif arguments.command == "reopen-capability":
        reopen_round0012_capability(arguments.root, arguments.upstream_root)
    else:  # pragma: no cover - argparse owns this boundary.
        raise AssertionError(arguments.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
