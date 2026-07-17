#!/usr/bin/env python3
"""CUDA-hidden Round 0013 observed-transform and 30M qualification driver."""

from __future__ import annotations

import argparse
import os

from basemap.minilm_input_pack_round0013 import (
    DEFAULT_BLOCK_ROWS,
    ROUND_ROOT,
    UPSTREAM_ROOT,
    assemble_round0013_capability,
    create_transform_execution_spec,
    full_round0013_read_only_reopen,
    record_intake,
    record_round0013_graph_provenance,
    reopen_round0013_capability,
    require_cuda_hidden,
    run_round0013_fixtures,
)


def _release_arguments(value: argparse.ArgumentParser) -> None:
    value.add_argument("--release-commit", required=True)
    value.add_argument("--release-root", required=True)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    subcommands = value.add_subparsers(dest="command", required=True)

    intake = subcommands.add_parser("intake")
    _release_arguments(intake)
    intake.add_argument("--root", default=str(ROUND_ROOT))
    intake.add_argument(
        "--codex-child-pid",
        type=int,
        default=int(os.environ.get("ROUNDWATCH_CODEX_CHILD_PID", "0")),
    )

    transform_spec = subcommands.add_parser("transform-spec")
    _release_arguments(transform_spec)
    transform_spec.add_argument("--root", default=str(ROUND_ROOT))

    fixtures = subcommands.add_parser("fixtures")
    _release_arguments(fixtures)
    fixtures.add_argument("--root", default=str(ROUND_ROOT))

    provenance = subcommands.add_parser("graph-provenance")
    provenance.add_argument("--release-commit", required=True)
    provenance.add_argument("--root", default=str(ROUND_ROOT))
    provenance.add_argument("--upstream-root", default=str(UPSTREAM_ROOT))

    reopen = subcommands.add_parser("full-reopen")
    reopen.add_argument("--release-commit", required=True)
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
    assemble.add_argument("--release-commit", required=True)
    assemble.add_argument("--root", default=str(ROUND_ROOT))
    assemble.add_argument("--upstream-root", default=str(UPSTREAM_ROOT))

    capability = subcommands.add_parser("reopen-capability")
    _release_arguments(capability)
    capability.add_argument("--root", default=str(ROUND_ROOT))
    capability.add_argument("--upstream-root", default=str(UPSTREAM_ROOT))
    return value


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    require_cuda_hidden()
    if arguments.command == "intake":
        if arguments.codex_child_pid <= 0:
            raise ValueError("--codex-child-pid must be positive")
        record_intake(
            arguments.release_commit,
            arguments.release_root,
            codex_child_pid=arguments.codex_child_pid,
            root=arguments.root,
        )
    elif arguments.command == "transform-spec":
        create_transform_execution_spec(
            arguments.release_commit, arguments.release_root, arguments.root
        )
    elif arguments.command == "fixtures":
        run_round0013_fixtures(
            arguments.release_commit, arguments.release_root, arguments.root
        )
    elif arguments.command == "graph-provenance":
        record_round0013_graph_provenance(
            arguments.release_commit, arguments.root, arguments.upstream_root
        )
    elif arguments.command == "full-reopen":
        full_round0013_read_only_reopen(
            arguments.release_commit,
            arguments.root,
            arguments.upstream_root,
            progress_path=arguments.progress_path,
            log_path=arguments.log_path,
            block_rows=arguments.block_rows,
            heavy_io_authorized=arguments.heavy_io_authorized,
        )
    elif arguments.command == "assemble":
        assemble_round0013_capability(
            arguments.release_commit, arguments.root, arguments.upstream_root
        )
    elif arguments.command == "reopen-capability":
        reopen_round0013_capability(
            arguments.release_commit,
            arguments.release_root,
            arguments.root,
            arguments.upstream_root,
        )
    else:  # pragma: no cover - argparse owns this boundary.
        raise AssertionError(arguments.command)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
