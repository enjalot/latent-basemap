#!/usr/bin/env python3
"""Cheap pre-launch guard: flag Load-context names a module never binds.

A NameError inside a long GPU node costs the whole node — R0216's first
correction died on one after six minutes of assembly, in the receipt seal at the
very end. `python -c "import module"` does not catch it, because the name is
only resolved when that line executes. This does, in milliseconds.

Deliberately conservative: it over-binds (any Store anywhere counts) so it
reports few false positives and is safe to gate a launch on.

    python experiments/check_undefined_names.py experiments/round0216_nodes.py
"""
from __future__ import annotations

import ast
import builtins
import sys


def undefined(path: str) -> list[str]:
    tree = ast.parse(open(path).read())
    bound = set(dir(builtins))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            bound.update(node.names)
    used = {
        n.id for n in ast.walk(tree)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    }
    return sorted(used - bound)


def main(argv: list[str]) -> int:
    bad = 0
    for path in argv[1:]:
        missing = undefined(path)
        print(f"{path}: {'CLEAN' if not missing else 'UNDEFINED -> ' + ', '.join(missing)}")
        bad += bool(missing)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
