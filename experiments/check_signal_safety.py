#!/usr/bin/env python3
"""Pre-launch guard: no GPU child may be bounded by a mechanism that signals it.

Why this exists
---------------
`subprocess.run(..., timeout=N)` is implemented in CPython as `Popen.kill()` on
expiry (`subprocess.py`: `process.kill()` in the `TimeoutExpired` handler), and
`Popen.kill()` is `send_signal(SIGKILL)`. Wrapping a cuML/cuVS child in it
therefore arms the exact action that has twice wedged this box's GPU into an
unrecoverable UVM deadlock. It **fired** on R0238's imbalance grid against a
live cuML child holding a CUDA context; the machine survived by luck.

The guard the program trusted since R0216 — `check_undefined_names.py` — and
the per-round AST test in the CPU smokes both match on *call names*
(`os.kill`, `.kill`, `.terminate`, `.send_signal`). Delegating the kill into
CPython's `subprocess` module makes the call name `subprocess.run`, so both are
**structurally blind** to this class. This check matches on the *textual
argument span* instead, which is what they missed.

What it forbids
---------------
1. `SIGNALLING_TIMEOUT` — a `timeout=` keyword anywhere in the argument span of
   `subprocess.run` / `.call` / `.check_call` / `.check_output`. Every one of
   these kills the child on expiry.
2. `DIRECT_SIGNAL` — `os.kill`, `os.killpg`, `signal.alarm`, `.kill()`,
   `.terminate()`, `.send_signal(...)`, and shell-outs to `pkill` / `killall`.

`Popen.communicate(timeout=)` and `Popen.wait(timeout=)` are **not** hazards on
their own: both raise `TimeoutExpired` and leave the child running. They become
hazards only when the handler signals, which rule 2 catches.

Waivers
-------
A `SIGNALLING_TIMEOUT` on a child that provably cannot hold a CUDA context may
be waived by an explicit `# signal-safe: <reason>` comment inside the call span.
The waiver is a positive declaration, not a default: an unannotated call fails.

**A call whose span references a GPU launcher can never be waived.** That rule
has no escape hatch by construction — see `_GPU_MARKERS`.

Usage
-----
    python experiments/check_signal_safety.py experiments/round0239_nodes.py ...
    python experiments/check_signal_safety.py --inventory <paths...>

Exit status is 1 if any unwaived hazard is found (0 in `--inventory` mode,
which classifies rather than gates).
"""
from __future__ import annotations

import argparse
import ast
import io
import re
import sys
import tokenize


#: Textual markers that mean "this child may hold a CUDA context". A call span
#: containing any of these is UNWAIVABLE: the whole point of the check is that
#: no annotation may talk a cuML launch out of being a hazard.
_GPU_MARKERS = (
    "CUML_LAUNCHER",
    "cuml_py",
    "cuml",
    "cuvs",
    "cupy",
    "rapids",
    "torchrun",
    "BUILD_SCRIPT",
)

#: Markers used ONLY to classify a direct signal by its enclosing function.
#: Broader than `_GPU_MARKERS` because a supervisor function names the thing it
#: supervises in prose ("stop a build that holds a CUDA context") rather than by
#: importing it. This set is deliberately NOT used on a call span: the phrase
#: "never opens the CUDA driver" appears inside the legitimate `nvidia-smi`
#: waivers, and matching it there would make an NVML query unwaivable.
_GPU_CONTEXT_MARKERS = _GPU_MARKERS + (
    "CUDA",
    "cuda",
    "GPU child",
    "gpu_child",
    "build child",
    "builder",
)

#: Calls whose CPython implementation signals the child when `timeout` expires.
_SIGNALLING_CALLS = re.compile(
    r"(?<![\w.])(?:subprocess\s*\.\s*)?(run|call|check_call|check_output)\s*\(",
)

#: Unconditionally forbidden: these deliver a signal directly.
_DIRECT_SIGNALS = (
    (re.compile(r"(?<![\w.])os\s*\.\s*kill\s*\("), "os.kill"),
    (re.compile(r"(?<![\w.])os\s*\.\s*killpg\s*\("), "os.killpg"),
    (re.compile(r"(?<![\w.])signal\s*\.\s*alarm\s*\("), "signal.alarm"),
    (re.compile(r"\.\s*terminate\s*\("), ".terminate()"),
    (re.compile(r"\.\s*kill\s*\("), ".kill()"),
    (re.compile(r"\.\s*send_signal\s*\("), ".send_signal()"),
)

#: Shell-outs that signal by name rather than by pid.
_SIGNALLING_BINARIES = re.compile(r"""["'](pkill|killall|kill)["']""")

_WAIVER = re.compile(r"#\s*signal-safe:\s*(?P<reason>\S.*)")

_TIMEOUT_KWARG = re.compile(r"(?<![\w.])timeout\s*=")


class Finding:
    """One signalling call site."""

    def __init__(
        self,
        *,
        path: str,
        line: int,
        kind: str,
        snippet: str,
        gpu: bool,
        waiver: str | None,
    ) -> None:
        self.path = path
        self.line = line
        self.kind = kind
        self.snippet = snippet
        self.gpu = gpu
        self.waiver = waiver

    @property
    def fatal(self) -> bool:
        """A finding gates the launch unless it is a validly waived non-GPU one."""
        if self.kind == "DIRECT_SIGNAL":
            return True
        if self.gpu:
            # unwaivable by construction
            return True
        return self.waiver is None

    def classification(self) -> str:
        if self.gpu:
            return "GPU-CHILD (may hold a CUDA context)"
        return "non-GPU child"

    def __str__(self) -> str:
        state = "HAZARD" if self.fatal else f"waived: {self.waiver}"
        return (
            f"{self.path}:{self.line}: {self.kind} [{self.classification()}] "
            f"{state}\n      {self.snippet}"
        )


def blank_noncode(source: str) -> str:
    """Replace comments and string literals with spaces, preserving positions.

    Detection runs on this text so that the extensive prose in these modules —
    which discusses `os.kill` and `SIGKILL` at length — cannot produce a
    finding, and so that a generated child script embedded in an f-string is not
    mistaken for parent code. Line and column numbers are preserved exactly, so
    findings still point at the real source.
    """
    out = list(source)
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        spans = [
            (tok.start, tok.end)
            for tok in tokens
            if tok.type in (tokenize.COMMENT, tokenize.STRING)
        ]
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return source
    # Map (row, col) -> absolute offset once.
    line_offsets = [0]
    for line in source.splitlines(keepends=True):
        line_offsets.append(line_offsets[-1] + len(line))
    for (srow, scol), (erow, ecol) in spans:
        start = line_offsets[srow - 1] + scol
        end = line_offsets[erow - 1] + ecol
        for i in range(start, min(end, len(out))):
            if out[i] != "\n":
                out[i] = " "
    return "".join(out)


def _line_of(source: str, offset: int) -> int:
    return source.count("\n", 0, offset) + 1


def _span_end(blanked: str, open_paren: int) -> int:
    """Index just past the `)` matching the `(` at `open_paren`.

    Safe on blanked text: every string literal is gone, so parentheses balance.
    """
    depth = 0
    for i in range(open_paren, len(blanked)):
        if blanked[i] == "(":
            depth += 1
        elif blanked[i] == ")":
            depth -= 1
            if depth == 0:
                return i + 1
    return len(blanked)


def _waiver_in(original_span: str) -> str | None:
    match = _WAIVER.search(original_span)
    return match.group("reason").strip() if match else None


def _is_gpu(original_span: str) -> bool:
    return any(marker in original_span for marker in _GPU_MARKERS)


def _is_gpu_context(function_source: str) -> bool:
    return any(marker in function_source for marker in _GPU_CONTEXT_MARKERS)


def _enclosing_function_source(source: str, line: int) -> str:
    """Source of the innermost function containing `line`, or "" if none.

    R0239 correction: a bare `process.kill()` is an eight-character span with no
    argv in it, so span-local classification called `_terminate_cooperatively`'s
    last-resort `SIGKILL` — which the module's own docstring describes as
    stopping "a build that holds a CUDA context" — a *non-GPU* child. The span
    is the right context for a `timeout=` bound, because the launcher argv is
    literally inside the call; it is the wrong context for a direct signal,
    where the only evidence of what is being signalled is what the enclosing
    function supervises.

    This widening affects CLASSIFICATION ONLY. A `DIRECT_SIGNAL` is fatal either
    way, so no call site changes gate status because of it — it changes what the
    published inventory *says* about a call site, which is the thing R0239 owes.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ""
    best: str = ""
    best_span = None
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        end = getattr(node, "end_lineno", None)
        if end is None or not (node.lineno <= line <= end):
            continue
        span = end - node.lineno
        if best_span is None or span < best_span:
            best_span = span
            best = ast.get_source_segment(source, node) or ""
    return best


def _snippet(original_span: str) -> str:
    flat = " ".join(original_span.split())
    return flat if len(flat) <= 150 else flat[:147] + "..."


def scan(path: str) -> list[Finding]:
    with open(path, encoding="utf-8") as handle:
        source = handle.read()
    blanked = blank_noncode(source)
    findings: list[Finding] = []

    # rule 1 — a signalling timeout bound
    for match in _SIGNALLING_CALLS.finditer(blanked):
        open_paren = match.end() - 1
        end = _span_end(blanked, open_paren)
        if not _TIMEOUT_KWARG.search(blanked[open_paren:end]):
            continue
        original_span = source[match.start():end]
        line = _line_of(source, match.start())
        findings.append(Finding(
            path=path,
            line=line,
            kind="SIGNALLING_TIMEOUT",
            snippet=_snippet(original_span),
            # The launcher argv is inside the call, so the span is the whole
            # evidence and widening it would let a neighbouring cuML mention
            # make an unrelated `filefrag` call unwaivable.
            gpu=_is_gpu(original_span),
            # R0239 correction: the waiver must live INSIDE the call span. The
            # interrupted attempt searched `end + 200`, which let a waiver
            # comment attached to one call silently excuse the *next*,
            # unannotated call within 200 characters.
            waiver=_waiver_in(original_span),
        ))

    # rule 2 — a direct signal, never waivable
    for pattern, name in _DIRECT_SIGNALS:
        for match in pattern.finditer(blanked):
            end = _span_end(blanked, match.end() - 1)
            original_span = source[match.start():end]
            line = _line_of(source, match.start())
            findings.append(Finding(
                path=path,
                line=line,
                kind="DIRECT_SIGNAL",
                snippet=f"{name}: {_snippet(original_span)}",
                # A direct signal carries no argv. Classify it by what the
                # enclosing function supervises — see `_enclosing_function_source`.
                gpu=(
                    _is_gpu(original_span)
                    or _is_gpu_context(_enclosing_function_source(source, line))
                ),
                waiver=None,
            ))

    # rule 2b — signalling by binary name
    for match in _SIGNALLING_BINARIES.finditer(source):
        findings.append(Finding(
            path=path,
            line=_line_of(source, match.start()),
            kind="DIRECT_SIGNAL",
            snippet=f"shell signal binary {match.group(1)!r}",
            gpu=False,
            waiver=None,
        ))

    findings.sort(key=lambda f: f.line)
    return findings


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+")
    parser.add_argument(
        "--inventory", action="store_true",
        help="classify and print every finding; always exit 0",
    )
    args = parser.parse_args(argv[1:])

    fatal = 0
    total = 0
    for path in args.paths:
        findings = scan(path)
        total += len(findings)
        if not findings:
            print(f"{path}: CLEAN")
            continue
        for finding in findings:
            print(str(finding))
            fatal += bool(finding.fatal)
        if not any(f.fatal for f in findings):
            print(f"{path}: CLEAN (all {len(findings)} finding(s) waived)")

    print(
        f"\nchecked {len(args.paths)} file(s): {total} signalling call site(s), "
        f"{fatal} unwaived hazard(s)"
    )
    if args.inventory:
        return 0
    return 1 if fatal else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
