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
1. `SIGNALLING_TIMEOUT` — a `timeout=` bound on `subprocess.run` / `.call` /
   `.check_call` / `.check_output`. Every one of these kills the child on expiry.
2. `DIRECT_SIGNAL` — `os.kill`, `os.killpg`, `signal.alarm`, `.kill()`,
   `.terminate()`, `.send_signal(...)`, and shell-outs to `pkill` / `killall`.

`Popen.communicate(timeout=)` and `Popen.wait(timeout=)` are **not** hazards on
their own: both raise `TimeoutExpired` and leave the child running. They become
hazards only when the handler signals, which rule 2 catches.

How it looks (R0240 hardening)
------------------------------
The original detector was a pure regex whose only known callee prefix was the
literal string `subprocess`, so every alias escaped it: review-0239-01/F1
planted twelve evasions and it caught four. Detection is now **AST-first**, with
the regexes kept as a backstop so a file that fails to parse is still scanned.
The AST pass resolves the callee through the module's own bindings and closes:

* aliased modules and functions — `import subprocess as sp` / `import os as _o`,
  `from subprocess import run as _launch`, and `sp = subprocess` /
  `_launch = subprocess.run` rebindings;
* a `timeout` delivered through `**kwargs` from a dict literal;
* backslash line continuations between the module and the attribute;
* GNU coreutils `timeout(1)` (`timeout -s KILL`, `timeout -k`, `/usr/bin/timeout`)
  appearing as `argv[0]` of *any* launch — it signals on expiry with no
  `timeout=` keyword anywhere in the call;
* `functools.partial(subprocess.run, timeout=…)`, including
  `from functools import partial as p`;
* `os.system` / `os.popen` shell-outs whose command mentions
  `kill` / `pkill` / `killall`;
* `getattr(proc, <signal method>)()` — the method named as a split literal
  (which the AST folds back together) or held in a local variable;
* signalling constructs that appear only inside a **generated child script**.
  These modules build child scripts as string literals; `blank_noncode()`
  erases them by design, so any literal that parses as Python and contains a
  call is re-scanned as code, reported at the literal's own line.

`timeout=None` arms nothing and is **not** reported (only the literal `None`; a
variable that might be `None` is still a hit). Findings are deduplicated per
`(path, kind, call site)`, keeping the most specific name — `os.kill(` used to
be emitted twice, once by the `os.kill` rule and once by the generic `.kill(`
rule, inflating every published tally (review-0239-01/F2).

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
import os.path
import re
import sys
import textwrap
import tokenize
import warnings


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
#: Any dotted prefix is accepted (`[\s\\]*` also spans a backslash continuation)
#: because this regex is only the backstop for a file the AST pass could not
#: parse, where no binding information is available at all.
_SIGNALLING_CALLS = re.compile(
    r"(?<![\w.])(?:[\w.]+[\s\\]*\.[\s\\]*)?(run|call|check_call|check_output)\s*\(",
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

#: GNU coreutils `timeout(1)` in argv[0] position: it sends SIGTERM (or
#: whatever `-s` names, and SIGKILL after `-k`) when the bound expires, so it is
#: a signalling bound even though the call carries no `timeout=` keyword.
_TIMEOUT_BINARY_ARG = re.compile(r"""["'](?:[\w./+-]*/)?timeout["']\s*,""")

#: Words that turn an `os.system` / `os.popen` shell-out into a signal.
_KILL_WORDS = re.compile(r"(?<![\w.-])(pkill|killall|kill)(?![\w-])")

_WAIVER = re.compile(r"#\s*signal-safe:\s*(?P<reason>\S.*)")

#: `timeout=None` arms nothing, so the negative lookahead excludes it. Only the
#: literal keyword: `timeout=maybe_none` stays a hit, because the detector
#: cannot know what the name holds.
_TIMEOUT_KWARG = re.compile(r"(?<![\w.])timeout\s*=\s*(?!None\b)")

#: Callees whose `timeout=` kills the child.
_SIGNALLING_FUNCS = frozenset({
    "subprocess.run",
    "subprocess.call",
    "subprocess.check_call",
    "subprocess.check_output",
})

#: Callees that signal directly, resolved through the module's own bindings.
_DIRECT_FUNCS = {
    "os.kill": "os.kill",
    "os.killpg": "os.killpg",
    "signal.alarm": "signal.alarm",
}

#: Methods that deliver a signal to a `Popen`, taken from the textual rules
#: above so the two can never drift (and so this module contains no bare
#: quoted signal name of its own, which its own rule 2b would flag).
_SIGNAL_METHODS = frozenset(
    name.strip(".()") for _, name in _DIRECT_SIGNALS if name.startswith(".")
)

#: Shell-outs: a signal delivered by a command line rather than by a syscall.
_SHELL_FUNCS = frozenset({"os.system", "os.popen"})

_PARTIAL_FUNCS = frozenset({"functools.partial"})

#: How deep to follow generated child scripts embedded as string literals.
_MAX_EMBED_DEPTH = 2

#: Ranking used when the same call site is reported by more than one rule: the
#: most specific name wins (`os.kill` over the generic `.kill()`).
_PRIORITY_REGEX = 1
_PRIORITY_AST_METHOD = 2
_PRIORITY_AST_QUALIFIED = 3


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
        key: object | None = None,
        priority: int = _PRIORITY_REGEX,
    ) -> None:
        self.path = path
        self.line = line
        self.kind = kind
        self.snippet = snippet
        self.gpu = gpu
        self.waiver = waiver
        #: Identity of the *call site*, used to collapse the several rules that
        #: legitimately match the same call into one finding. Two findings with
        #: the same `(path, kind, key)` are the same hazard seen twice.
        self.key = key
        self.priority = priority

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


# --------------------------------------------------------------------------- #
# AST pass: resolve the callee through the module's own bindings
# --------------------------------------------------------------------------- #
class _Bindings:
    """What the names in one module actually refer to.

    The regex only ever knew the literal prefix `subprocess`. This is the whole
    fix for the alias family of evasions: `sp`, `_launch`, `_o` and friends are
    resolved back to their canonical dotted name before any rule looks at them.
    """

    def __init__(self) -> None:
        self.modules: dict[str, str] = {}   # local name -> module ("subprocess")
        self.funcs: dict[str, str] = {}     # local name -> "subprocess.run"
        self.dicts: dict[str, ast.Dict] = {}  # local name -> dict literal
        self.strings: dict[str, str] = {}   # local name -> str literal


def _collect_bindings(tree: ast.AST) -> _Bindings:
    binds = _Bindings()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                local = alias.asname or alias.name.split(".")[0]
                binds.modules[local] = alias.name if alias.asname else local
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                local = alias.asname or alias.name
                binds.funcs[local] = f"{node.module}.{alias.name}"
    # A second pass, so plain assignments can lean on the imports above.
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        value = node.value
        if isinstance(value, ast.Dict):
            binds.dicts[target.id] = value
        elif isinstance(value, ast.Constant) and isinstance(value.value, str):
            binds.strings[target.id] = value.value
        elif isinstance(value, ast.Name) and value.id in binds.modules:
            binds.modules[target.id] = binds.modules[value.id]
        elif isinstance(value, (ast.Name, ast.Attribute)):
            resolved = _resolve(value, binds)
            if resolved and "." in resolved:
                binds.funcs[target.id] = resolved
    return binds


def _resolve(node: ast.AST, binds: _Bindings) -> str | None:
    """Canonical dotted name for a callee expression, or None."""
    if isinstance(node, ast.Name):
        return binds.funcs.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        base = node.value
        if isinstance(base, ast.Name):
            module = binds.modules.get(base.id)
            if module:
                return f"{module}.{node.attr}"
            qualified = binds.funcs.get(base.id, base.id)
            return f"{qualified}.{node.attr}"
        inner = _resolve(base, binds)
        return f"{inner}.{node.attr}" if inner else None
    return None


def _line_starts(source: str) -> list[int]:
    starts = [0]
    for line in source.splitlines(keepends=True):
        starts.append(starts[-1] + len(line))
    return starts


def _offset_of(source: str, starts: list[int], lineno: int, col: int) -> int:
    """Absolute character offset of an AST position.

    `col_offset` counts UTF-8 *bytes*, and these modules are full of em dashes,
    so the prefix is decoded rather than added.
    """
    if lineno < 1 or lineno >= len(starts):
        return 0
    line = source[starts[lineno - 1]:starts[lineno]]
    prefix = line.encode("utf-8")[:col].decode("utf-8", "ignore")
    return starts[lineno - 1] + len(prefix)


def _call_paren(blanked: str, func_end: int) -> int:
    """Offset of the `(` that opens a call whose callee ends at `func_end`.

    This is the call-site identity used for deduplication: every rule that can
    match a given call — regex or AST, specific or generic — lands on the same
    parenthesis.
    """
    index = blanked.find("(", func_end)
    return index if index != -1 else func_end


def _timeout_arming(call: ast.Call, binds: _Bindings) -> str | None:
    """How this call arms a signalling bound, or None if it does not.

    `timeout=None` arms nothing: CPython's `Popen.communicate(timeout=None)`
    waits forever, so no `kill()` is ever reached. Reporting it was a false
    positive (review-0239-01).
    """
    for keyword in call.keywords:
        if keyword.arg == "timeout":
            if _is_literal_none(keyword.value):
                continue
            return "timeout="
        if keyword.arg is None:
            mapping = keyword.value
            if isinstance(mapping, ast.Name):
                mapping = binds.dicts.get(mapping.id)
            if not isinstance(mapping, ast.Dict):
                continue
            for key, value in zip(mapping.keys, mapping.values):
                if not (isinstance(key, ast.Constant) and key.value == "timeout"):
                    continue
                if _is_literal_none(value):
                    continue
                return "**{'timeout': ...}"
    return None


def _is_literal_none(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


def _timeout_binary_arg(call: ast.Call) -> ast.AST | None:
    """The argv node whose command is GNU `timeout(1)`, if any."""
    candidates = list(call.args) + [kw.value for kw in call.keywords if kw.arg == "args"]
    for arg in candidates:
        if isinstance(arg, (ast.List, ast.Tuple)) and arg.elts:
            head = arg.elts[0]
            if isinstance(head, ast.Constant) and _is_timeout_binary(head.value):
                return arg
        elif isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            head = arg.value.strip().split(" ")[0] if arg.value.strip() else ""
            if _is_timeout_binary(head):
                return arg
    return None


def _is_timeout_binary(value: object) -> bool:
    return isinstance(value, str) and os.path.basename(value.strip()) == "timeout"


def _getattr_signal_name(call: ast.Call, binds: _Bindings) -> str | None:
    """The signal method named by a `getattr(obj, <name>)`-style lookup.

    Python folds adjacent string literals, so a deliberately split spelling of a
    signal method arrives here already joined — which is exactly why this is an
    AST rule and not a textual one.
    """
    if len(call.args) < 2:
        return None
    attribute = call.args[1]
    name: str | None = None
    if isinstance(attribute, ast.Constant) and isinstance(attribute.value, str):
        name = attribute.value
    elif isinstance(attribute, ast.Name):
        name = binds.strings.get(attribute.id)
    return name if name in _SIGNAL_METHODS else None


def _ast_findings(
    path: str, source: str, blanked: str, depth: int,
) -> list[Finding]:
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return []   # the regex backstop still ran
    starts = _line_starts(source)
    binds = _collect_bindings(tree)
    findings: list[Finding] = []

    def emit(node: ast.AST, paren: int, kind: str, name: str, priority: int) -> None:
        start = _offset_of(source, starts, node.lineno, node.col_offset)
        end = _offset_of(
            source, starts,
            getattr(node, "end_lineno", node.lineno) or node.lineno,
            getattr(node, "end_col_offset", 0) or 0,
        )
        span = source[start:max(end, start)]
        gpu = _is_gpu(span)
        if kind == "DIRECT_SIGNAL" and not gpu:
            gpu = _is_gpu_context(_enclosing_function_source(source, node.lineno))
        findings.append(Finding(
            path=path,
            line=node.lineno,
            kind=kind,
            snippet=f"{name}: {_snippet(span)}",
            gpu=gpu,
            waiver=_waiver_in(span) if kind == "SIGNALLING_TIMEOUT" else None,
            key=("call", paren),
            priority=priority,
        ))

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _resolve(node.func, binds)
        func_end = _offset_of(
            source, starts,
            getattr(node.func, "end_lineno", node.lineno) or node.lineno,
            getattr(node.func, "end_col_offset", 0) or 0,
        )
        paren = _call_paren(blanked, func_end)

        if name in _SIGNALLING_FUNCS:
            arming = _timeout_arming(node, binds)
            if arming:
                emit(node, paren, "SIGNALLING_TIMEOUT",
                     f"{name}({arming})", _PRIORITY_AST_QUALIFIED)
        elif name in _PARTIAL_FUNCS and node.args:
            inner = _resolve(node.args[0], binds)
            if inner in _SIGNALLING_FUNCS and _timeout_arming(node, binds):
                emit(node, paren, "SIGNALLING_TIMEOUT",
                     f"partial({inner}, timeout=)", _PRIORITY_AST_QUALIFIED)

        argv = _timeout_binary_arg(node)
        if argv is not None:
            emit(argv, _offset_of(source, starts, argv.lineno, argv.col_offset),
                 "SIGNALLING_TIMEOUT", "GNU timeout(1) argv",
                 _PRIORITY_AST_QUALIFIED)

        if name in _DIRECT_FUNCS:
            emit(node, paren, "DIRECT_SIGNAL", _DIRECT_FUNCS[name],
                 _PRIORITY_AST_QUALIFIED)
        elif name in _SHELL_FUNCS and _KILL_WORDS.search(ast.unparse(node)):
            emit(node, paren, "DIRECT_SIGNAL", f"{name} shell signal",
                 _PRIORITY_AST_QUALIFIED)
        elif (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in _SIGNAL_METHODS
        ):
            emit(node, paren, "DIRECT_SIGNAL", f".{node.func.attr}()",
                 _PRIORITY_AST_METHOD)
        elif name == "getattr":
            attribute = _getattr_signal_name(node, binds)
            if attribute:
                emit(node, paren, "DIRECT_SIGNAL",
                     f'getattr(..., "{attribute}")', _PRIORITY_AST_QUALIFIED)

    if depth < _MAX_EMBED_DEPTH:
        findings.extend(_embedded_findings(path, tree, depth))
    return findings


# --------------------------------------------------------------------------- #
# generated child scripts: code that only exists inside a string literal
# --------------------------------------------------------------------------- #
def _string_literals(tree: ast.AST) -> list[tuple[ast.AST, str]]:
    """Every string literal in the module, f-strings reconstructed.

    A `FormattedValue` is replaced by its own source (`{script!r}` becomes
    `(script)`) rather than by a placeholder, so the reconstruction keeps both
    its syntax *and* any GPU marker the interpolation carries.
    """
    consumed: set[int] = set()
    literals: list[tuple[ast.AST, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.JoinedStr) or id(node) in consumed:
            continue
        pieces = []
        for value in node.values:
            if isinstance(value, ast.Constant):
                pieces.append(str(value.value))
            else:
                try:
                    pieces.append(f"({ast.unparse(value.value)})")
                except Exception:      # pragma: no cover - defensive
                    pieces.append("(None)")
        for sub in ast.walk(node):
            consumed.add(id(sub))
        literals.append((node, "".join(pieces)))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in consumed
        ):
            literals.append((node, node.value))
    return literals


def _embedded_findings(path: str, tree: ast.AST, depth: int) -> list[Finding]:
    findings: list[Finding] = []
    for node, text in _string_literals(tree):
        if "(" not in text or len(text) < 12:
            continue
        for candidate in (text, textwrap.dedent(text)):
            try:
                with warnings.catch_warnings():
                    # a regex literal inside a child script is not our problem
                    warnings.simplefilter("ignore", SyntaxWarning)
                    inner_tree = ast.parse(candidate)
            except (SyntaxError, ValueError):
                continue
            if not any(isinstance(n, ast.Call) for n in ast.walk(inner_tree)):
                break
            for finding in _scan_text(path, candidate, depth + 1):
                finding.line = node.lineno + finding.line - 1
                finding.key = ("embedded", node.lineno, finding.key)
                finding.snippet = f"in generated child script: {finding.snippet}"
                findings.append(finding)
            break
    return findings


# --------------------------------------------------------------------------- #
# the scan itself
# --------------------------------------------------------------------------- #
def _regex_findings(path: str, source: str, blanked: str) -> list[Finding]:
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
            key=("call", open_paren),
        ))

    # rule 1b — GNU `timeout(1)` as argv[0]: the bound signals with no `timeout=`
    for match in _TIMEOUT_BINARY_ARG.finditer(source):
        start, end = _enclosing_brackets(blanked, match.start())
        original_span = source[start:end]
        findings.append(Finding(
            path=path,
            line=_line_of(source, match.start()),
            kind="SIGNALLING_TIMEOUT",
            snippet=f"GNU timeout(1) argv: {_snippet(original_span)}",
            gpu=_is_gpu(original_span),
            waiver=_waiver_in(original_span),
            key=("call", start),
        ))

    # rule 2 — a direct signal, never waivable
    for pattern, name in _DIRECT_SIGNALS:
        for match in pattern.finditer(blanked):
            open_paren = match.end() - 1
            end = _span_end(blanked, open_paren)
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
                # `os.kill(` matches both the `os.kill` rule and the generic
                # `.kill(` rule. Both land on the same parenthesis, so keying on
                # it collapses them to one finding (review-0239-01/F2).
                key=("call", open_paren),
                priority=(
                    _PRIORITY_AST_QUALIFIED if "." in name.strip(".()")
                    else _PRIORITY_REGEX
                ),
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
            key=("binary", match.start()),
        ))

    return findings


def _enclosing_brackets(blanked: str, index: int) -> tuple[int, int]:
    """Span of the innermost `[`/`(` enclosing `index`, or the whole line.

    Safe on blanked text, where every string literal is gone so brackets
    balance. Used to give the argv-level backstop rules a span wide enough to
    carry a GPU marker or a waiver.
    """
    depth = 0
    for start in range(index, -1, -1):
        char = blanked[start]
        if char in ")]}":
            depth += 1
        elif char in "([{":
            if depth == 0:
                return start, _span_end_of(blanked, start)
            depth -= 1
    line_start = blanked.rfind("\n", 0, index) + 1
    line_end = blanked.find("\n", index)
    return line_start, len(blanked) if line_end == -1 else line_end


def _span_end_of(blanked: str, opener: int) -> int:
    """Index just past the bracket matching the one at `opener`."""
    pairs = {"(": ")", "[": "]", "{": "}"}
    close = pairs[blanked[opener]]
    depth = 0
    for i in range(opener, len(blanked)):
        if blanked[i] == blanked[opener]:
            depth += 1
        elif blanked[i] == close:
            depth -= 1
            if depth == 0:
                return i + 1
    return len(blanked)


def _scan_text(path: str, source: str, depth: int = 0) -> list[Finding]:
    blanked = blank_noncode(source)
    findings: list[Finding] = []
    if depth == 0:
        # The textual backstop runs on real files only. Inside a generated
        # child script the AST rules are enough, and the regexes would fire on
        # the surrounding prose of whatever built the string.
        findings.extend(_regex_findings(path, source, blanked))
    findings.extend(_ast_findings(path, source, blanked, depth))
    return findings


def _dedupe(findings: list[Finding]) -> list[Finding]:
    """One finding per `(path, kind, call site)`, keeping the most specific.

    Several rules legitimately match the same call — `os.kill(` is both an
    `os.kill` and a `.kill(`, and every AST hit is normally also a regex hit.
    Emitting each of them separately inflated the `unwaived hazard(s)` tally and
    every inventory built from it (review-0239-01/F2). Ties are broken towards
    the *worse* verdict first, so deduplication can never lose a hazard.
    """
    best: dict[tuple, Finding] = {}
    for finding in findings:
        identity = (finding.path, finding.kind, finding.key)
        current = best.get(identity)
        rank = (finding.gpu, finding.fatal, finding.priority)
        if current is None or rank > (current.gpu, current.fatal, current.priority):
            best[identity] = finding
    kept = list(best.values())
    # A quoted signal name inside a call that another rule already reported is
    # the same signal seen twice (the `getattr` form), not a second one. Same
    # for a construct inside a generated child script that the textual rules
    # also happened to see: CPython 3.12 stopped emitting f-string bodies as
    # `STRING` tokens, so `blank_noncode()` no longer hides them.
    covered: dict[tuple, Finding] = {
        (f.path, f.kind, f.line): f for f in kept if _key_kind(f.key) == "call"
    }
    survivors = []
    for finding in kept:
        if _key_kind(finding.key) == "call":
            survivors.append(finding)
            continue
        winner = covered.get((finding.path, finding.kind, finding.line))
        if winner is None:
            survivors.append(finding)
        else:
            # never lose the worse classification while collapsing
            winner.gpu = winner.gpu or finding.gpu
    return survivors


def _key_kind(key: object) -> str:
    return key[0] if isinstance(key, tuple) and key else ""


def scan(path: str) -> list[Finding]:
    with open(path, encoding="utf-8") as handle:
        source = handle.read()
    findings = _dedupe(_scan_text(path, source, depth=0))
    findings.sort(key=lambda f: (f.line, f.kind, f.snippet))
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
