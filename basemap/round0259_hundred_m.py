"""R0259 — the 100M rung gets a loader, an entry, and a bitwise-chunkable sum.

review-0258-01 §H.1 established the thing this module exists to fix:

    `round0113_nodes.run_train` is not a 100M trainer that lacks a loader; it is
    a ~2M, k=50, 768-d, R0113-schema trainer. Putting an `AbortPollGate` on it
    does not make a 100M train stoppable, because it cannot run one.

So R0250–R0258 hardened a code path that cannot run the target job. The
engineering transfers; it was applied one layer too early. This module supplies
the layer underneath it.

The incompatibilities, enumerated
---------------------------------
review-0258-01 §H.1 says *"the container mismatch is the last of four
incompatibilities, and the first three fire before it"* while its own table
lists **four** manifest checks. Counted from the source there are **five**
distinct blocking mismatches between `round0113_nodes.run_train` and R0243's
100M graph, and every one of them fires before a single edge is read:

===  ==============================  ===========================  ==========================
#    check (`round0113_prompt_contrast.load_graph`)              R0113 requires / R0243 has
===  ==============================  ===========================  ==========================
I1   `manifest["schema"]`            `round0113-prompt-arm-      `round0243-minilm-mixed-
                                     fuzzy-graph-v1`             100000k-k15-fuzzy-and-
                                                                 canonical-tripwire-v1`
I2   `manifest["retained_rows"]`     `1_993_761`                 `100_000_000` (under `rows`)
I3   `manifest["k"]`                 `50`                        `15`
I4   `manifest["dimension"]`         `768`                       `384` (absent; the substrate
                                                                 carries it)
I5   `load_edge_arrays` container    one bulk `.npz` with a      three streamed `.npy`
                                     `sources` member            members + a scalar header
===  ==============================  ===========================  ==========================

`I1`–`I4` raise `Round0113Error` inside `load_graph`; `I5` raises `KeyError`,
`IndexError` or `IsADirectoryError` inside `load_edge_arrays` depending on which
100M path is handed to it. R0113's own contract is *correct for R0113* — none of
it may be widened, because a 2M config that silently ran a 100M path (or the
reverse) would mis-train in a way no acceptance rule in this program checks.

What this module ships instead
------------------------------
1. **A rung registry.** Two rungs, `r0113-2m` and `r0243-100m`, each declaring
   schema, rows, `k`, dimension and container. `rung_of()` resolves a manifest to
   exactly one, and `assert_rung()` is the applicability check every entry calls
   on its first line. A manifest that matches no rung, or matches a schema whose
   rows/`k` disagree, raises — it does not fall through to a default.
2. **A loader branch** (`streamed_member_layout`, `load_streamed_edge_members`)
   that opens R0243's three `.npy` members as read-only memmaps and reads
   `n_nodes`/`k`/`directed_edges` from the header `.npz`. `load_edge_arrays`
   dispatches to it on a **content** discriminator, never on a filename, so the
   bulk-`.npz` path every rung at or below 50M uses is byte-for-byte unchanged.
3. **A chunked `np.sum` that is bitwise identical** (`pairwise_sum_polled`).
   review-0258-01 §E falsified R0258's "irreducible unpollable residue" by
   replicating numpy's own pairwise split, `n2 = n//2 − n//2 % 8`. The reviewer
   could not verify that rule across numpy versions, so this module does not
   rely on it silently: `assert_pairwise_rule()` re-proves bitwise identity at
   startup over six sizes and four leaf sizes and **raises** if the rule has
   moved, and the numpy version it was verified against is recorded in the
   receipt.
4. **A residency classifier that does not use `isinstance`.** review-0258-01
   §H.3: `HostStreamEdgeSampler._src_h` is file-backed *without being an
   `np.memmap` instance*, because `ascontiguousarray` returns a base `ndarray`
   view over the mapping — so R0230's `isinstance np.memmap` reader rule
   mis-classifies exactly the object that most needs classifying. `file_backed()`
   walks the `.base` chain instead, and `endpoint_residency()` reports both
   verdicts side by side.
5. **A strengthened chunk-loop guard.** review-0258-01 §D.3 planted five defects
   against R0258's `chunk_loop_polls` and **three passed**; a fourth
   (`poll_rebound_via_a_later_def`) passed too. `chunk_loop_polls_v2` closes all
   of them and `poll_is_effective()` closes the runtime one, and every plant here
   is verified to be **accepted by the shipped R0258 guard** — so each refusal is
   a demonstrated behaviour change, not an unfalsifiable pass.

Safety
------
Nothing here signals a process, starts a child, hands cuVS anything, or wraps a
subprocess in a timeout. Every bulk input is opened read-only. The three edge
members are 10,044,413,144 B each and the substrate is 153,600,000,128 B; none
of them is ever copied by this module except through `materialise_endpoints`,
which is the copy review-0258-01 §H.3 priced at `10.2` s / `18.7` GiB and asked
for by name.
"""
from __future__ import annotations

import ast
import dis
import inspect
import os
import textwrap
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from basemap.round0253_stop_hooks import registered_ceiling_s
from basemap.round0258_graph_load import (
    SCAN_CHUNK_ELEMENTS,
    bitwise_identical,
    chunk_loop_polls,
    contiguous_int32_polled,
    endpoint_bounds_polled,
    load_array_polled,
    weight_validity_polled,
)

ROUND_ID = "0259"


class Round0259Error(RuntimeError):
    """R0259 fails closed."""


class Round0259RungError(Round0259Error):
    """A config was handed to an entry that does not serve its rung."""


class Round0259ContainerError(Round0259Error):
    """A graph container does not have the shape the loader branch requires."""


# --------------------------------------------------------------------------- #
# 1. the rung registry -- applicability, made explicit and checkable
# --------------------------------------------------------------------------- #

RUNG_2M = "r0113-2m"
RUNG_100M = "r0243-100m"

#: One entry per rung. `rows_key` differs between the two manifests on purpose:
#: R0113 seals `retained_rows` because rows were excluded; R0243 seals `rows`
#: because *"the 100,000,000 substrate rows ARE the universe"* (its own
#: canonicalization note). Reading the wrong key is `I2`.
RUNGS: dict[str, dict[str, Any]] = {
    RUNG_2M: {
        "rung": RUNG_2M,
        "schema": "round0113-prompt-arm-fuzzy-graph-v1",
        "rows_key": "retained_rows",
        "rows": 1_993_761,
        "k": 50,
        "dimension": 768,
        "container": "bulk_npz",
        "entry": "experiments.round0113_nodes.run_train",
    },
    RUNG_100M: {
        "rung": RUNG_100M,
        "schema": (
            "round0243-minilm-mixed-100000k-k15-fuzzy-and-canonical-tripwire-v1"
        ),
        "rows_key": "rows",
        "rows": 100_000_000,
        "k": 15,
        "dimension": 384,
        "container": "streamed_npy_members",
        "entry": "experiments.round0259_nodes.run_train_100m",
    },
}

#: The 100M substrate. Its shape is the only checkable source of `I4` — R0243's
#: graph manifest carries no `dimension` key, so an entry that merely *asserted*
#: 384 would be asserting, not checking. `assert_substrate_dimension` opens the
#: `.npy` header (no payload read) and compares the shape.
SUBSTRATE_100M_PATH = (
    "/data/latent-basemap/runs/round-0238/queue/artifacts/"
    "minilm-mixed-100000k-nested-substrate-and-reserves-v1/substrate.f32.npy"
)
SUBSTRATE_100M_SHAPE = (100_000_000, 384)
SUBSTRATE_100M_BYTES = 153_600_000_128


def rung_of(manifest: Mapping[str, Any]) -> str:
    """Resolve a graph manifest to exactly one registered rung, or raise.

    There is deliberately no default and no "closest match". A manifest whose
    schema is registered but whose rows or `k` disagree is a **louder** failure
    than an unknown schema, because that is the shape a silently-widened entry
    would produce.
    """
    schema = manifest.get("schema")
    for rung in RUNGS.values():
        if schema != rung["schema"]:
            continue
        rows = manifest.get(rung["rows_key"])
        k = manifest.get("k")
        if int(rows if rows is not None else -1) != rung["rows"]:
            raise Round0259RungError(
                f"R0259: manifest carries rung schema {schema!r} but "
                f"{rung['rows_key']} = {rows!r}, not {rung['rows']}. Refusing "
                "to guess which rung this is."
            )
        if int(k if k is not None else -1) != rung["k"]:
            raise Round0259RungError(
                f"R0259: manifest carries rung schema {schema!r} but k = {k!r}, "
                f"not {rung['k']}."
            )
        return str(rung["rung"])
    raise Round0259RungError(
        f"R0259: no registered rung has schema {schema!r}. Registered: "
        f"{sorted(r['schema'] for r in RUNGS.values())}"
    )


def assert_rung(manifest: Mapping[str, Any], *, expected: str, entry: str) -> dict[str, Any]:
    """The first line of every rung-specific entry.

    `run_train_100m` calls it with `expected=RUNG_100M`. Handing it R0113's 2M
    manifest raises; handing R0113's own entry a 100M manifest raises inside
    `round0113_prompt_contrast.load_graph` for `I1`–`I4`. Both directions are
    positive-controlled in `rung_applicability_controls()`.
    """
    if expected not in RUNGS:
        raise Round0259RungError(f"R0259: {expected!r} is not a registered rung")
    observed = rung_of(manifest)
    if observed != expected:
        raise Round0259RungError(
            f"R0259: {entry} serves rung {expected!r} only; this manifest is "
            f"rung {observed!r}. A config must never run through another rung's "
            "path -- that is the failure mode R0250-R0258 spent nine rounds "
            "hardening around."
        )
    return dict(RUNGS[expected])


def assert_substrate_dimension(
    path: str = SUBSTRATE_100M_PATH, *, expected: tuple[int, int] = SUBSTRATE_100M_SHAPE
) -> dict[str, Any]:
    """`I4`, checked rather than asserted: open the header, compare the shape.

    Reads the `.npy` header only. No payload byte of the `153,600,000,128` B
    substrate is touched.
    """
    if not os.path.isfile(path):
        raise Round0259Error(f"R0259: substrate {path} is absent")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    shape = tuple(int(value) for value in array.shape)
    dtype = str(array.dtype)
    size = os.path.getsize(path)
    del array
    if shape != tuple(expected):
        raise Round0259RungError(
            f"R0259: substrate at {path} has shape {shape}, not {tuple(expected)}. "
            "The rung's dimension is not what the entry serves."
        )
    return {
        "schema": "round0259-substrate-dimension-v1",
        "canonical_path": path,
        "shape": list(shape),
        "dtype": dtype,
        "bytes": size,
        "dimension": int(shape[1]),
        "note": (
            "R0243's graph manifest carries no `dimension` key, so I4 is checked "
            "against the substrate's own .npy header. Header only -- no payload "
            "read."
        ),
    }


# --------------------------------------------------------------------------- #
# 2. the loader branch -- I5
# --------------------------------------------------------------------------- #

#: R0243's member names, relative to the artifact directory.
MEMBER_FILENAMES = {
    "sources": "edges-k15-fuzzy-src.i32.npy",
    "targets": "edges-k15-fuzzy-dst.i32.npy",
    "weights": "edges-k15-fuzzy-wts.f32.npy",
    "header": "edges-k15-fuzzy-header.npz",
}

#: The scalar keys a streamed header must carry, and no array member.
HEADER_KEYS = ("directed_edges", "k", "n_nodes")

#: What the *bulk* container carries. `load_edge_arrays` must keep serving this
#: shape unchanged; the branch below must never claim it.
BULK_NPZ_KEYS = ("sources", "targets")


def _npz_member_names(path: str) -> list[str] | None:
    """Member names of an `.npz`, or None if `path` is not a readable `.npz`."""
    try:
        with np.load(path, mmap_mode="r", allow_pickle=False) as archive:
            return sorted(archive.files)
    except Exception:  # noqa: BLE001 -- "not an npz" is the only thing asked
        return None


def streamed_member_layout(path: str) -> dict[str, Any] | None:
    """Is `path` a streamed-member 100M graph? Return its members, or None.

    The discriminator is **content**, never a filename or a size:

    * a directory that contains all four member files; or
    * an `.npz` whose member set is exactly the scalar header keys and which
      therefore carries no edge array at all.

    A bulk `.npz` with a `sources` member returns None and goes down the shipped
    path untouched. `bulk_npz_is_not_claimed()` is the positive control.
    """
    if os.path.isdir(path):
        members = {
            name: os.path.join(path, filename)
            for name, filename in MEMBER_FILENAMES.items()
        }
        if all(os.path.isfile(member) for member in members.values()):
            return {"directory": path, "members": members, "matched_on": "directory"}
        if os.path.isfile(members["header"]):
            # The header is what makes this a streamed-member claim. A directory
            # that carries one and is missing a member is a broken 100M graph,
            # not "some other container" -- falling through would surface as an
            # `IsADirectoryError` from `np.load`, which names nothing.
            missing = sorted(
                name for name, member in members.items()
                if not os.path.isfile(member)
            )
            raise Round0259ContainerError(
                f"R0259: {path} carries a streamed edge header but is missing "
                f"members: {missing}"
            )
        return None
    if not os.path.isfile(path):
        return None
    names = _npz_member_names(path)
    if names is None or tuple(names) != tuple(sorted(HEADER_KEYS)):
        return None
    directory = os.path.dirname(path)
    members = {
        name: os.path.join(directory, filename)
        for name, filename in MEMBER_FILENAMES.items()
    }
    members["header"] = path
    missing = sorted(
        name for name, member in members.items() if not os.path.isfile(member)
    )
    if missing:
        raise Round0259ContainerError(
            f"R0259: {path} is a streamed edge header but its members are "
            f"missing: {missing}"
        )
    return {"directory": directory, "members": members, "matched_on": "header"}


def _open_member(path: str, *, label: str, dtype: np.dtype) -> np.memmap:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if not isinstance(array, np.memmap) or array.flags.writeable:
        raise Round0259ContainerError(
            f"R0259 {label} at {path} is not a read-only np.memmap"
        )
    if array.dtype != dtype or array.ndim != 1:
        raise Round0259ContainerError(
            f"R0259 {label} at {path} is {array.dtype} with shape {array.shape}; "
            f"expected a 1-D {dtype} member"
        )
    return array


def load_streamed_edge_members(
    path: str, *, load_weights: bool = True
) -> tuple[np.memmap, np.memmap, np.memmap | None, int]:
    """Open the 100M rung's three `.npy` members as read-only memmaps.

    Returns `(sources, targets, weights_or_None, n_nodes)` — the same 4-tuple
    contract `load_edge_arrays` has always returned, so every caller downstream
    is unchanged.

    Memmaps, not resident arrays. A resident load of the endpoints is a
    deliberate, separately measured step (`materialise_endpoints`), because
    review-0258-01 §H.2/§H.3 showed the *residency* is what costs `4,000x` per
    batch, not the open. Nothing >= 2 GB is materialised here.

    **Fails loudly on a wrong shape.** A member whose length disagrees with the
    header's `directed_edges`, a header whose scalars disagree with each other,
    or a member of the wrong dtype each raise `Round0259ContainerError` rather
    than returning a truncated or reinterpreted array.
    """
    layout = streamed_member_layout(path)
    if layout is None:
        raise Round0259ContainerError(
            f"R0259: {path} is not a streamed-member edge graph"
        )
    members = layout["members"]
    with np.load(members["header"], allow_pickle=False) as archive:
        header = {key: int(np.asarray(archive[key])) for key in sorted(archive.files)}
    missing = sorted(set(HEADER_KEYS) - set(header))
    if missing:
        raise Round0259ContainerError(
            f"R0259: streamed header {members['header']} lacks {missing}"
        )
    n_nodes = header["n_nodes"]
    directed_edges = header["directed_edges"]
    if n_nodes <= 0 or directed_edges <= 0 or header["k"] <= 0:
        raise Round0259ContainerError(
            f"R0259: streamed header carries a non-positive scalar: {header}"
        )

    sources = _open_member(members["sources"], label="sources", dtype=np.dtype(np.int32))
    targets = _open_member(members["targets"], label="targets", dtype=np.dtype(np.int32))
    for name, array in (("sources", sources), ("targets", targets)):
        if int(array.shape[0]) != directed_edges:
            raise Round0259ContainerError(
                f"R0259: {name} member holds {int(array.shape[0])} edges, header "
                f"declares {directed_edges}"
            )
    weights: np.memmap | None = None
    if load_weights:
        weights = _open_member(
            members["weights"], label="weights", dtype=np.dtype(np.float32)
        )
        if int(weights.shape[0]) != directed_edges:
            raise Round0259ContainerError(
                f"R0259: weights member holds {int(weights.shape[0])} edges, "
                f"header declares {directed_edges}"
            )
    return sources, targets, weights, int(n_nodes)


def streamed_member_signature(path: str) -> dict[str, Any]:
    """Sizes, dtypes and lengths of the members, without reading a payload byte."""
    layout = streamed_member_layout(path)
    if layout is None:
        raise Round0259ContainerError(f"R0259: {path} is not a streamed-member graph")
    members = layout["members"]
    with np.load(members["header"], allow_pickle=False) as archive:
        header = {key: int(np.asarray(archive[key])) for key in sorted(archive.files)}
    described: dict[str, Any] = {}
    for name in ("sources", "targets", "weights"):
        array = np.load(members[name], mmap_mode="r", allow_pickle=False)
        described[name] = {
            "canonical_path": members[name],
            "bytes": os.path.getsize(members[name]),
            "dtype": str(array.dtype),
            "elements": int(array.shape[0]),
        }
        del array
    return {
        "schema": "round0259-streamed-member-signature-v1",
        "matched_on": layout["matched_on"],
        "header": header,
        "members": described,
        "header_path": members["header"],
    }


# --------------------------------------------------------------------------- #
# 3. residency -- review-0258-01 §H.3's correction to R0230's isinstance rule
# --------------------------------------------------------------------------- #


def file_backed(array: Any) -> bool:
    """Does `array` share storage with a file mapping?

    Walks the `.base` chain. `np.ascontiguousarray(np.asarray(memmap))` returns a
    base `ndarray` whose `.base` **is** the memmap, so `isinstance(x, np.memmap)`
    is `False` while the bytes are still on disk. That is exactly
    `HostStreamEdgeSampler._src_h`, and it is why R0230's reader rule cannot be
    the residency test.
    """
    seen: set[int] = set()
    current = array
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, np.memmap):
            return True
        current = getattr(current, "base", None)
    return False


def r0230_isinstance_verdict(array: Any) -> bool:
    """R0230's rule, verbatim, kept only so the contrast is measurable."""
    return isinstance(array, np.memmap)


def endpoint_residency(array: Any, *, label: str) -> dict[str, Any]:
    """Both verdicts, side by side, on one object."""
    return {
        "label": label,
        "dtype": str(np.asarray(array).dtype),
        "elements": int(np.asarray(array).shape[0]),
        "bytes": int(np.asarray(array).nbytes),
        "file_backed_by_the_base_chain": file_backed(array),
        "r0230_isinstance_np_memmap": r0230_isinstance_verdict(array),
        "the_two_rules_agree": (
            file_backed(array) == r0230_isinstance_verdict(array)
        ),
    }


def assert_resident(array: Any, *, label: str) -> dict[str, Any]:
    report = endpoint_residency(array, label=label)
    if report["file_backed_by_the_base_chain"]:
        raise Round0259Error(
            f"R0259: {label} is still file-backed. A per-batch gather from a "
            "file-backed endpoint array was measured at ~4,000x the resident "
            "cost (review-0258-01 §H.2)."
        )
    return report


# --------------------------------------------------------------------------- #
# 4. the chunked pairwise sum -- review-0258-01 §E, implemented and re-proved
# --------------------------------------------------------------------------- #

#: numpy's scalar pairwise reduction recurses only above this length, and splits
#: at `n2 = n//2 - n//2 % 8`. Both constants come from numpy's
#: `pairwise_sum_@TYPE@`; neither is an API guarantee, which is why
#: `assert_pairwise_rule()` re-proves the identity at startup and raises.
NUMPY_PAIRWISE_BLOCKSIZE = 128
NUMPY_PAIRWISE_UNROLL = 8

#: The numpy this rule was proved bitwise-identical on, in this release's venv.
#: Recorded, and compared at runtime; a different version does not fail on the
#: version string alone -- it fails, loudly, only if the self-check stops
#: reproducing. review-0258-01 §K listed cross-version stability as the thing it
#: could not verify.
NUMPY_VERSION_VERIFIED = "2.4.4"

#: Below this a leaf would land inside numpy's non-recursing block and the split
#: points would stop matching. `1 << 13` is the smallest leaf proved here.
MIN_LEAF_ELEMENTS = 1 << 13

#: The runtime chunk bound review-0258-01 §D.3 found missing. A chunk at or above
#: this is refused, so `one chunk == the whole array` cannot audit clean at any
#: rung.
MAX_CHUNK_ELEMENTS = SCAN_CHUNK_ELEMENTS

ABORT_POLL_SITE_PAIRWISE = "round0259_hundred_m.pairwise_sum_polled leaf"


def assert_chunk_bounded(n: int, chunk: int) -> None:
    """The chunk-size bound R0258's structural guard does not impose."""
    if chunk <= 0 or chunk > MAX_CHUNK_ELEMENTS:
        raise Round0259Error(
            f"R0259: chunk {chunk} is outside (0, {MAX_CHUNK_ELEMENTS}]. A loop "
            "whose chunk is the whole array polls once and is not pollable."
        )
    if n > MAX_CHUNK_ELEMENTS and chunk >= n:
        raise Round0259Error(
            f"R0259: chunk {chunk} covers the whole {n}-element array"
        )


def poll_is_effective(poll: Any) -> bool:
    """Is `poll` something that can actually observe an abort flag?

    review-0258-01 §D.3: `poll=lambda w: None` audits clean under R0258's
    structural guard, because the guard bounds the *shape* of the loop and not
    the *identity* of `poll`. This is the runtime half. A callable whose bytecode
    does nothing but return a constant is refused.
    """
    if not callable(poll):
        return False
    code = getattr(poll, "__code__", None)
    if code is None:
        return True  # a callable object with a __call__ body; not a bare no-op
    trivial = {
        "RESUME", "NOP", "CACHE", "LOAD_CONST", "RETURN_CONST", "RETURN_VALUE",
        "LOAD_FAST", "LOAD_FAST_BORROW", "POP_TOP", "MAKE_CELL", "COPY_FREE_VARS",
    }
    opnames = {instruction.opname for instruction in dis.get_instructions(code)}
    return not opnames <= trivial


def assert_poll_is_effective(poll: Any, *, label: str) -> None:
    if not poll_is_effective(poll):
        raise Round0259Error(
            f"R0259: {label} was handed a poll that does nothing. The loop would "
            "audit clean and stop for nothing."
        )


def _pairwise_leaves(n: int, leaf: int) -> list[tuple[int, int]]:
    """numpy's split points, flattened to leaf segments in evaluation order."""
    segments: list[tuple[int, int]] = []

    def walk(lo: int, hi: int) -> None:
        span = hi - lo
        if span <= leaf:
            segments.append((lo, hi))
            return
        half = span // 2
        half -= half % NUMPY_PAIRWISE_UNROLL
        walk(lo, lo + half)
        walk(lo + half, hi)

    walk(0, n)
    return segments


def _pairwise_value(
    array: np.ndarray, lo: int, hi: int, leaf: int, poll: Any, counter: list[int]
) -> np.float64:
    span = hi - lo
    if span <= leaf:
        counter[0] += 1
        poll(ABORT_POLL_SITE_PAIRWISE)
        return np.sum(array[lo:hi])
    half = span // 2
    half -= half % NUMPY_PAIRWISE_UNROLL
    left = _pairwise_value(array, lo, lo + half, leaf, poll, counter)
    right = _pairwise_value(array, lo + half, hi, leaf, poll, counter)
    return left + right


def pairwise_sum_polled(
    array: np.ndarray, *, poll: Any, leaf: int = 1 << 20
) -> tuple[float, int]:
    """`array.sum()`, bitwise identical, with an abort read at every leaf.

    Returns `(total, leaves)`. The association is numpy's own: recurse while the
    segment is longer than `leaf`, splitting at `n2 = n//2 - n//2 % 8`, then let
    `np.sum` compute the leaf — which re-enters the identical recursion inside
    numpy and therefore issues the identical additions in the identical order.

    float64 only. review-0258-01 §E: for float32 numpy's SIMD path breaks the
    identity below a `2^24` leaf. The shipped CDF sums float64
    (`w = np.array(weights, dtype=np.float64)` then `w.sum()`), so the applicable
    case is the one that works, and the other one is refused rather than
    silently mis-summed.
    """
    array = np.asarray(array)
    if array.dtype != np.float64:
        raise Round0259Error(
            f"R0259: the pairwise decomposition is proved for float64 only; got "
            f"{array.dtype}. numpy's float32 SIMD reduction does not honour the "
            "scalar split points below a 2^24 leaf."
        )
    if array.ndim != 1:
        raise Round0259Error("R0259: pairwise_sum_polled takes a 1-D array")
    if leaf < MIN_LEAF_ELEMENTS:
        raise Round0259Error(
            f"R0259: leaf {leaf} is below {MIN_LEAF_ELEMENTS}; the split points "
            "stop matching inside numpy's non-recursing block"
        )
    if not callable(poll):
        raise Round0259Error("R0259: pairwise_sum_polled requires a callable poll")
    counter = [0]
    total = _pairwise_value(array, 0, int(array.shape[0]), int(leaf), poll, counter)
    return float(total), counter[0]


# ---- the five planted numeric defects ------------------------------------- #


def _defect_naive_block_sum(array, *, leaf):
    total = np.float64(0.0)
    for start in range(0, len(array), leaf):
        total = total + np.sum(array[start:start + leaf])
    return float(total)


def _defect_split_without_the_multiple_of_eight(array, *, leaf):
    def walk(lo, hi):
        span = hi - lo
        if span <= leaf:
            return np.sum(array[lo:hi])
        half = span // 2            # the `- half % 8` is the defect
        return walk(lo, lo + half) + walk(lo + half, hi)

    return float(walk(0, len(array)))


def _defect_split_rounded_up_to_eight(array, *, leaf):
    def walk(lo, hi):
        span = hi - lo
        if span <= leaf:
            return np.sum(array[lo:hi])
        half = span // 2
        half += (-half) % NUMPY_PAIRWISE_UNROLL   # rounds up, numpy rounds down
        if half >= span:
            half = span - NUMPY_PAIRWISE_UNROLL
        return walk(lo, lo + half) + walk(lo + half, hi)

    return float(walk(0, len(array)))


def _defect_float32_leaf_accumulator(array, *, leaf):
    total = np.float32(0.0)
    for start in range(0, len(array), leaf):
        total = np.float32(total + np.float32(np.sum(array[start:start + leaf])))
    return float(total)


def _defect_left_to_right_over_the_leaves(array, *, leaf):
    total = np.float64(0.0)
    for lo, hi in _pairwise_leaves(len(array), leaf):
        total = total + np.sum(array[lo:hi])
    return float(total)


NUMERIC_SUM_DEFECTS = {
    "naive_block_sum": _defect_naive_block_sum,
    "split_without_the_multiple_of_eight": _defect_split_without_the_multiple_of_eight,
    "split_rounded_up_to_eight": _defect_split_rounded_up_to_eight,
    "float32_leaf_accumulator": _defect_float32_leaf_accumulator,
    "left_to_right_over_the_leaves": _defect_left_to_right_over_the_leaves,
}


def pairwise_sum_controls(
    *,
    n: int = 4_000_019,
    leaf: int = 1 << 14,
    seed: int = 20260812,
    values: np.ndarray | None = None,
) -> dict[str, Any]:
    """Five planted sum defects the bitwise comparison must refuse.

    The R0254 standard applies: a defect both predicates refuse proves nothing,
    so each is also scored against the `np.allclose`-shaped predicate an
    implementer would reach for. `n` is not a multiple of `leaf` on purpose.

    **The plants are data- and size-dependent and this is registered, not
    discovered.** `split_without_the_multiple_of_eight` and
    `split_rounded_up_to_eight` differ from numpy's split only when some
    `n // 2` along the recursion is not already a multiple of `8`; at
    `n = 4_000_017` every level happens to be, and both plants come out bitwise
    equal. `n = 4_000_019` is chosen because it is not. review-0258-01 §E made
    the same observation about R0258's own `blocked_total` plant, which fired at
    its control's parameters and would not always. `pairwise_sum_controls` is
    therefore also run over the **real** `2,511,103,254`-edge weights in the
    measurement node, where nothing was chosen.
    """
    if values is None:
        rng = np.random.default_rng(seed)
        values = rng.random(n, dtype=np.float64) + 1e-6
    else:
        values = np.asarray(values)
        n = int(values.shape[0])
    reference = float(values.sum())
    polled, leaves = pairwise_sum_polled(
        values, poll=lambda _where: None, leaf=leaf
    )
    verdicts: dict[str, Any] = {}
    for name, builder in sorted(NUMERIC_SUM_DEFECTS.items()):
        planted = builder(values, leaf=leaf)
        verdicts[name] = {
            "refused_by_the_bitwise_comparison": planted != reference,
            "accepted_by_an_allclose_comparison": bool(
                np.allclose(planted, reference)
            ),
            "absolute_deviation": abs(planted - reference),
        }
    return {
        "schema": "round0259-pairwise-sum-controls-v1",
        "elements": int(n),
        "leaf_elements": int(leaf),
        "tail_leaf_exercised": bool(n % leaf),
        "seed": int(seed),
        "reference_total": reference,
        "polled_total": polled,
        "leaves_visited": int(leaves),
        "the_polled_sum_is_bitwise_identical": polled == reference,
        "planted": verdicts,
        "defects_planted": len(NUMERIC_SUM_DEFECTS),
        "defects_refused_by_the_bitwise_comparison": sum(
            1 for v in verdicts.values() if v["refused_by_the_bitwise_comparison"]
        ),
        "defects_accepted_by_an_allclose_comparison": sum(
            1 for v in verdicts.values() if v["accepted_by_an_allclose_comparison"]
        ),
    }


def assert_pairwise_rule(
    *, sizes: Sequence[int] = (12_345_678, 3_000_001, 1_000_003, 262_144, 100_000, 16_385),
    leaves: Sequence[int] = (1 << 24, 1 << 20, 1 << 14, 1 << 13),
    seed: int = 20260812,
) -> dict[str, Any]:
    """Re-prove the split rule bitwise, here, before anything relies on it.

    review-0258-01 §K could not verify the rule across numpy versions. This does
    not accept it on a version string: it recomputes `np.sum` and the
    decomposition over six sizes, four leaf sizes and three distributions
    (uniform, real-weight-shaped, and magnitudes spanning `1e-25`-`1e+25`) and
    **raises** on the first mismatch. A numpy whose reduction changed fails here
    rather than in a 10 h fit.
    """
    rng = np.random.default_rng(seed)
    checks: list[dict[str, Any]] = []
    for n in sizes:
        uniform = rng.random(n, dtype=np.float64)
        weightlike = (rng.random(n, dtype=np.float64).astype(np.float32) + np.float32(1e-6)).astype(np.float64)
        spread = rng.random(n, dtype=np.float64) * rng.choice(
            [1e-25, 1.0, 1e25], size=n
        )
        for label, values in (
            ("uniform", uniform), ("weightlike", weightlike), ("spread", spread)
        ):
            reference = values.sum()
            for leaf in leaves:
                if leaf < MIN_LEAF_ELEMENTS:
                    continue
                total, count = pairwise_sum_polled(
                    values, poll=lambda _where: None, leaf=leaf
                )
                identical = total == float(reference)
                checks.append({
                    "elements": int(n),
                    "distribution": label,
                    "leaf_elements": int(leaf),
                    "leaves_visited": int(count),
                    "bitwise_identical": bool(identical),
                    "absolute_deviation": abs(total - float(reference)),
                })
                if not identical:
                    raise Round0259Error(
                        "R0259: numpy's pairwise split rule does not reproduce "
                        f"bitwise on numpy {np.__version__} at n={n} "
                        f"leaf={leaf} distribution={label}. The chunked sum is "
                        "NOT safe to ship on this numpy."
                    )
    controls = pairwise_sum_controls()
    if not controls["the_polled_sum_is_bitwise_identical"]:
        raise Round0259Error("R0259: the polled pairwise sum is not bitwise identical")
    if controls["defects_refused_by_the_bitwise_comparison"] != len(NUMERIC_SUM_DEFECTS):
        raise Round0259Error("R0259: a planted sum defect was not refused")
    if controls["defects_accepted_by_an_allclose_comparison"] < 1:
        raise Round0259Error(
            "R0259: no planted sum defect was accepted by the weaker comparison, "
            "so the refusals are not demonstrated behaviour changes"
        )
    return {
        "schema": "round0259-pairwise-rule-selfcheck-v1",
        "numpy_version_observed": str(np.__version__),
        "numpy_version_verified": NUMPY_VERSION_VERIFIED,
        "numpy_version_matches_the_verified_one": (
            str(np.__version__) == NUMPY_VERSION_VERIFIED
        ),
        "split_rule": "n2 = n // 2 - (n // 2) % 8, recursing while n > leaf",
        "numpy_pairwise_blocksize": NUMPY_PAIRWISE_BLOCKSIZE,
        "numpy_pairwise_unroll": NUMPY_PAIRWISE_UNROLL,
        "checks": checks,
        "checks_run": len(checks),
        "every_check_bitwise_identical": all(c["bitwise_identical"] for c in checks),
        "numeric_controls": controls,
    }


# --------------------------------------------------------------------------- #
# 5. the polled CDF, with the residue removed
# --------------------------------------------------------------------------- #


def weight_cdf_fully_polled(
    weights: np.ndarray,
    *,
    poll: Any,
    chunk: int = SCAN_CHUNK_ELEMENTS,
    leaf: int = 1 << 20,
    sum_proof_leaves: Sequence[int] = (),
    sum_proof_controls: bool = False,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """R0258's `weight_cdf_polled` with its one unpollable interval removed.

    Identical in every other respect — the f64 conversion, the carry-folded
    cumsum and the division are R0258's, and are bitwise identical to the shipped
    path by its §C/§D evidence and mine. The only change is that `w.sum()` is
    replaced by `pairwise_sum_polled`, which review-0258-01 §E proved reproduces
    it bitwise at `1.00x` wall.
    """
    if not callable(poll):
        raise Round0259Error("R0259: the fully polled CDF requires a callable poll")
    assert_poll_is_effective(poll, label="weight_cdf_fully_polled")
    n = int(weights.shape[0])
    assert_chunk_bounded(n, chunk)
    timings: dict[str, Any] = {}

    started = time.monotonic()
    w = np.empty(n, dtype=np.float64)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        w[start:end] = weights[start:end]
        poll("round0259_hundred_m.weight_cdf_fully_polled convert chunk")
    timings["convert_s"] = time.monotonic() - started

    if sum_proof_leaves or sum_proof_controls:
        # The proof runs on the REAL float64 array, before the cumsum, so
        # nothing about it is a scaled-down stand-in. review-0258-01 §E's table
        # is reproduced here at the same three leaf sizes over the same
        # 2,511,103,254 elements.
        proof_started = time.monotonic()
        reference_started = time.monotonic()
        reference = float(np.sum(w))
        reference_s = time.monotonic() - reference_started
        rows: list[dict[str, Any]] = []
        for proof_leaf in sum_proof_leaves:
            leaf_started = time.monotonic()
            candidate, sites = pairwise_sum_polled(w, poll=poll, leaf=int(proof_leaf))
            leaf_wall = time.monotonic() - leaf_started
            rows.append({
                "leaf_elements": int(proof_leaf),
                "poll_sites": int(sites),
                "wall_s": leaf_wall,
                "wall_over_one_np_sum": leaf_wall / reference_s if reference_s else None,
                "mean_poll_spacing_s": leaf_wall / sites if sites else None,
                "bitwise_identical": candidate == reference,
                "absolute_deviation": abs(candidate - reference),
            })
        timings["sum_proof"] = {
            "schema": "round0259-real-rung-pairwise-proof-v1",
            "elements": int(n),
            "dtype": "float64",
            "one_np_sum_s": reference_s,
            "reference_total": reference,
            "by_leaf": rows,
            "every_leaf_bitwise_identical": all(r["bitwise_identical"] for r in rows),
            "numpy_version": str(np.__version__),
            "controls_at_the_real_rung": (
                pairwise_sum_controls(values=w, leaf=leaf)
                if sum_proof_controls else None
            ),
            "proof_wall_s": time.monotonic() - proof_started,
        }

    started = time.monotonic()
    total, leaves = pairwise_sum_polled(w, poll=poll, leaf=leaf)
    timings["sum_s"] = time.monotonic() - started
    timings["sum_leaves"] = int(leaves)
    if not (total > 0):
        raise Round0259Error(f"R0259: non-positive weight total {total}: unsamplable")

    started = time.monotonic()
    carry = 0.0
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        w[start] += carry
        np.cumsum(w[start:end], out=w[start:end])
        carry = float(w[end - 1])
        poll("round0259_hundred_m.weight_cdf_fully_polled cumsum chunk")
    timings["cumsum_s"] = time.monotonic() - started

    started = time.monotonic()
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        w[start:end] /= total
        poll("round0259_hundred_m.weight_cdf_fully_polled divide chunk")
    timings["divide_s"] = time.monotonic() - started
    timings["leaf_elements"] = int(leaf)
    timings["chunk_elements"] = int(chunk)
    return w, total, timings


# --------------------------------------------------------------------------- #
# 6. endpoint materialisation -- review-0258-01 §H.3's named remedy
# --------------------------------------------------------------------------- #


def materialise_endpoints(
    sources: Any,
    targets: Any,
    *,
    poll: Any,
    chunk: int = SCAN_CHUNK_ELEMENTS,
    source_route: str = "contiguous_polled",
    target_route: str = "load_polled",
    source_path: str | None = None,
    target_path: str | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Turn the two endpoint memmaps into genuinely resident int32 arrays.

    Two routes are wired, and both are R0258's polled stages, because both are
    ways a 100M loader could reach residency and each must be proved bitwise:

    * `contiguous_polled` — `round0258_graph_load.contiguous_int32_polled` from a
      read-only memmap. This is `HostStreamEdgeSampler.__init__`'s call, made to
      actually copy.
    * `load_polled` — `round0258_graph_load.load_array_polled`, which streams the
      `.npy` payload into an anonymous mapping in `64` MiB polled reads.

    Both results are verified `bitwise_identical` to the memmap they came from
    and `assert_resident`-checked by the base-chain rule, not by `isinstance`.
    """
    assert_poll_is_effective(poll, label="materialise_endpoints")
    assert_chunk_bounded(int(np.asarray(sources).shape[0]), chunk)
    report: dict[str, Any] = {"routes": {}}

    def _run(name: str, route: str, array: Any, path: str | None) -> np.ndarray:
        started = time.monotonic()
        if route == "contiguous_polled":
            out = contiguous_int32_polled(array, poll=poll, chunk=chunk)
        elif route == "load_polled":
            if not path:
                raise Round0259Error(
                    f"R0259: the load_polled route for {name} needs the member path"
                )
            out = load_array_polled(path, poll=poll, o_direct=False)
        else:
            raise Round0259Error(f"R0259: unknown materialisation route {route!r}")
        wall = time.monotonic() - started
        identical = bitwise_identical(np.asarray(out), np.asarray(array))
        residency = assert_resident(out, label=name)
        if not identical:
            raise Round0259Error(
                f"R0259: the materialised {name} is not bitwise identical to the "
                "member it was read from"
            )
        report["routes"][name] = {
            "route": route,
            "wall_s": wall,
            "bytes": int(np.asarray(out).nbytes),
            "bitwise_identical_to_the_member": identical,
            "residency": residency,
            "source_residency": endpoint_residency(array, label=f"{name} member"),
        }
        return out

    resident_sources = _run("sources", source_route, sources, source_path)
    resident_targets = _run("targets", target_route, targets, target_path)
    report["total_wall_s"] = float(
        sum(entry["wall_s"] for entry in report["routes"].values())
    )
    report["total_bytes"] = int(
        sum(entry["bytes"] for entry in report["routes"].values())
    )
    report["schema"] = "round0259-endpoint-materialisation-v1"
    return resident_sources, resident_targets, report


# --------------------------------------------------------------------------- #
# 7. the strengthened chunk-loop guard -- review-0258-01 §D.3's three holes
# --------------------------------------------------------------------------- #

_REBINDING_NODES = (
    ast.Assign, ast.AugAssign, ast.AnnAssign, ast.NamedExpr,
    ast.FunctionDef, ast.AsyncFunctionDef, ast.For, ast.AsyncFor,
    ast.With, ast.AsyncWith, ast.Import, ast.ImportFrom,
)


def _rebinds_poll(target: ast.AST) -> bool:
    """Any construct that could rebind the name `poll` inside the function."""
    for node in ast.walk(target):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node is not target and node.name == "poll":
                return True
        elif isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == "poll" for t in node.targets):
                return True
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            if isinstance(node.target, ast.Name) and node.target.id == "poll":
                return True
        elif isinstance(node, ast.NamedExpr):
            if isinstance(node.target, ast.Name) and node.target.id == "poll":
                return True
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            if any(
                isinstance(inner, ast.Name) and inner.id == "poll"
                for inner in ast.walk(node.target)
            ):
                return True
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars is not None and any(
                    isinstance(inner, ast.Name) and inner.id == "poll"
                    for inner in ast.walk(item.optional_vars)
                ):
                    return True
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            if any((alias.asname or alias.name) == "poll" for alias in node.names):
                return True
    return False


def _step_is_the_whole_array(loop: ast.AST) -> bool:
    """Does the loop step over the array's full length -- i.e. one chunk?"""
    if not isinstance(loop, ast.For):
        return False
    call = loop.iter
    if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
            and call.func.id == "range" and len(call.args) == 3):
        return False
    step = call.args[2]
    if isinstance(step, ast.Call) and isinstance(step.func, ast.Name) \
            and step.func.id == "len":
        return True
    if isinstance(step, ast.Attribute) and step.attr in ("size",):
        return True
    if isinstance(step, ast.Subscript) and isinstance(step.value, ast.Attribute) \
            and step.value.attr == "shape":
        return True
    return False


def chunk_loop_polls_v2(
    source: str | None = None, *, function: Any = None
) -> dict[str, Any]:
    """R0258's guard with review-0258-01 §D.3's holes closed.

    Three rules are added, and each corresponds to a plant that passes the
    shipped R0258 guard:

    * the poll must be a **direct expression statement** of the loop body, not
      merely reachable from it — closes `poll_inside_a_dead_nested_loop`;
    * the loop body must contain no `break` or `return` among its direct
      statements — closes `poll_then_break_first_iteration`;
    * the loop's `range` step must not be the array's length — closes
      `one_chunk_equals_whole_array`;

    and the rebinding test is widened from `ast.Assign` to every binding form,
    which closes `poll_rebound_via_a_later_def`. The fifth plant,
    `poll_is_a_noop_by_default_arg`, is a *runtime* defect and is closed by
    `poll_is_effective`, not here — a static guard cannot see it, and saying so
    is the point.
    """
    if source is None:
        if function is None:
            raise Round0259Error("R0259 chunk-loop guard needs a function or source")
        source = inspect.getsource(function)
    tree = ast.parse(textwrap.dedent(source))
    functions = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if not functions:
        raise Round0259Error("R0259 chunk-loop guard was given no function")
    target = functions[0]

    loops = [
        loop for loop in ast.walk(target)
        if isinstance(loop, (ast.While, ast.For))
        and any(
            isinstance(inner, ast.Subscript)
            or (isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Attribute)
                and inner.func.attr in ("readv", "readinto", "read"))
            for inner in ast.walk(loop)
        )
    ]

    direct_polls: list[int] = []
    escaping: list[str] = []
    unbounded: list[int] = []
    for loop in loops:
        for statement in loop.body:
            if isinstance(statement, (ast.Break, ast.Return)):
                escaping.append(type(statement).__name__)
            if (isinstance(statement, ast.Expr)
                    and isinstance(statement.value, ast.Call)
                    and isinstance(statement.value.func, ast.Name)
                    and statement.value.func.id == "poll"):
                direct_polls.append(int(getattr(statement, "lineno", -1)))
        if _step_is_the_whole_array(loop):
            unbounded.append(int(getattr(loop, "lineno", -1)))

    rebound = _rebinds_poll(target)
    return {
        "schema": "round0259-chunk-loop-poll-guard-v2",
        "function": target.name,
        "checked": "the shipped function" if function is not None else "planted source",
        "chunk_loops_found": len(loops),
        "direct_poll_statements_in_a_chunk_loop": len(direct_polls),
        "direct_poll_linenos": sorted(direct_polls),
        "loops_whose_step_is_the_whole_array": unbounded,
        "escaping_statements_in_a_chunk_loop": sorted(set(escaping)),
        "poll_is_rebound_anywhere_in_the_function": rebound,
        "the_chunk_loop_polls": bool(
            loops and direct_polls and not rebound and not escaping and not unbounded
        ),
    }


POLLED_FUNCTIONS_V2 = (
    load_array_polled,
    endpoint_bounds_polled,
    weight_validity_polled,
    contiguous_int32_polled,
    weight_cdf_fully_polled,
)


def assert_chunk_loops_poll_v2() -> dict[str, Any]:
    """Every stage the 100M entry routes through must pass the stronger guard."""
    audit = {
        function.__name__: chunk_loop_polls_v2(function=function)
        for function in POLLED_FUNCTIONS_V2
    }
    missing = sorted(
        name for name, report in audit.items() if not report["the_chunk_loop_polls"]
    )
    if missing:
        raise Round0259Error(
            f"R0259: these polled stages fail the strengthened guard: {missing}"
        )
    return {
        "schema": "round0259-chunk-loop-audit-v2",
        "functions_checked": len(POLLED_FUNCTIONS_V2),
        "functions_that_pass": len(POLLED_FUNCTIONS_V2),
        "by_function": audit,
    }


# ---- the five planted structural defects, all of which pass R0258's guard --- #

_V2_PLANT_HONEST = '''
def stage(array, *, poll, chunk):
    out = np.empty(len(array))
    for start in range(0, len(array), chunk):
        out[start:start + chunk] = array[start:start + chunk]
        poll("site")
    return out
'''

_V2_PLANT_POLL_THEN_BREAK = '''
def stage(array, *, poll, chunk):
    out = np.empty(len(array))
    for start in range(0, len(array), chunk):
        out[start:start + chunk] = array[start:start + chunk]
        poll("site")
        break
    return out
'''

_V2_PLANT_ONE_CHUNK_IS_THE_WHOLE_ARRAY = '''
def stage(array, *, poll, chunk):
    out = np.empty(len(array))
    for start in range(0, len(array), len(array)):
        out[start:start + chunk] = array[start:start + chunk]
        poll("site")
    return out
'''

_V2_PLANT_POLL_REBOUND_VIA_A_LATER_DEF = '''
def stage(array, *, poll, chunk):
    out = np.empty(len(array))
    for start in range(0, len(array), chunk):
        out[start:start + chunk] = array[start:start + chunk]
        poll("site")

    def poll(where):
        return None

    return out
'''

_V2_PLANT_POLL_INSIDE_A_DEAD_NESTED_LOOP = '''
def stage(array, *, poll, chunk):
    out = np.empty(len(array))
    for start in range(0, len(array), chunk):
        out[start:start + chunk] = array[start:start + chunk]
        for _ in range(0):
            poll("site")
    return out
'''

#: The fifth is not a source plant. `poll_is_a_noop_by_default_arg` ships the
#: honest source above and passes it a `lambda where: None` at call time, which
#: no static guard can see. It is scored by `poll_is_effective`.
_V2_PLANT_NOOP_POLL_SOURCE = _V2_PLANT_HONEST

STRUCTURAL_DEFECTS_V2 = {
    "poll_then_break_first_iteration": _V2_PLANT_POLL_THEN_BREAK,
    "one_chunk_equals_whole_array": _V2_PLANT_ONE_CHUNK_IS_THE_WHOLE_ARRAY,
    "poll_rebound_via_a_later_def": _V2_PLANT_POLL_REBOUND_VIA_A_LATER_DEF,
    "poll_inside_a_dead_nested_loop": _V2_PLANT_POLL_INSIDE_A_DEAD_NESTED_LOOP,
}

#: The runtime plant and its honest counterpart.
RUNTIME_POLL_DEFECTS = {
    "poll_is_a_noop_by_default_arg": (lambda where: None),
    "poll_is_a_noop_def": None,   # filled in below, a def rather than a lambda
}


def _noop_poll_def(where):
    pass


RUNTIME_POLL_DEFECTS["poll_is_a_noop_def"] = _noop_poll_def


def structural_defect_controls_v2() -> dict[str, Any]:
    """Plant five defects that the SHIPPED R0258 guard accepts.

    That is the whole design. review-0258-01 §D.3 wrote four of these itself and
    reported them passing `chunk_loop_polls`; a guard whose plants it already
    refuses proves nothing. Each entry below therefore records **both** verdicts,
    and the round fails if the R0258 guard does not accept the plant — because
    then the plant is not the defect that was found.
    """
    verdicts: dict[str, Any] = {}
    for name, source in sorted(STRUCTURAL_DEFECTS_V2.items()):
        shipped = chunk_loop_polls(source)
        strengthened = chunk_loop_polls_v2(source)
        verdicts[name] = {
            "kind": "static",
            "accepted_by_the_shipped_r0258_guard": bool(
                shipped["the_chunk_loop_polls"]
            ),
            "refused_by_the_r0259_guard": not strengthened["the_chunk_loop_polls"],
            "why": {
                "escaping_statements": strengthened[
                    "escaping_statements_in_a_chunk_loop"],
                "loops_whose_step_is_the_whole_array": strengthened[
                    "loops_whose_step_is_the_whole_array"],
                "poll_is_rebound": strengthened[
                    "poll_is_rebound_anywhere_in_the_function"],
                "direct_poll_statements": strengthened[
                    "direct_poll_statements_in_a_chunk_loop"],
            },
        }
    for name, callable_plant in sorted(RUNTIME_POLL_DEFECTS.items()):
        shipped = chunk_loop_polls(_V2_PLANT_NOOP_POLL_SOURCE)
        verdicts[name] = {
            "kind": "runtime",
            "accepted_by_the_shipped_r0258_guard": bool(
                shipped["the_chunk_loop_polls"]
            ),
            "refused_by_the_r0259_guard": not poll_is_effective(callable_plant),
            "why": {
                "note": (
                    "the SOURCE is the honest install and passes both static "
                    "guards; the defect is the object bound to `poll` at call "
                    "time, which only poll_is_effective can see"
                )
            },
        }

    honest_static = chunk_loop_polls_v2(_V2_PLANT_HONEST)
    honest_shipped = chunk_loop_polls(_V2_PLANT_HONEST)

    def _honest_poll(where):
        record.append(where)

    record: list[str] = []
    return {
        "schema": "round0259-structural-defect-controls-v2",
        "planted": verdicts,
        "defects_planted": len(verdicts),
        "defects_accepted_by_the_shipped_r0258_guard": sum(
            1 for v in verdicts.values() if v["accepted_by_the_shipped_r0258_guard"]
        ),
        "defects_refused_by_the_r0259_guard": sum(
            1 for v in verdicts.values() if v["refused_by_the_r0259_guard"]
        ),
        "an_honest_install_passes_both_guards": bool(
            honest_static["the_chunk_loop_polls"]
            and honest_shipped["the_chunk_loop_polls"]
        ),
        "an_honest_poll_is_effective": poll_is_effective(_honest_poll),
        "chunk_bound_refuses_a_whole_array_chunk": _chunk_bound_refuses(),
    }


def _chunk_bound_refuses() -> bool:
    try:
        assert_chunk_bounded(2_511_103_254, 2_511_103_254)
    except Round0259Error:
        return True
    return False


def assert_structural_defect_controls_v2() -> dict[str, Any]:
    report = structural_defect_controls_v2()
    if report["defects_refused_by_the_r0259_guard"] != report["defects_planted"]:
        raise Round0259Error(
            "R0259: a planted structural defect passed the strengthened guard"
        )
    if report["defects_accepted_by_the_shipped_r0258_guard"] != report["defects_planted"]:
        raise Round0259Error(
            "R0259: a plant was already refused by the SHIPPED R0258 guard, so it "
            "is not the defect review-0258-01 §D.3 found and proves no change"
        )
    if not report["an_honest_install_passes_both_guards"]:
        raise Round0259Error("R0259: the strengthened guard refuses an honest install")
    if not report["an_honest_poll_is_effective"]:
        raise Round0259Error("R0259: poll_is_effective refuses an honest poll")
    if not report["chunk_bound_refuses_a_whole_array_chunk"]:
        raise Round0259Error("R0259: the chunk bound accepts a whole-array chunk")
    return report


# --------------------------------------------------------------------------- #
# 8. applicability and container positive controls
# --------------------------------------------------------------------------- #

R0243_FUZZY_MANIFEST = (
    "/data/latent-basemap/runs/round-0243/queue/artifacts/"
    "minilm-mixed-100000k-cluster-spill-k15-fuzzy-graph-v1/fuzzy-graph.json"
)


def _wrong_shape_manifests() -> dict[str, dict[str, Any]]:
    """Manifests a widened entry would silently accept."""
    good = {
        "schema": RUNGS[RUNG_100M]["schema"],
        "rows": 100_000_000,
        "k": 15,
        "directed_edges": 2_511_103_254,
    }
    two_m = {
        "schema": RUNGS[RUNG_2M]["schema"],
        "retained_rows": 1_993_761,
        "k": 50,
        "dimension": 768,
    }
    return {
        "the_real_100m_manifest": good,
        "the_real_2m_manifest": two_m,
        "a_100m_schema_with_2m_rows": {**good, "rows": 1_993_761},
        "a_100m_schema_with_k_50": {**good, "k": 50},
        "a_2m_schema_with_100m_rows": {**two_m, "retained_rows": 100_000_000},
        "an_unregistered_schema": {**good, "schema": "round9999-not-a-rung-v1"},
    }


def rung_applicability_controls() -> dict[str, Any]:
    """Prove the entry refuses every config that is not its rung, both ways.

    The expectation for each manifest is written down first, then evaluated, so a
    plant that silently starts passing changes the verdict rather than the prose.
    """
    expected = {
        "the_real_100m_manifest": RUNG_100M,
        "the_real_2m_manifest": RUNG_2M,
        "a_100m_schema_with_2m_rows": "raises",
        "a_100m_schema_with_k_50": "raises",
        "a_2m_schema_with_100m_rows": "raises",
        "an_unregistered_schema": "raises",
    }
    verdicts: dict[str, Any] = {}
    for name, manifest in sorted(_wrong_shape_manifests().items()):
        try:
            resolved: Any = rung_of(manifest)
            raised = None
        except Round0259RungError as error:
            resolved = None
            raised = type(error).__name__
        try:
            assert_rung(manifest, expected=RUNG_100M, entry="run_train_100m")
            entered_100m = True
            entry_error = None
        except Round0259RungError as error:
            entered_100m = False
            entry_error = str(error).split(".")[0]
        verdicts[name] = {
            "expected": expected[name],
            "resolved_rung": resolved,
            "raised": raised,
            "the_100m_entry_accepts_it": entered_100m,
            "the_100m_entry_refusal": entry_error,
            "behaves_as_expected": (
                (resolved == expected[name]) if expected[name] in RUNGS
                else raised == "Round0259RungError"
            ),
        }
    accepted = sorted(
        name for name, v in verdicts.items() if v["the_100m_entry_accepts_it"]
    )
    return {
        "schema": "round0259-rung-applicability-controls-v1",
        "planted": verdicts,
        "manifests_checked": len(verdicts),
        "manifests_that_behave_as_expected": sum(
            1 for v in verdicts.values() if v["behaves_as_expected"]
        ),
        "manifests_the_100m_entry_accepts": accepted,
        "the_100m_entry_accepts_exactly_the_100m_manifest": (
            accepted == ["the_real_100m_manifest"]
        ),
    }


def assert_rung_applicability_controls() -> dict[str, Any]:
    report = rung_applicability_controls()
    if report["manifests_that_behave_as_expected"] != report["manifests_checked"]:
        raise Round0259RungError("R0259: a rung applicability control did not hold")
    if not report["the_100m_entry_accepts_exactly_the_100m_manifest"]:
        raise Round0259RungError(
            "R0259: the 100M entry accepts a config that is not its rung"
        )
    return report


def bulk_npz_is_not_claimed(tmpdir: str) -> dict[str, Any]:
    """The branch must not claim the container every other rung uses.

    Writes a miniature bulk `.npz` of the shape `build_*_index_modal.py` ships,
    runs it through the SHIPPED `load_edge_arrays`, and requires the returned
    arrays to be the bulk ones. This is the positive control for "changes no
    science path for existing rungs".
    """
    from basemap.pumap.parametric_umap.datasets.edge_list_dataset import (
        load_edge_arrays,
    )

    os.makedirs(tmpdir, exist_ok=True)
    bulk = os.path.join(tmpdir, "edges-bulk.npz")
    sources = np.arange(7, dtype=np.int32)
    targets = (np.arange(7, dtype=np.int32) + 1) % 7
    weights = np.linspace(0.1, 0.9, 7).astype(np.float32)
    np.savez(bulk, sources=sources, targets=targets, weights=weights, n_nodes=7)

    layout = streamed_member_layout(bulk)
    got_sources, got_targets, got_weights, n_nodes = load_edge_arrays(bulk)
    return {
        "schema": "round0259-bulk-npz-untouched-v1",
        "bulk_path": bulk,
        "the_branch_claims_it": layout is not None,
        "sources_match": bool(np.array_equal(np.asarray(got_sources), sources)),
        "targets_match": bool(np.array_equal(np.asarray(got_targets), targets)),
        "weights_match": bool(np.array_equal(np.asarray(got_weights), weights)),
        "n_nodes": int(n_nodes),
        "the_bulk_path_is_unchanged": bool(
            layout is None
            and np.array_equal(np.asarray(got_sources), sources)
            and np.array_equal(np.asarray(got_targets), targets)
            and np.array_equal(np.asarray(got_weights), weights)
            and int(n_nodes) == 7
        ),
    }


def container_defect_controls(tmpdir: str) -> dict[str, Any]:
    """Wrong-shape streamed containers must fail loudly, not quietly.

    Five plants, each a container a careless build could produce, each run
    through the shipped `load_edge_arrays` entry point.
    """
    from basemap.pumap.parametric_umap.datasets.edge_list_dataset import (
        load_edge_arrays,
    )

    os.makedirs(tmpdir, exist_ok=True)

    def _build(name: str, *, edges: int, declared: int, dtype, drop: str | None,
               k: int = 15) -> str:
        directory = os.path.join(tmpdir, name)
        os.makedirs(directory, exist_ok=True)
        arrays = {
            "sources": np.arange(edges, dtype=np.int32),
            "targets": (np.arange(edges, dtype=np.int32) + 1) % max(edges, 1),
            "weights": np.linspace(0.1, 0.9, edges).astype(dtype),
        }
        for member, values in arrays.items():
            if drop == member:
                continue
            np.save(os.path.join(directory, MEMBER_FILENAMES[member]), values)
        np.savez(
            os.path.join(directory, MEMBER_FILENAMES["header"]),
            n_nodes=np.int64(edges), k=np.int64(k),
            directed_edges=np.int64(declared),
        )
        return directory

    honest = _build("honest", edges=64, declared=64, dtype=np.float32, drop=None)
    plants = {
        "declared_edge_count_disagrees_with_the_member": _build(
            "wrong-count", edges=64, declared=65, dtype=np.float32, drop=None),
        "weights_member_is_float64": _build(
            "wrong-dtype", edges=64, declared=64, dtype=np.float64, drop=None),
        "weights_member_is_absent": _build(
            "missing-member", edges=64, declared=64, dtype=np.float32,
            drop="weights"),
        "header_declares_a_non_positive_k": _build(
            "bad-k", edges=64, declared=64, dtype=np.float32, drop=None, k=0),
    }
    truncated = os.path.join(tmpdir, "truncated")
    os.makedirs(truncated, exist_ok=True)
    for member, values in (
        ("sources", np.arange(64, dtype=np.int32)),
        ("targets", np.arange(64, dtype=np.int32)),
        ("weights", np.linspace(0.1, 0.9, 60).astype(np.float32)),
    ):
        np.save(os.path.join(truncated, MEMBER_FILENAMES[member]), values)
    np.savez(
        os.path.join(truncated, MEMBER_FILENAMES["header"]),
        n_nodes=np.int64(64), k=np.int64(15), directed_edges=np.int64(64),
    )
    plants["weights_member_is_truncated"] = truncated

    verdicts: dict[str, Any] = {}
    for name, directory in sorted(plants.items()):
        try:
            load_edge_arrays(directory)
            failed_loudly = False
            error = None
        except Round0259ContainerError as raised:
            failed_loudly = True
            error = type(raised).__name__
        except Exception as raised:  # noqa: BLE001
            failed_loudly = False
            error = f"{type(raised).__name__} (not the round's error type)"
        verdicts[name] = {"failed_loudly": failed_loudly, "error": error}

    sources, targets, weights, n_nodes = load_edge_arrays(honest)
    honest_ok = (
        int(n_nodes) == 64
        and int(sources.shape[0]) == 64
        and int(targets.shape[0]) == 64
        and weights is not None
        and str(weights.dtype) == "float32"
        and file_backed(sources)
    )
    return {
        "schema": "round0259-container-defect-controls-v1",
        "planted": verdicts,
        "defects_planted": len(verdicts),
        "defects_that_failed_loudly": sum(
            1 for v in verdicts.values() if v["failed_loudly"]
        ),
        "an_honest_streamed_graph_loads": bool(honest_ok),
        "an_honest_streamed_graph_returns_memmaps": bool(file_backed(sources)),
        "honest_directory": honest,
    }


def assert_container_defect_controls(tmpdir: str) -> dict[str, Any]:
    report = container_defect_controls(tmpdir)
    if report["defects_that_failed_loudly"] != report["defects_planted"]:
        raise Round0259ContainerError(
            "R0259: a wrong-shape streamed container did not fail loudly"
        )
    if not report["an_honest_streamed_graph_loads"]:
        raise Round0259ContainerError(
            "R0259: the loader branch cannot open an honest streamed graph"
        )
    return report


__all__ = [
    "ABORT_POLL_SITE_PAIRWISE",
    "HEADER_KEYS",
    "MAX_CHUNK_ELEMENTS",
    "MEMBER_FILENAMES",
    "MIN_LEAF_ELEMENTS",
    "NUMERIC_SUM_DEFECTS",
    "NUMPY_PAIRWISE_BLOCKSIZE",
    "NUMPY_PAIRWISE_UNROLL",
    "NUMPY_VERSION_VERIFIED",
    "POLLED_FUNCTIONS_V2",
    "R0243_FUZZY_MANIFEST",
    "ROUND_ID",
    "RUNGS",
    "RUNG_100M",
    "RUNG_2M",
    "Round0259ContainerError",
    "Round0259Error",
    "Round0259RungError",
    "STRUCTURAL_DEFECTS_V2",
    "SUBSTRATE_100M_BYTES",
    "SUBSTRATE_100M_PATH",
    "SUBSTRATE_100M_SHAPE",
    "assert_chunk_bounded",
    "assert_chunk_loops_poll_v2",
    "assert_container_defect_controls",
    "assert_pairwise_rule",
    "assert_poll_is_effective",
    "assert_resident",
    "assert_rung",
    "assert_rung_applicability_controls",
    "assert_structural_defect_controls_v2",
    "assert_substrate_dimension",
    "bulk_npz_is_not_claimed",
    "chunk_loop_polls_v2",
    "container_defect_controls",
    "endpoint_residency",
    "file_backed",
    "load_streamed_edge_members",
    "materialise_endpoints",
    "pairwise_sum_controls",
    "pairwise_sum_polled",
    "poll_is_effective",
    "r0230_isinstance_verdict",
    "rung_applicability_controls",
    "rung_of",
    "streamed_member_layout",
    "streamed_member_signature",
    "weight_cdf_fully_polled",
]
