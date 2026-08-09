"""Execute R0237 — the nested 50M rung, its graph, the 100M verdict, the I/O.

Five nodes, one queue, deliberately narrow (no map is trained here):

* `reachability_high_c_25000k` (GPU, seconds) closes the one real gap in the
  100M case on artifacts that already exist. Recall has never been measured
  above `c = 64` anywhere in this program, so "is a `c = 400` graph any good"
  has been an open assumption behind every 100M projection. This node partitions
  R0236's **sealed** 25,000,000-row substrate at `c in (64, 128, 200, 400)` with
  R0226's `_kmeans`/`_assign` imported unmodified and measures the strict
  reachability *ceiling* over R0236's registered 1,000,000-row uniform probe
  against its sealed exact truth. Nothing is assembled and nothing is built. It
  runs FIRST so the 100M answer survives any later failure.
* `assemble_50000k` (GPU lease, CPU work) assembles the 50,000,000-row mixed
  substrate so that R0236's 25,000,000 training rows are its **positional
  prefix**, draws the increment uniformly from the complement of rung 3's
  training AND the inherited reserve rows, inherits that reserve verbatim,
  asserts nesting by hash and by provenance-key containment, publishes all four
  ladder prefix hashes, asserts reserve disjointness, and asserts `>= 99.9%`
  shard coverage per corpus on the union *and* on the increment alone.
* `truth_probe_50000k` (GPU) computes exact brute-force k15 truth for the
  registered uniform probe of 1,000,000 query rows against **all** 50,000,000
  rows, streaming the substrate through the device. Full truth over every row
  costs `~18.8 GPU-h` at this rung and the round says so instead of pretending.
* `ladder_50000k` (GPU) measures the **five-seed imbalance grid** at this `N` at
  `c in (16, 32, 64, 128, 200, 400)`, fits the device law on sealed points,
  selects `c` on the WORST of the five under a guard that applies the registered
  imbalance margin, and builds the one selected cell under a SIGNAL-FREE
  watchdog with the block-layer I/O instrument and a `mincore` page-cache
  residency reading attached.
* `qualify_50000k` (GPU) scores the emitted graph against the probe truth,
  applies the R0215 degree-zero tripwire over **every** row in and out, verifies
  rather than assumes the `min = 0.0` explanation, symmetrises through R0216's
  identical fuzzy law, reports the measured I/O regime without inheriting
  R0236's fitted exponent (review-0236-01 F5: it is a regression on an algebraic
  identity), re-derives every Phase-2 rung, and issues the 100M recommendation.

Nothing here registers a gate, trains a map, or claims an adoption, and nothing
is built at 100,000,000 rows.
"""
from __future__ import annotations

import gc
import json
import os
import shutil
import subprocess
import threading
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from basemap.artifact_identity import expected_input_signature, ordered_array_sha256
from basemap.output_safety import (
    atomic_build_new_file,
    atomic_save_new_npy,
    atomic_save_new_npz,
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from basemap.round0220_cuvs_qualification import (
    graph_validity,
    strict_containment_rows,
    summarize,
    tie_aware_rows,
)
from basemap.round0237_rung4 import (
    CANDIDATE,
    COMPOSITION,
    C_BUILD_MIN,
    C_BUILD_MIN_NOTE,
    DATA_READ_CONTIGUOUS_BYTES_PER_S,
    DATA_READ_FRAGMENTED_BYTES_PER_S,
    DATA_WRITE_BYTES_PER_S,
    DENSITY_DECILES,
    DETERMINISM_NOTE,
    DEVICE_FREE_HEADROOM_BYTES,
    DEVICE_FREE_NOTE,
    DIMENSION,
    DUPLICATE_FAMILY_KTH_COSINE,
    EXCLUDED_SHARDS,
    FEASIBILITY_CAPABILITY,
    FP32_TIE_NOISE_FLOOR,
    FUZZY_RANDOM_STATE_SEED,
    GRANDPARENT_ROUND_ID,
    GRANDPARENT_ROWS,
    GRAPH_CAPABILITY,
    GREAT_GRANDPARENT_ROUND_ID,
    GREAT_GRANDPARENT_ROWS,
    GRAPH_DEGREE,
    GRAPH_K,
    GRAPH_SCHEMA,
    GUARD_DEVICE_BUDGET_BYTES,
    GUARD_IMBALANCE_MARGIN,
    GUARD_NOTE,
    GUARD_SWAP_CONJUNCTION_ANON_BYTES,
    GUARD_SWAP_CONJUNCTION_MEMAVAILABLE_BYTES,
    GUARD_SWAP_CONJUNCTION_NOTE,
    GUARD_SWAP_GROWTH_ABORT_BYTES,
    HOST_ANON_AUTHORITATIVE_SAMPLER,
    HOST_ANON_FIELD_NOTE,
    IMBALANCE_CAPABILITY,
    IMBALANCE_PROBE_CLUSTERS,
    IMBALANCE_PROBE_ROWS,
    IMBALANCE_REPLICATE_SEEDS,
    INCREMENT_BY_CORPUS,
    INTERMEDIATE_GRAPH_DEGREE,
    IO_CAPABILITY,
    IO_INSTRUMENT_NOTE,
    IO_REGIME_NOTE,
    LADDER_CAPABILITY,
    LADDER_PREFIX_ROWS,
    LADDER_RULE,
    LADDER_SCHEMA,
    LAW_GRAPH_DEGREE,
    LAW_HOMOGENEITY_NOTE,
    LAW_INTERMEDIATE_GRAPH_DEGREE,
    LAW_MAX_ITERATIONS,
    LAW_RESIDUAL_MARGIN,
    LAW_SCHEMA,
    MARGIN_NOTE,
    MAX_ITERATIONS,
    MAX_ZERO_DEGREE_ROWS,
    NN_DESCENT_SETTING,
    PAGE_CACHE_BUDGET_BYTES,
    PARENT_COMPOSITION,
    PARENT_ROUND_ID,
    PARENT_ROWS,
    PHASE2_RUNGS,
    PRIMARY_IMBALANCE_SEED,
    RAW_FORMAT,
    REACHABILITY_CAPABILITY,
    REACHABILITY_CLUSTERS,
    REACHABILITY_CONCERN_FLOOR,
    REACHABILITY_NOTE,
    REACHABILITY_ROWS,
    REACHABILITY_SCHEMA,
    REACHABILITY_SEED,
    RECALL_MEAN_FLOOR,
    RECALL_P10_FLOOR,
    RECALL_POPULATION,
    REPLICATE_NOTE,
    RESERVE_NOTE,
    RESERVE_QUERY_ROWS,
    RESERVE_ROWS,
    RESERVE_ROWS_PER_CORPUS,
    ROUND_ID,
    ROWS,
    ROW_POLICY,
    Round0237Error,
    SAMPLE_INTERVAL_S,
    SELECTION_CANDIDATES,
    SELECTION_LAW,
    SELECTION_SEED,
    SPILL,
    STRUCTURAL_POPULATION,
    SUBSTRATE_CAPABILITY,
    SUBSTRATE_SCHEMA,
    TRAILING_FRAGMENT_POLICY,
    TRUTH_AFFORDABILITY_NOTE,
    TRUTH_CAPABILITY,
    TRUTH_METHOD,
    TRUTH_PROBE_ROWS,
    TRUTH_PROBE_SEED,
    TRUTH_SCHEMA,
    ZERO_RECALL_FORENSIC_NOTE,
    ZERO_ROW_POLICY,
    admissible_max_cluster_rows,
    architectural_io,
    assert_nesting,
    assert_no_signal_policy,
    assert_reserve_disjoint,
    carry_distance,
    fit_device_law,
    guard_decision,
    hundred_m_verdict,
    imbalance_tolerance,
    io_hours,
    io_projection,
    io_scaling_fit,
    json_safe,
    physical_io_prediction,
    predicted_substrate_passes,
    provenance_keys,
    reachability_cell_summary,
    replicate_grid_table,
    replicate_summary,
    rung_derivation,
    select_clusters,
    truth_probe_query_rows,
    validate_composition,
    validate_shard_span,
    zero_recall_forensic,
)
from basemap import round0113_prompt_contrast as prompt_contract
from experiments.round0226_nodes import (
    _child_environment,
    _nvidia_smi_device_bytes,
    _nvidia_smi_per_process_bytes,
    _proc_memory_bytes,
    _swap_used_bytes,
)
from experiments.round0227_nodes import CUML_LAUNCHER
from experiments.round0233_nodes import (
    FlagWatchdog,
    _cold_read_rate,
    _draw,
    _extent_count,
    _shards,
)

EMB = "/data/embeddings"
BUILD_SCRIPT = "basemap/round0237_build.py"

REACHABILITY_ACTION = "reachability_high_c_25000k"
ASSEMBLE_ACTION = "assemble_50000k"
TRUTH_ACTION = "truth_probe_50000k"
LADDER_ACTION = "ladder_50000k"
QUALIFY_ACTION = "qualify_50000k"

#: Truth blocking, unchanged from R0236 and independent of `N` by construction.
#: The 76.8 GB substrate is far past the card, so the database streams through in
#: `TRUTH_DB_CHUNK` slices while the 1.54 GB probe query block and the running
#: top-k stay resident. Device: `1.54 + 6.14 + 4.29 + 0.19 GiB` at any `N`.
TRUTH_QUERY_BLOCK = 16_384
TRUTH_DB_CHUNK = 4_000_000
TRUTH_DB_BLOCK = 65_536
COSINE_BLOCK = 32_768
COPY_BLOCK = 250_000
IMBALANCE_PROBE_TIMEOUT_S = 3_600


# --------------------------------------------------------------------------- #
# the I/O instrument — review-0235-01 F6 / B4
# --------------------------------------------------------------------------- #
def _proc_io(pid: int) -> dict[str, int]:
    """One read of `/proc/<pid>/io`. Read-only; never signals, never ptraces.

    `read_bytes` / `write_bytes` are **block-layer** bytes and exclude page-cache
    hits; `rchar` / `wchar` are syscall bytes and exclude memmap faults. Both are
    published because neither alone answers "did the quadratic term reach the
    disk".
    """
    fields = (
        "rchar", "wchar", "syscr", "syscw", "read_bytes", "write_bytes",
        "cancelled_write_bytes",
    )
    out: dict[str, int] = {}
    try:
        with open(f"/proc/{int(pid)}/io", encoding="utf-8") as handle:
            for line in handle:
                key, _, value = line.partition(":")
                if key in fields:
                    out[key] = int(value.strip())
    except (OSError, ValueError):
        return {}
    return out


def _data_block_device_stat() -> dict[str, Any]:
    """Block-layer counters for the device backing `/data`, from sysfs."""
    try:
        stat = os.stat("/data")
        major, minor = os.major(stat.st_dev), os.minor(stat.st_dev)
        path = f"/sys/dev/block/{major}:{minor}/stat"
        with open(path, encoding="utf-8") as handle:
            values = handle.read().split()
        return {
            "device": f"{major}:{minor}",
            "sectors_read": int(values[2]),
            "sectors_written": int(values[6]),
            "bytes_read": int(values[2]) * 512,
            "bytes_written": int(values[6]) * 512,
        }
    except (OSError, IndexError, ValueError):
        return {}


def _mem_available_bytes() -> int:
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return -1


# --------------------------------------------------------------------------- #
# the page-cache instrument — review-0236-01 F5, the flip point measured
# --------------------------------------------------------------------------- #
def _page_cache_resident_bytes(path: str) -> dict[str, Any]:
    """How much of a file is actually resident in page cache, via `mincore(2)`.

    Review-0236-01 F5: the `~52M`-row flip point is `page_cache_budget / 1536`,
    a restatement of a registered `80 GB` assumption that no round has measured.
    This measures it directly instead. It maps the file copy-on-write
    (`MAP_PRIVATE`, never written, so no page is ever copied) and calls
    `mincore`, which reports residency for pages already in the page cache and
    **faults nothing in**; the mapping is closed immediately. It reads, it does
    not allocate, and it touches no GPU.
    """
    import ctypes
    import mmap as mmap_module

    try:
        size = os.path.getsize(path)
    except OSError as error:
        return {"path": str(path), "error": str(error)}
    if size <= 0:
        return {"path": str(path), "bytes": 0, "resident_bytes": 0}
    page = mmap_module.PAGESIZE
    pages = (size + page - 1) // page
    handle = os.open(path, os.O_RDONLY)
    try:
        mapping = mmap_module.mmap(handle, 0, access=mmap_module.ACCESS_COPY)
    except (OSError, ValueError) as error:
        os.close(handle)
        return {"path": str(path), "bytes": int(size), "error": str(error)}
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        vector = (ctypes.c_ubyte * pages)()
        address = ctypes.addressof(ctypes.c_char.from_buffer(mapping))
        rc = libc.mincore(
            ctypes.c_void_p(address), ctypes.c_size_t(size),
            ctypes.cast(vector, ctypes.POINTER(ctypes.c_ubyte)),
        )
        if rc != 0:
            return {
                "path": str(path), "bytes": int(size),
                "error": f"mincore errno {ctypes.get_errno()}",
            }
        resident = int(
            np.frombuffer(bytes(vector), dtype=np.uint8).astype(np.int64).__and__(1).sum()
        )
    except (OSError, ValueError, AttributeError) as error:
        return {"path": str(path), "bytes": int(size), "error": str(error)}
    finally:
        try:
            mapping.close()
        finally:
            os.close(handle)
    resident_bytes = int(resident) * int(page)
    return {
        "path": str(path),
        "bytes": int(size),
        "pages": int(pages),
        "resident_pages": int(resident),
        "resident_bytes": min(resident_bytes, int(size)),
        "resident_fraction": float(resident) / float(pages) if pages else None,
        "page_size": int(page),
        "note": (
            "mincore(2) residency of the substrate file in page cache. This is "
            "the direct measurement of the assumption behind the registered "
            "page-cache budget, which review-0236-01 F5 showed no round had "
            "tested. It faults nothing in."
        ),
    }


# --------------------------------------------------------------------------- #
# the shared-card instrument — new at this rung
# --------------------------------------------------------------------------- #
def _device_free_bytes() -> int:
    """Free device memory, read from `nvidia-smi`. Never an allocation.

    This is a *query*, not a probe: it reads the driver's own accounting and
    allocates nothing on the card. The distinction matters because "never probe
    a wedged GPU" forbids the allocation, not the query.
    """
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=30,
        )
        return int(float(completed.stdout.strip().splitlines()[0])) * 1024 * 1024
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return -1


def _foreign_gpu_processes(exclude_pid: int | None = None) -> list[dict[str, Any]]:
    """Compute processes on the card that are not ours. Recorded, never touched.

    The card is shared: an unrelated project intermittently holds ~6.3 GB here.
    It is never signalled, never killed and never waited on - it is measured, so
    that this round's guard sizes its cell against memory that actually exists.
    """
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory,process_name",
             "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    out: list[dict[str, Any]] = []
    for line in completed.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2 or not parts[0].isdigit():
            continue
        pid = int(parts[0])
        if exclude_pid is not None and pid == int(exclude_pid):
            continue
        try:
            used = int(float(parts[1])) * 1024 * 1024
        except ValueError:
            used = -1
        out.append({
            "pid": pid,
            "used_bytes": used,
            "process_name": parts[2] if len(parts) > 2 else None,
        })
    return out


def _device_free_admission(predicted_device_bytes: float) -> dict[str, Any]:
    """Does the predicted charge fit the memory the card can actually give?

    An ADDITIONAL refusal condition on top of the registered 24 GiB budget. It
    can only refuse a cell the budget admitted; it never admits one the budget
    refused.
    """
    free = _device_free_bytes()
    foreign = _foreign_gpu_processes()
    need = float(predicted_device_bytes) + float(DEVICE_FREE_HEADROOM_BYTES)
    admitted = bool(free < 0 or need <= float(free))
    return {
        "device_free_bytes": int(free),
        "device_free_headroom_bytes": int(DEVICE_FREE_HEADROOM_BYTES),
        "predicted_device_bytes": float(predicted_device_bytes),
        "required_free_bytes": float(need),
        "admitted": admitted,
        "measurement_unavailable": bool(free < 0),
        "foreign_compute_processes": foreign,
        "foreign_device_bytes": int(
            sum(max(0, int(entry["used_bytes"])) for entry in foreign)
        ),
        "reason": (
            None if admitted else
            f"predicted device {need / 1024 ** 3:.2f} GiB (charge + headroom) "
            f"exceeds the {free / 1024 ** 3:.2f} GiB free on the card; "
            f"{len(foreign)} foreign compute process(es) hold "
            f"{sum(max(0, int(e['used_bytes'])) for e in foreign) / 1024 ** 3:.2f} "
            "GiB and are left alone"
        ),
        "note": DEVICE_FREE_NOTE,
    }


class _IoSampler(threading.Thread):
    """Poll a child's `/proc/<pid>/io` and keep the peak of each counter.

    The counters are monotonic, so the peak is the final value; polling exists
    only because `/proc/<pid>/io` vanishes when the process exits. This thread
    reads a file and nothing else - it sends no signal and attaches to nothing,
    which is the whole point after R0232.
    """

    def __init__(self, *, pid: int, poll_s: float = 0.1) -> None:
        super().__init__(daemon=True)
        self._pid = int(pid)
        self._poll = float(poll_s)
        self._halt = threading.Event()
        self.peak: dict[str, int] = {}
        self.samples = 0

    def run(self) -> None:
        while not self._halt.is_set():
            reading = _proc_io(self._pid)
            if reading:
                self.samples += 1
                for key, value in reading.items():
                    self.peak[key] = max(self.peak.get(key, 0), int(value))
            self._halt.wait(self._poll)

    def halt(self) -> None:
        self._halt.set()
        self.join(timeout=10)

    def readings(self) -> dict[str, Any]:
        return {
            "child_io_samples": int(self.samples),
            "child_io_rchar_bytes": self.peak.get("rchar"),
            "child_io_wchar_bytes": self.peak.get("wchar"),
            "child_io_read_bytes": self.peak.get("read_bytes"),
            "child_io_write_bytes": self.peak.get("write_bytes"),
            "child_io_cancelled_write_bytes": self.peak.get("cancelled_write_bytes"),
            "child_io_note": IO_INSTRUMENT_NOTE,
        }


# --------------------------------------------------------------------------- #
# signal-free child supervision — R0233's watchdog, plus the I/O sampler
# --------------------------------------------------------------------------- #
class MemAwareFlagWatchdog(FlagWatchdog):
    """R0233's signal-free watchdog with the conjunctive swap rule R0236 registered.

    Two changes, both registered in advance in `round-0237-2026-08-09.md`:

    * **`MemAvailable` is sampled continuously**, not once before launch. R0236
      sampled it by hand when swap growth was questioned mid-round and had to
      reason from a single reading.
    * **Swap growth alone no longer aborts.** It must coincide with host
      anonymous bytes above `GUARD_SWAP_CONJUNCTION_ANON_BYTES` or `MemAvailable`
      below `GUARD_SWAP_CONJUNCTION_MEMAVAILABLE_BYTES`. R0236 measured swap
      rising `2,111 -> 3,576 MB` while `MemAvailable` held `97.3%` of RAM and the
      largest anonymous consumer box-wide was `372 MB`: page cache filling under
      memmap I/O, evicting cold anonymous pages, with no memory pressure
      anywhere. The standalone 60 GiB anonymous abort is unchanged.

    `swap_only_rule_would_have_fired` records whether the OLD rule would have
    tripped, so the relaxation is auditable rather than invisible.

    It inherits `_trip` unchanged: a flag file, no `os.kill`, no `terminate()`,
    no `kill()`, no ptrace, on any path.
    """

    def __init__(
        self,
        *,
        swap_conjunction_anon_bytes: int = GUARD_SWAP_CONJUNCTION_ANON_BYTES,
        swap_conjunction_mem_available_bytes: int = (
            GUARD_SWAP_CONJUNCTION_MEMAVAILABLE_BYTES
        ),
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._conjunction_anon = int(swap_conjunction_anon_bytes)
        self._conjunction_mem = int(swap_conjunction_mem_available_bytes)
        self.mem_available_min_bytes: int | None = None
        self.swap_only_rule_would_have_fired = False
        self.conjunction_samples = 0

    def run(self) -> None:  # noqa: D401 - overrides the sampling loop, not `_trip`
        while not self._halt.is_set():
            self.sample_once()
            self._halt.wait(self._poll)

    def sample_once(self) -> None:
        """One pass of the loop. Split out so a CPU test can drive it exactly."""
        self.samples += 1
        self.device_peak_bytes = max(
            self.device_peak_bytes, _nvidia_smi_device_bytes()
        )
        self.per_process_peak_bytes = max(
            self.per_process_peak_bytes,
            _nvidia_smi_per_process_bytes(self._pid),
        )
        rss, anon = _proc_memory_bytes(self._pid)
        self.rss_peak_bytes = max(self.rss_peak_bytes, rss)
        self.anon_peak_bytes = max(self.anon_peak_bytes, anon)
        swap = _swap_used_bytes()
        self.swap_peak_bytes = max(self.swap_peak_bytes, swap)
        available = _mem_available_bytes()
        if available >= 0:
            self.mem_available_min_bytes = (
                available if self.mem_available_min_bytes is None
                else min(self.mem_available_min_bytes, available)
            )
        growth = swap - self._swap_baseline
        if growth > self._swap_abort:
            self.swap_only_rule_would_have_fired = True
            pressure = bool(
                anon > self._conjunction_anon
                or (0 <= available < self._conjunction_mem)
            )
            if pressure:
                self.conjunction_samples += 1
                self._trip(
                    f"system swap grew {growth / 1024 ** 3:.2f} GiB over its "
                    f"{self._swap_baseline / 1024 ** 3:.2f} GiB pre-launch "
                    f"baseline AND the box is under memory pressure "
                    f"(child anonymous {anon / 1024 ** 3:.2f} GiB, "
                    f"MemAvailable {available / 1024 ** 3:.2f} GiB)"
                )
        if anon > self._anon_budget:
            self._trip(
                f"child anonymous memory {anon / 1024 ** 3:.2f} GiB exceeds "
                f"the {self._anon_budget / 1024 ** 3:.2f} GiB budget"
            )

    def readings(self) -> dict[str, Any]:
        return {
            **super().readings(),
            "mem_available_min_bytes_sampled": (
                None if self.mem_available_min_bytes is None
                else int(self.mem_available_min_bytes)
            ),
            "swap_only_rule_would_have_fired": bool(
                self.swap_only_rule_would_have_fired
            ),
            "swap_conjunction_trip_samples": int(self.conjunction_samples),
            "swap_conjunction_anon_bytes": int(self._conjunction_anon),
            "swap_conjunction_mem_available_bytes": int(self._conjunction_mem),
            "swap_conjunction_note": GUARD_SWAP_CONJUNCTION_NOTE,
        }


class _HostSampler(threading.Thread):
    """Peak `RssAnon` / minimum `MemAvailable` for an IN-PROCESS stage.

    The build cell has the parent watchdog; the qualification node's fuzzy stage
    has nothing, and at 50,000,000 rows it is the largest host-memory consumer in
    the queue. This samples this process and the box, and does nothing else.
    """

    def __init__(self, *, poll_s: float = 0.5) -> None:
        super().__init__(daemon=True)
        self._poll = float(poll_s)
        self._halt = threading.Event()
        self.rss_peak_bytes = 0
        self.anon_peak_bytes = 0
        self.mem_available_min_bytes: int | None = None
        self.samples = 0

    def run(self) -> None:
        pid = os.getpid()
        while not self._halt.is_set():
            self.samples += 1
            rss, anon = _proc_memory_bytes(pid)
            self.rss_peak_bytes = max(self.rss_peak_bytes, rss)
            self.anon_peak_bytes = max(self.anon_peak_bytes, anon)
            available = _mem_available_bytes()
            if available >= 0:
                self.mem_available_min_bytes = (
                    available if self.mem_available_min_bytes is None
                    else min(self.mem_available_min_bytes, available)
                )
            self._halt.wait(self._poll)

    def halt(self) -> None:
        self._halt.set()
        self.join(timeout=10)

    def readings(self) -> dict[str, Any]:
        return {
            "node_host_rss_peak_bytes": int(self.rss_peak_bytes),
            "node_host_anon_peak_bytes": int(self.anon_peak_bytes),
            "node_mem_available_min_bytes": (
                None if self.mem_available_min_bytes is None
                else int(self.mem_available_min_bytes)
            ),
            "node_host_samples": int(self.samples),
        }


def _run_child(
    *,
    config: Mapping[str, Any],
    out_dir: str,
    cache_root: str,
    repo_root: str,
    guard: Mapping[str, Any],
    timeout_s: float,
) -> dict[str, Any]:
    """Guard, launch, watch, record. Every exit path yields a measurement.

    No signal is delivered to the child at any point, including on timeout: the
    parent sets the cooperative flag and then waits for the child to unwind its
    own CUDA context. `FlagWatchdog` has no `os.kill`/`terminate()`/`kill()`
    path at all, and neither does the I/O sampler.

    review-0235-01 F4: the child's phase-sampled `host_anon_peak_bytes` and the
    parent watchdog's continuous one are kept under DIFFERENT names here, and the
    authoritative one is named in the record.

    Two additions at this rung, both registered in advance and both STRICTER
    than what R0236 ran: free device memory is read immediately before launch
    and the cell is refused if the predicted charge does not fit what the card
    can actually give (it is shared), and the substrate's page-cache residency
    is measured by `mincore` before and after the cell so the I/O regime is a
    measurement rather than an inference from a registered budget.
    """
    identity = {
        "setting_id": str(config["setting_id"]),
        "candidate": CANDIDATE,
        "rows": int(config["rows"]),
        "clusters": int(config["clusters"]),
        "spill": int(config["spill"]),
        "config": dict(config),
    }
    ensure_data_directory(out_dir)
    config_path = os.path.join(out_dir, "config.json")
    atomic_write_new_json(config_path, dict(config), immutable=True)
    if not guard.get("allowed"):
        receipt = {
            **identity,
            "fit": False,
            "oom": False,
            "timed_out": False,
            "cooperatively_aborted": False,
            "refused_a_priori": True,
            "error_type": "RefusedAPriori",
            "guard": dict(guard),
            "refusal_reasons": list(guard.get("refusal_reasons") or []),
            "builder_seconds": None,
            "device_wide_peak_bytes": None,
            "device_wide_peak_over_baseline_bytes": None,
            "nvidia_smi_per_process_peak_bytes": None,
            "child_device_peak_sampled_bytes": None,
            "rmm_peak_bytes": None,
            "host_rss_peak_bytes": None,
            "host_anon_peak_bytes": None,
            "host_anon_peak_bytes_parent_watchdog": None,
            "host_anon_peak_bytes_child_phase_sampled": None,
            "host_anon_authoritative_sampler": HOST_ANON_AUTHORITATIVE_SAMPLER,
            "host_vmhwm_bytes": None,
            "system_swap_growth_bytes": None,
            "peak_scratch_bytes": None,
            "spill_groups": None,
            "substrate_passes": None,
            "watchdog_escalations": [],
            "child_io_read_bytes": None,
            "child_io_write_bytes": None,
            "data_block_device_read_bytes": None,
            "data_block_device_write_bytes": None,
        }
        atomic_write_new_json(
            os.path.join(out_dir, "build-receipt.json"), receipt, immutable=True
        )
        return receipt

    # The card is shared. Size the cell against memory that actually exists.
    admission = _device_free_admission(
        float((guard.get("prediction") or {}).get("predicted_device_bytes")
              or guard.get("predicted_device_bytes") or 0.0)
    )
    if not admission["admitted"]:
        receipt = {
            **identity,
            "fit": False,
            "oom": False,
            "timed_out": False,
            "cooperatively_aborted": False,
            "refused_a_priori": True,
            "error_type": "RefusedDeviceFree",
            "guard": dict(guard),
            "device_free_admission": admission,
            "refusal_reasons": [admission["reason"]],
            "builder_seconds": None,
            "watchdog_escalations": [],
        }
        atomic_write_new_json(
            os.path.join(out_dir, "build-receipt.json"), receipt, immutable=True
        )
        return receipt

    flag_path = os.path.join(out_dir, "abort.flag")
    device_baseline = _nvidia_smi_device_bytes()
    swap_baseline = _swap_used_bytes()
    disk_before = _data_block_device_stat()
    mem_available_before = _mem_available_bytes()
    page_cache_before = _page_cache_resident_bytes(str(config["substrate"]))
    started = time.perf_counter()
    process = subprocess.Popen(
        [
            CUML_LAUNCHER,
            os.path.join(repo_root, BUILD_SCRIPT),
            "--config", config_path,
            "--out", out_dir,
        ],
        cwd=repo_root,
        env=_child_environment(cache_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    watchdog = MemAwareFlagWatchdog(
        flag_path=flag_path,
        pid=process.pid,
        poll_s=0.25,
        host_anon_budget_bytes=int(guard["host_anon_budget_bytes"]),
        swap_growth_abort_bytes=GUARD_SWAP_GROWTH_ABORT_BYTES,
        device_baseline_bytes=device_baseline,
        swap_baseline_bytes=swap_baseline,
    )
    io_sampler = _IoSampler(pid=process.pid)
    watchdog.start()
    io_sampler.start()
    timed_out = False
    try:
        try:
            stdout, stderr = process.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            watchdog._trip(  # noqa: SLF001 - the registered in-band abort path
                f"cell exceeded its {timeout_s:.0f} s deadline; cooperative flag "
                "set, no signal sent"
            )
            stdout, stderr = process.communicate()
    finally:
        io_sampler.halt()
        watchdog.halt()
    readings = watchdog.readings()
    assert_no_signal_policy(readings.get("watchdog_escalations") or [])
    subprocess_seconds = time.perf_counter() - started
    disk_after = _data_block_device_stat()
    page_cache_after = _page_cache_resident_bytes(str(config["substrate"]))

    receipt_path = os.path.join(out_dir, "build-receipt.json")
    if os.path.exists(receipt_path):
        with open(receipt_path, encoding="utf-8") as handle:
            child = json.load(handle)
    else:
        raise Round0237Error(
            f"R0237 child {config['setting_id']} produced no receipt "
            f"(rc={process.returncode}, timed_out={timed_out}):\n"
            f"{stdout[-2000:]}\n{stderr[-2000:]}"
        )
    child_anon = child.get("host_anon_peak_bytes")
    parent_anon = readings.get("host_anon_peak_bytes")
    delta_disk = {
        f"data_block_device_{key}": (
            int(disk_after[key]) - int(disk_before[key])
            if key in disk_after and key in disk_before else None
        )
        for key in ("bytes_read", "bytes_written")
    }
    return {
        **identity,
        **{
            key: value for key, value in child.items()
            if key not in ("config", "setting_id", "host_anon_peak_bytes")
        },
        **readings,
        **io_sampler.readings(),
        "host_anon_peak_bytes_child_phase_sampled": (
            None if child_anon is None else int(child_anon)
        ),
        "host_anon_peak_bytes_parent_watchdog": (
            None if parent_anon is None else int(parent_anon)
        ),
        "host_anon_peak_bytes": (
            None if parent_anon is None else int(parent_anon)
        ),
        "host_anon_authoritative_sampler": HOST_ANON_AUTHORITATIVE_SAMPLER,
        "host_anon_field_note": HOST_ANON_FIELD_NOTE,
        "data_block_device_read_bytes": delta_disk["data_block_device_bytes_read"],
        "data_block_device_write_bytes": delta_disk[
            "data_block_device_bytes_written"
        ],
        "data_block_device_before": disk_before,
        "data_block_device_after": disk_after,
        "mem_available_bytes_before": int(mem_available_before),
        "device_free_admission": admission,
        "substrate_page_cache_before": page_cache_before,
        "substrate_page_cache_after": page_cache_after,
        "subprocess_seconds": subprocess_seconds,
        "refused_a_priori": False,
        "timed_out": bool(timed_out),
        "guard": dict(guard),
        "stderr_tail": stderr[-2000:],
    }


# --------------------------------------------------------------------------- #
# node 0 — the reachability ceiling at high c, on R0236's sealed 25M substrate
# --------------------------------------------------------------------------- #
REACHABILITY_TIMEOUT_S = 3_600
REACHABILITY_PROBE_BLOCK = 100_000


def run_reachability(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    """The one real gap in the 100M case, measured for seconds of GPU.

    Runs entirely on artifacts that already exist: R0236's sealed 25,000,000-row
    substrate, and R0236's sealed exact k15 truth for its registered
    1,000,000-row uniform probe. Nothing is assembled, nothing is built, no new
    bulk bytes are written. For each `c` it partitions with R0226's `_kmeans` and
    `_assign` imported unmodified, then measures what fraction of every probe
    row's 15 exact-truth neighbours share a cluster with it. That fraction is a
    hard ceiling on strict recall for the partition, and review-0236-01 showed
    the builder attains it (`c = 64` at 25M realised `0.9996659` against the 2M
    ceiling `0.9996687`).

    It hands cuVS nothing. The dataset it hands k-means is a read-only memmap and
    the child asserts that before any work.
    """
    manifest = active["manifest"]
    if str(manifest.get("round_id")) != ROUND_ID:
        raise Round0237Error("R0237 handler received another queue")
    repo_root = str(manifest["repo_root"])
    cache_root = str(job["cache_root"])

    parent = prompt_contract.verify_signature(
        dict(job["parent_substrate_manifest"]), label="R0236 substrate manifest"
    )
    parent_sealed = prompt_contract.read_sealed(parent, label="R0236 substrate manifest")
    if str(parent_sealed.get("round_id")) != PARENT_ROUND_ID or int(
        parent_sealed.get("rows", -1)
    ) != REACHABILITY_ROWS:
        raise Round0237Error("R0237 reachability needs R0236's 25M substrate manifest")
    substrate_path = prompt_contract.verify_signature(
        dict(parent_sealed["substrate"]), label="R0236 substrate"
    )
    truth_manifest = prompt_contract.verify_signature(
        dict(job["parent_truth_manifest"]), label="R0236 probe truth manifest"
    )
    truth_sealed = prompt_contract.read_sealed(
        truth_manifest, label="R0236 probe truth manifest"
    )
    if str(truth_sealed.get("round_id")) != PARENT_ROUND_ID or int(
        truth_sealed.get("rows", -1)
    ) != REACHABILITY_ROWS:
        raise Round0237Error("R0237 reachability needs R0236's own probe truth")
    probe_rows_path = prompt_contract.verify_signature(
        dict(truth_sealed["outputs"]["query_rows"]), label="R0236 probe rows"
    )
    truth_ids_path = prompt_contract.verify_signature(
        dict(truth_sealed["outputs"]["ids"]), label="R0236 truth ids"
    )
    probe_rows_count = int(truth_sealed["probe_rows"])

    _reach_path, reach_2m = _read_bound(
        job, "r0229_reachability", label="R0229 spill reachability", sealed=False
    )
    ceilings_2m = {
        int(cell["clusters"]): {
            "strict_ceiling_all_rows": cell.get("strict_ceiling_all_rows"),
            "tie_ceiling_query_sample": cell.get("tie_ceiling_query_sample"),
            "realised_imbalance": cell.get("realised_imbalance"),
        }
        for cell in reach_2m.get("cells") or []
        if int(cell.get("spill") or 0) == SPILL
    }

    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0237 high-c reachability"
    )
    started = time.monotonic()
    script = os.path.join(output, "reachability-probe.py")
    body = f'''
import json, os, sys, time
import numpy as np
sys.path.insert(0, {repo_root!r})
import cupy as cp
import cupyx
from basemap.round0226_cluster_spill_build import _assign, _kmeans

rows = {REACHABILITY_ROWS}
spill = {SPILL}
seed = {REACHABILITY_SEED}
block = {REACHABILITY_PROBE_BLOCK}
substrate = np.load({substrate_path!r}, mmap_mode="r")
assert isinstance(substrate, np.memmap) and not substrate.flags.writeable
dataset = substrate[:rows]
assert isinstance(dataset, np.memmap) and not dataset.flags.writeable
probe = np.load({probe_rows_path!r}).astype(np.int64, copy=False)
truth = np.load({truth_ids_path!r}).astype(np.int64, copy=False)
assert probe.shape == ({probe_rows_count},) and truth.shape[0] == probe.shape[0]
k = int(truth.shape[1])
cells = []
for clusters in {tuple(int(v) for v in REACHABILITY_CLUSTERS)!r}:
    t0 = time.perf_counter()
    centroids = _kmeans(cp, cupyx, dataset, clusters=clusters, seed=seed)
    kmeans_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    assignment = _assign(cp, dataset, centroids, rows=rows, spill=spill)
    assign_s = time.perf_counter() - t0
    del centroids
    cp.get_default_memory_pool().free_all_blocks()
    sizes = np.bincount(assignment.ravel(), minlength=clusters).astype(np.int64)
    t0 = time.perf_counter()
    strict = np.empty(probe.size, dtype=np.float64)
    for begin in range(0, probe.size, block):
        end = min(begin + block, probe.size)
        mine = assignment[probe[begin:end]]
        gathered = assignment[truth[begin:end]]
        shared = (
            gathered[:, :, :, None] == mine[:, None, None, :]
        ).any(axis=3).any(axis=2)
        strict[begin:end] = shared.sum(axis=1) / float(k)
        del mine, gathered, shared
    strict_s = time.perf_counter() - t0
    np.save(os.path.join({output!r}, "strict-c%d.f64.npy" % clusters), strict)
    cells.append({{
        "clusters": int(clusters),
        "spill": spill,
        "seed": seed,
        "rows": rows,
        "k": k,
        "rows_scored": int(probe.size),
        "strict_ceiling_mean": float(strict.mean()),
        "strict_ceiling_p10": float(np.percentile(strict, 10)),
        "strict_ceiling_min": float(strict.min()),
        "fraction_fully_reachable": float((strict >= 1.0).mean()),
        "rows_with_zero_reachable": int((strict <= 0.0).sum()),
        "rows_below_one": int((strict < 1.0).sum()),
        "cluster_sizes": {{
            "min": int(sizes.min()),
            "max": int(sizes.max()),
            "mean": float(sizes.mean()),
            "empty": int((sizes == 0).sum()),
        }},
        "imbalance_max_over_mean": float(sizes.max() / sizes.mean()),
        "kmeans_seconds": float(kmeans_s),
        "assign_seconds": float(assign_s),
        "strict_scan_seconds": float(strict_s),
    }})
    del assignment, sizes, strict
with open(os.path.join({output!r}, "reachability-cells.json"), "w") as handle:
    json.dump({{"cells": cells}}, handle, indent=2, sort_keys=True)
'''
    with open(script, "w", encoding="utf-8") as handle:
        handle.write(body)
    completed = subprocess.run(
        [CUML_LAUNCHER, script], cwd=repo_root,
        env=_child_environment(cache_root), capture_output=True, text=True,
        timeout=REACHABILITY_TIMEOUT_S,
    )
    result_path = os.path.join(output, "reachability-cells.json")
    if completed.returncode != 0 or not os.path.exists(result_path):
        raise Round0237Error(
            f"R0237 reachability scan failed ({completed.returncode}):\n"
            f"{completed.stdout[-2000:]}\n{completed.stderr[-2000:]}"
        )
    with open(result_path, encoding="utf-8") as handle:
        measured = json.load(handle)

    cells: list[dict[str, Any]] = []
    for cell in measured["cells"]:
        clusters = int(cell["clusters"])
        vector_path = os.path.join(output, f"strict-c{clusters}.f64.npy")
        strict = np.load(vector_path, allow_pickle=False)
        summary = reachability_cell_summary(strict, clusters=clusters, spill=SPILL)
        if abs(float(summary["strict_ceiling_mean"])
               - float(cell["strict_ceiling_mean"])) > 1e-12:
            raise Round0237Error(
                f"R0237 reachability c = {clusters}: the child's mean and the "
                "node's recomputation from the saved per-row vector disagree"
            )
        cells.append({
            **cell,
            **summary,
            "strict_vector": expected_input_signature(vector_path),
            "r0229_2m_reference": ceilings_2m.get(clusters),
        })
        del strict

    by_c = {
        int(cell["clusters"]): float(cell["strict_ceiling_mean"]) for cell in cells
    }
    control = next((cell for cell in cells if int(cell["clusters"]) == 64), None)
    receipt = prompt_contract.seal(json_safe({
        "schema": REACHABILITY_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": manifest["release_sha"],
        "capability": REACHABILITY_CAPABILITY,
        "capabilities": [REACHABILITY_CAPABILITY],
        "rows": REACHABILITY_ROWS,
        "spill": SPILL,
        "seed": REACHABILITY_SEED,
        "clusters_scanned": [int(value) for value in REACHABILITY_CLUSTERS],
        "population": (
            "R0236's registered 1,000,000-row uniform probe over its 25,000,000 "
            "rows, scored against R0236's sealed exact k15 truth. Inherited "
            "hash-bound, never re-drawn."
        ),
        "probe_rows": probe_rows_count,
        "method": (
            "R0226's _kmeans and _assign imported unmodified at the registered "
            "A_SEED, then a host-side count of how many of each probe row's 15 "
            "exact-truth neighbours share at least one cluster with it. No "
            "graph is built and cuVS is handed nothing."
        ),
        "strict_ceiling_by_clusters": by_c,
        "concern_floor": REACHABILITY_CONCERN_FLOOR,
        "cells": cells,
        "control": {
            "clusters": 64,
            "note": (
                "c = 64 is the partition R0236 actually BUILT at this N and "
                "scored 0.9981179 strict / 0.9996659 tie-aware on this same "
                "probe. Its ceiling here says how much of that graph's shortfall "
                "is partition loss rather than builder loss, and whether the "
                "ceiling remains a usable predictor above 2,000,000 rows."
            ),
            "measured_strict_ceiling": (
                None if control is None else float(control["strict_ceiling_mean"])
            ),
            "r0236_realised_strict_recall": 0.9981179333333333,
            "r0236_realised_tie_aware_recall": 0.9996659333333333,
        },
        "r0229_2m_s8_reference": ceilings_2m,
        "reachability_note": REACHABILITY_NOTE,
        "substrate": expected_input_signature(substrate_path),
        "parent_substrate_manifest": expected_input_signature(parent),
        "parent_truth_manifest": expected_input_signature(truth_manifest),
        "probe_script": expected_input_signature(script),
        "abort_policy": (
            "none needed: this node builds no graph, holds the card for tens of "
            "seconds per cell, and hands cuVS nothing. No signal is sent to it."
        ),
        "performance": {
            "total_wall_s": time.monotonic() - started,
            "child_seconds": sum(
                float(cell["kmeans_seconds"]) + float(cell["assign_seconds"])
                + float(cell["strict_scan_seconds"]) for cell in cells
            ),
        },
        "training_performed": False,
        "gate_registered": False,
        "adoption_claimed": False,
    }))
    atomic_write_new_json(
        os.path.join(output, "reachability.json"), receipt, immutable=True
    )


# --------------------------------------------------------------------------- #
# node 1 — the nested substrate and the inherited reserve
# --------------------------------------------------------------------------- #
def _verify_source_sizes(job: Mapping[str, Any]) -> str:
    """Re-verify every base-corpus shard byte size against the bound manifest."""
    signature = job.get("source_size_manifest")
    if signature is None:
        raise Round0237Error("R0237 requires the bound source size manifest")
    path = prompt_contract.verify_signature(
        dict(signature), label="R0237 source size manifest"
    )
    with open(path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    drift: list[str] = []
    for _corpus, entry in (manifest.get("corpora") or {}).items():
        for relative, want in (entry.get("shard_sizes") or {}).items():
            actual = os.path.getsize(os.path.join(EMB, relative))
            if actual != int(want):
                drift.append(f"{relative}: {actual} != {want}")
    if drift:
        raise Round0237Error(
            f"R0237 source shards changed size since preparation: {drift[:4]}"
        )
    return path


def _parent_substrate(job: Mapping[str, Any]) -> dict[str, Any]:
    """Read and fully verify R0236's sealed rung-2 substrate manifest."""
    path = prompt_contract.verify_signature(
        dict(job["parent_substrate_manifest"]), label="R0236 substrate manifest"
    )
    sealed = prompt_contract.read_sealed(path, label="R0236 substrate manifest")
    if str(sealed.get("round_id")) != PARENT_ROUND_ID or int(
        sealed.get("rows", -1)
    ) != PARENT_ROWS:
        raise Round0237Error("R0237 parent manifest is not R0236's 25M rung")
    resolved = {
        key: prompt_contract.verify_signature(
            dict(sealed[key]), label=f"R0236 {key}"
        )
        for key in (
            "substrate", "provenance", "reserve_substrate", "reserve_provenance",
            "reserve_query_rows",
        )
    }
    return {"manifest_path": path, "sealed": sealed, "paths": resolved}


def _corpus_offsets(shards: Sequence[tuple[str, int, bool]]) -> np.ndarray:
    return np.concatenate(
        [[0], np.cumsum([rows for _p, rows, _n in shards])]
    ).astype(np.int64)


def _global_rows(records: np.ndarray, index: int, offsets: np.ndarray) -> np.ndarray:
    """Global corpus row ids for one corpus's slice of a provenance array."""
    mask = np.asarray(records["corpus"], dtype=np.int64) == int(index)
    shard = np.asarray(records["shard"], dtype=np.int64)[mask]
    row = np.asarray(records["row"], dtype=np.int64)[mask]
    return offsets[shard] + row


def run_assemble(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    manifest = active["manifest"]
    if str(manifest.get("round_id")) != ROUND_ID:
        raise Round0237Error("R0237 handler received another queue")
    size_manifest_path = _verify_source_sizes(job)
    parent = _parent_substrate(job)
    parent_sealed = parent["sealed"]
    output = create_fresh_directory(str(job["outputs"][0]), label="R0237 substrate")
    started = time.monotonic()

    parent_prov = np.load(parent["paths"]["provenance"], allow_pickle=False)
    parent_reserve_prov = np.load(
        parent["paths"]["reserve_provenance"], allow_pickle=False
    )
    if parent_prov.shape[0] != PARENT_ROWS or parent_reserve_prov.shape[0] != (
        RESERVE_ROWS
    ):
        raise Round0237Error("R0237 parent provenance has the wrong row count")

    provenance = np.empty(ROWS, dtype=parent_prov.dtype)
    provenance[:PARENT_ROWS] = parent_prov
    counts: dict[str, int] = {}
    rejects: dict[str, int] = {}
    spans: dict[str, Any] = {}
    sources: dict[str, Any] = {}
    per_shard_counts: dict[str, Any] = {}
    substrate_path = os.path.join(output, "substrate.f32.npy")

    # ---- draw the increment, per corpus, from the complement of rung 2 ----
    at = PARENT_ROWS
    increment_vectors: dict[str, str] = {}
    staging = ensure_data_directory(os.path.join(output, ".staging"))
    for index, (corpus, want_total) in enumerate(COMPOSITION):
        shards = _shards(corpus)
        total = int(sum(rows for _p, rows, _n in shards))
        sealed_source = (parent_sealed.get("sources") or {}).get(corpus) or {}
        if int(sealed_source.get("corpus_rows", -1)) != total or int(
            sealed_source.get("shards", -1)
        ) != len(shards):
            raise Round0237Error(
                f"{corpus}: shard enumeration differs from R0236's sealed record "
                f"({len(shards)} shards / {total} rows against "
                f"{sealed_source.get('shards')} / {sealed_source.get('corpus_rows')}); "
                "row ids would not mean the same thing and nesting would be a lie"
            )
        offsets = _corpus_offsets(shards)
        parent_train = _global_rows(parent_prov, index, offsets)
        parent_reserve = _global_rows(parent_reserve_prov, index, offsets)
        if parent_train.size != dict(PARENT_COMPOSITION)[corpus]:
            raise Round0237Error(f"{corpus}: parent training rows do not close")
        if parent_reserve.size != RESERVE_ROWS_PER_CORPUS:
            raise Round0237Error(f"{corpus}: parent reserve rows do not close")

        picked = np.zeros(total, dtype=bool)
        picked[parent_train] = True
        picked[parent_reserve] = True
        want = int(INCREMENT_BY_CORPUS[corpus])
        rng = np.random.RandomState(SELECTION_SEED + index)
        selected, vectors, dropped, rounds = _draw(
            shards=shards, offsets=offsets, picked=picked, rng=rng,
            want=want, corpus=f"{corpus}[increment]",
        )
        if np.intersect1d(selected, parent_train).size != 0:
            raise Round0237Error(f"{corpus}: increment overlaps rung 2's training")
        if np.intersect1d(selected, parent_reserve).size != 0:
            raise Round0237Error(f"{corpus}: increment overlaps the inherited reserve")

        shard_of = np.searchsorted(offsets, selected, side="right") - 1
        provenance["corpus"][at:at + want] = index
        provenance["shard"][at:at + want] = shard_of
        provenance["row"][at:at + want] = selected - offsets[shard_of]

        union = np.concatenate([parent_train, selected])
        union_shard = np.searchsorted(offsets, union, side="right") - 1
        spans[corpus] = {
            "union": validate_shard_span(
                corpus=corpus, shards_touched=int(np.unique(union_shard).size),
                shards_total=len(shards), label="union",
            ),
            "increment": validate_shard_span(
                corpus=corpus, shards_touched=int(np.unique(shard_of).size),
                shards_total=len(shards), label="increment",
            ),
            "replacement_rounds": int(rounds),
        }
        per_shard_counts[corpus] = {
            "increment": np.bincount(
                shard_of, minlength=len(shards)
            ).astype(int).tolist(),
            "union": np.bincount(
                union_shard, minlength=len(shards)
            ).astype(int).tolist(),
            "shard_rows": [int(rows) for _p, rows, _n in shards],
        }
        stage_path = os.path.join(staging, f"increment-{index}.npy")
        np.save(stage_path, vectors.astype(np.float32, copy=False))
        increment_vectors[corpus] = stage_path

        counts[corpus] = int(want_total)
        rejects[corpus] = int(dropped)
        sources[corpus] = {
            "shards": len(shards),
            "corpus_rows": total,
            "selected_rows": int(want_total),
            "inherited_rows": dict(PARENT_COMPOSITION)[corpus],
            "increment_rows": want,
            "reserve_rows": RESERVE_ROWS_PER_CORPUS,
            "format": "npy" if shards[0][2] else RAW_FORMAT,
            "first_shard": expected_input_signature(shards[0][0]),
        }
        at += want
        del vectors, shard_of, union, union_shard, picked
        del parent_train, parent_reserve, selected
        gc.collect()
    if at != ROWS:
        raise Round0237Error(f"assembled {at} rows, expected {ROWS}")

    # ---- write: rung 2's bytes verbatim, then the normalised increment ----
    parent_substrate_path = parent["paths"]["substrate"]
    write_io_before = _proc_io(os.getpid())
    write_disk_before = _data_block_device_stat()
    write_started = time.monotonic()

    def _write(tmp: str) -> None:
        train = np.lib.format.open_memmap(
            tmp, mode="w+", dtype=np.float32, shape=(ROWS, DIMENSION)
        )
        source = np.load(parent_substrate_path, mmap_mode="r", allow_pickle=False)
        if source.shape != (PARENT_ROWS, DIMENSION):
            raise Round0237Error("R0236 substrate geometry is not (12500000, 384)")
        for begin in range(0, PARENT_ROWS, COPY_BLOCK):
            end = min(begin + COPY_BLOCK, PARENT_ROWS)
            block = np.asarray(source[begin:end])
            norms = np.linalg.norm(block, axis=1)
            if not np.isfinite(block).all() or float(norms.min()) <= 0:
                raise Round0237Error("R0236 substrate prefix is degenerate")
            if float(np.abs(norms - 1.0).max()) > 1e-4:
                raise Round0237Error(
                    "R0236 substrate prefix is not unit-normalised; renormalising "
                    "it would break the byte-identical nesting this rung asserts"
                )
            # Copied verbatim: renormalising already-normalised rows changes bits.
            train[begin:end] = block
            del block, norms
        cursor = PARENT_ROWS
        for corpus, _want_total in COMPOSITION:
            want = int(INCREMENT_BY_CORPUS[corpus])
            staged = np.load(increment_vectors[corpus], mmap_mode="r")
            if staged.shape != (want, DIMENSION):
                raise Round0237Error(f"{corpus}: staged increment has {staged.shape}")
            for begin in range(0, want, COPY_BLOCK):
                end = min(begin + COPY_BLOCK, want)
                block = np.asarray(staged[begin:end], dtype=np.float32)
                norms = np.linalg.norm(block, axis=1)
                if not np.isfinite(block).all() or float(norms.min()) <= 0:
                    raise Round0237Error(f"{corpus}: increment holds a degenerate row")
                train[cursor + begin:cursor + end] = block / norms[:, None]
                del block, norms
            cursor += want
            del staged
        if cursor != ROWS:
            raise Round0237Error(f"wrote {cursor} rows, expected {ROWS}")
        train.flush()
        del train

    atomic_build_new_file(substrate_path, _write, immutable=True)
    write_seconds = time.monotonic() - write_started
    write_io_after = _proc_io(os.getpid())
    write_disk_after = _data_block_device_stat()
    shutil.rmtree(staging, ignore_errors=True)

    written = np.load(substrate_path, mmap_mode="r", allow_pickle=False)
    prefix_sha = ordered_array_sha256(written[:PARENT_ROWS])
    parent_ordered = str(parent_sealed["ordered_substrate_sha256"])
    if prefix_sha != parent_ordered:
        raise Round0237Error(
            "R0237 substrate prefix is not R0236's bytes: "
            f"{prefix_sha} != {parent_ordered}. The registered nesting is "
            "positional and byte-identical."
        )
    #: One file now carries rungs 1-4, so all four prefix hashes are published.
    #: Every one is computed here from THIS round's bytes; none is transcribed
    #: from another round's prose.
    ladder_prefix_sha = {
        int(rows): ordered_array_sha256(written[:int(rows)])
        for rows in LADDER_PREFIX_ROWS
    }
    grandparent_sha = ladder_prefix_sha[GRANDPARENT_ROWS]
    great_grandparent_sha = ladder_prefix_sha[GREAT_GRANDPARENT_ROWS]
    ordered = ladder_prefix_sha[ROWS]
    if ladder_prefix_sha[PARENT_ROWS] != prefix_sha:
        raise Round0237Error("R0237 prefix hash is not stable across two reads")
    del written

    composition = validate_composition(counts)
    nesting = assert_nesting(parent=parent_prov, child=provenance)
    nesting.update({
        "prefix_ordered_sha256": prefix_sha,
        "parent_ordered_sha256": parent_ordered,
        "byte_identical_prefix": True,
        "parent_substrate": dict(parent_sealed["substrate"]),
        "grandparent": {
            "round_id": GRANDPARENT_ROUND_ID,
            "rows": GRANDPARENT_ROWS,
            "prefix_ordered_sha256": grandparent_sha,
            "declared_ordered_sha256": (
                (parent_sealed.get("nesting") or {}).get("parent_ordered_sha256")
            ),
        },
        "great_grandparent": {
            "round_id": GREAT_GRANDPARENT_ROUND_ID,
            "rows": GREAT_GRANDPARENT_ROWS,
            "prefix_ordered_sha256": great_grandparent_sha,
            "declared_ordered_sha256": (
                ((parent_sealed.get("nesting") or {}).get("grandparent") or {})
                .get("prefix_ordered_sha256")
            ),
        },
        "ladder_prefix_ordered_sha256": {
            str(int(rows)): value for rows, value in ladder_prefix_sha.items()
        },
        "ladder_prefix_note": (
            "R0236's prefix is byte-identical to R0235's and R0235's to "
            "R0233's, so ONE file now carries all four rungs. Every hash here "
            "is computed from this round's own bytes: 6,250,000 must reproduce "
            "R0233's, 12,500,000 R0235's, 25,000,000 R0236's, and the last is "
            "this rung's own ordered_substrate_sha256."
        ),
    })

    prov_path = atomic_save_new_npy(
        os.path.join(output, "provenance.npy"), provenance, immutable=True
    )

    # ---- the reserve, inherited verbatim and copied out of the parent tree ----
    reserve_copies: dict[str, Any] = {}
    for key, name in (
        ("reserve_substrate", "reserve.f32.npy"),
        ("reserve_provenance", "reserve-provenance.npy"),
        ("reserve_query_rows", "reserve-query-rows.i64.npy"),
    ):
        source_path = parent["paths"][key]
        destination = os.path.join(output, name)

        def _copy(tmp: str, _source: str = source_path) -> None:
            shutil.copyfile(_source, tmp)

        atomic_build_new_file(destination, _copy, immutable=True)
        signature = expected_input_signature(destination)
        if signature["sha256"] != dict(parent_sealed[key])["sha256"]:
            raise Round0237Error(f"R0237 inherited {key} copy is not byte-identical")
        reserve_copies[key] = signature

    reserve_prov_copy = np.load(
        reserve_copies["reserve_provenance"]["canonical_path"], allow_pickle=False
    )
    disjoint = assert_reserve_disjoint(training=provenance, reserve=reserve_prov_copy)
    increment_keys = provenance_keys(provenance[PARENT_ROWS:])
    reserve_keys = provenance_keys(reserve_prov_copy)
    increment_overlap = int(np.intersect1d(increment_keys, reserve_keys).size)
    if increment_overlap != 0:
        raise Round0237Error(
            f"R0237 increment captured {increment_overlap} reserve rows"
        )
    del increment_keys, reserve_keys

    reserve_index: dict[str, Any] = {}
    reserve_span: dict[str, Any] = {}
    for index, (corpus, _want) in enumerate(COMPOSITION):
        reserve_index[corpus] = {
            "block_start": int(index * RESERVE_ROWS_PER_CORPUS),
            "block_rows": RESERVE_ROWS_PER_CORPUS,
            "heldout_corpus_rows": RESERVE_ROWS_PER_CORPUS - RESERVE_QUERY_ROWS,
            "heldout_query_rows": RESERVE_QUERY_ROWS,
        }
        mask = np.asarray(reserve_prov_copy["corpus"], dtype=np.int64) == index
        shard_total = int(sources[corpus]["shards"])
        touched = int(np.unique(np.asarray(reserve_prov_copy["shard"])[mask]).size)
        reserve_span[corpus] = {
            "shards_touched": touched,
            "shards_total": shard_total,
            "coverage": touched / float(shard_total),
        }

    read_rate = _cold_read_rate(substrate_path, limit_bytes=8 * 1024 ** 3)
    substrate_bytes = os.path.getsize(substrate_path)
    write_measurement = {
        "artifact_bytes": int(substrate_bytes),
        "wall_seconds": float(write_seconds),
        "effective_bytes_per_s": (
            float(substrate_bytes) / write_seconds if write_seconds > 0 else None
        ),
        "process_io_write_bytes_delta": (
            int(write_io_after.get("write_bytes", 0))
            - int(write_io_before.get("write_bytes", 0))
            if write_io_after and write_io_before else None
        ),
        "block_device_bytes_written_delta": (
            int(write_disk_after["bytes_written"])
            - int(write_disk_before["bytes_written"])
            if write_disk_after and write_disk_before else None
        ),
        "note": (
            "the wall covers reading the 12.5M-row parent prefix and the staged "
            "increment as well as writing 38.4 GB, so effective_bytes_per_s is a "
            "floor on write throughput, not a write benchmark"
        ),
    }
    receipt = prompt_contract.seal(json_safe({
        "schema": SUBSTRATE_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": manifest["release_sha"],
        "capability": SUBSTRATE_CAPABILITY,
        "capabilities": [SUBSTRATE_CAPABILITY],
        "rows": ROWS,
        "dimension": DIMENSION,
        "reserve_rows": RESERVE_ROWS,
        "parent": {
            "round_id": PARENT_ROUND_ID,
            "rows": PARENT_ROWS,
            "manifest": expected_input_signature(parent["manifest_path"]),
        },
        "composition": composition,
        "sources": sources,
        "loading_contract": {
            "raw_format": RAW_FORMAT,
            "row_policy": ROW_POLICY,
            "trailing_fragment_policy": TRAILING_FRAGMENT_POLICY,
        },
        "selection": {
            "seed": SELECTION_SEED,
            "law": SELECTION_LAW,
            "zero_row_policy": ZERO_ROW_POLICY,
            "degenerate_rows_dropped": rejects,
            "shard_span": spans,
            "per_shard_selection_counts": per_shard_counts,
            "excluded_shards": {
                key: value["reason"] for key, value in EXCLUDED_SHARDS.items()
            },
        },
        "nesting": nesting,
        "reserve": {
            "note": RESERVE_NOTE,
            "inherited_from_round": PARENT_ROUND_ID,
            "originally_drawn_by_round": GRANDPARENT_ROUND_ID,
            "rows_per_corpus": RESERVE_ROWS_PER_CORPUS,
            "query_rows_per_corpus": RESERVE_QUERY_ROWS,
            "index": reserve_index,
            "shard_span": reserve_span,
            "disjointness": disjoint,
            "increment_intersection_rows": increment_overlap,
            "enables": ["heldout_recall_at_10", "projection_ffr"],
        },
        "source_size_manifest": expected_input_signature(size_manifest_path),
        "substrate": expected_input_signature(substrate_path),
        "provenance": expected_input_signature(prov_path),
        "reserve_substrate": reserve_copies["reserve_substrate"],
        "reserve_provenance": reserve_copies["reserve_provenance"],
        "reserve_query_rows": reserve_copies["reserve_query_rows"],
        "ordered_substrate_sha256": ordered,
        "substrate_read_measurement": read_rate,
        "substrate_write_measurement": write_measurement,
        "substrate_extents": _extent_count(substrate_path),
        "read_rate_reference": {
            "fragmented_bytes_per_s": DATA_READ_FRAGMENTED_BYTES_PER_S,
            "contiguous_bytes_per_s": DATA_READ_CONTIGUOUS_BYTES_PER_S,
            "write_bytes_per_s": DATA_WRITE_BYTES_PER_S,
            "source": "review-0232-2026-08-09-01",
            "caveat": (
                "review-0233-01 D6 refuted extent COUNT as the discriminating "
                "variable; the fragmented rate is kept as a conservative floor, "
                "not as an extent-count model"
            ),
        },
        "performance": {"total_wall_s": time.monotonic() - started},
        "training_performed": False,
    }))
    atomic_write_new_json(
        os.path.join(output, "substrate.json"), receipt, immutable=True
    )


# --------------------------------------------------------------------------- #
# node 2 — exact k15 truth for the registered uniform probe
# --------------------------------------------------------------------------- #
def _substrate_from_manifest(job: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    path, _observed = _intra(job, "substrate_manifest", label="R0237 substrate manifest")
    sealed = prompt_contract.read_sealed(path, label="R0237 substrate manifest")
    if str(sealed.get("round_id")) != ROUND_ID or int(sealed.get("rows", -1)) != ROWS:
        raise Round0237Error("R0237 substrate manifest is not this round's rung")
    return prompt_contract.verify_signature(
        dict(sealed["substrate"]), label="R0237 substrate"
    ), sealed


def _host_substrate(path: str) -> np.ndarray:
    host = np.load(path, mmap_mode="r", allow_pickle=False)
    if host.shape != (ROWS, DIMENSION) or host.dtype != np.float32:
        raise Round0237Error(f"R0237 substrate geometry is {host.shape}/{host.dtype}")
    return host


def run_truth(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    import torch

    manifest = active["manifest"]
    substrate_path, _sealed = _substrate_from_manifest(job)
    output = create_fresh_directory(str(job["outputs"][0]), label="R0237 probe truth")
    started = time.monotonic()

    host = _host_substrate(substrate_path)
    if int(job["probe_rows"]) != TRUTH_PROBE_ROWS or int(
        job["probe_seed"]
    ) != TRUTH_PROBE_SEED:
        raise Round0237Error(
            "R0237 queue declares a probe this release did not register: "
            f"{job['probe_rows']} rows at seed {job['probe_seed']} against "
            f"{TRUTH_PROBE_ROWS} at {TRUTH_PROBE_SEED}"
        )
    probe_rows = truth_probe_query_rows(
        rows=ROWS, size=TRUTH_PROBE_ROWS, seed=TRUTH_PROBE_SEED
    )
    if probe_rows.shape != (TRUTH_PROBE_ROWS,):
        raise Round0237Error(
            f"R0237 probe has {probe_rows.shape} rows, registered {TRUTH_PROBE_ROWS}"
        )
    if int(np.unique(probe_rows).size) != TRUTH_PROBE_ROWS:
        raise Round0237Error("R0237 probe holds a duplicated query row")
    if int(probe_rows.min()) < 0 or int(probe_rows.max()) >= ROWS:
        raise Round0237Error("R0237 probe holds an out-of-range query row")
    if not np.array_equal(probe_rows, np.sort(probe_rows)):
        raise Round0237Error("R0237 probe rows are not ascending")

    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    queries = torch.from_numpy(
        np.ascontiguousarray(host[probe_rows], dtype=np.float32)
    ).to(device)
    width = GRAPH_K + 1
    best_s = torch.full((TRUTH_PROBE_ROWS, width), -float("inf"), device=device)
    best_i = torch.full(
        (TRUTH_PROBE_ROWS, width), -1, device=device, dtype=torch.int64
    )
    search_started = time.monotonic()
    for chunk_start in range(0, ROWS, TRUTH_DB_CHUNK):
        chunk_end = min(chunk_start + TRUTH_DB_CHUNK, ROWS)
        database = torch.empty(
            (chunk_end - chunk_start, DIMENSION), dtype=torch.float32, device=device
        )
        for begin in range(chunk_start, chunk_end, COPY_BLOCK):
            end = min(begin + COPY_BLOCK, chunk_end)
            database[begin - chunk_start:end - chunk_start] = torch.from_numpy(
                np.ascontiguousarray(host[begin:end])
            ).to(device)
        for qs in range(0, TRUTH_PROBE_ROWS, TRUTH_QUERY_BLOCK):
            qe = min(qs + TRUTH_QUERY_BLOCK, TRUTH_PROBE_ROWS)
            query = queries[qs:qe]
            block_s = best_s[qs:qe]
            block_i = best_i[qs:qe]
            for cs in range(0, chunk_end - chunk_start, TRUTH_DB_BLOCK):
                ce = min(cs + TRUTH_DB_BLOCK, chunk_end - chunk_start)
                sims = query @ database[cs:ce].T
                take = min(width, ce - cs)
                top_s, top_i = torch.topk(sims, take, dim=1)
                merged_s = torch.cat([block_s, top_s], 1)
                merged_i = torch.cat(
                    [block_i, top_i.to(torch.int64) + chunk_start + cs], 1
                )
                order = torch.argsort(merged_s, dim=1, descending=True)[:, :width]
                block_s = torch.gather(merged_s, 1, order)
                block_i = torch.gather(merged_i, 1, order)
                del sims, top_s, top_i, merged_s, merged_i, order
            best_s[qs:qe] = block_s
            best_i[qs:qe] = block_i
            del block_s, block_i
        del database
        torch.cuda.empty_cache()
    search_s = time.monotonic() - search_started
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    block_ids = best_i.cpu().numpy()
    block_cos = best_s.cpu().numpy()
    del best_s, best_i, queries
    torch.cuda.empty_cache()
    gc.collect()

    is_self = block_ids == probe_rows[:, None]
    keep = np.argsort(is_self, axis=1, kind="stable")
    ids = np.take_along_axis(block_ids, keep, axis=1)[:, :GRAPH_K].astype(np.int32)
    cosines = np.take_along_axis(block_cos, keep, axis=1)[:, :GRAPH_K].astype(
        np.float32
    )
    del block_ids, block_cos, is_self, keep

    if int((ids.astype(np.int64) == probe_rows[:, None]).sum()) != 0:
        raise Round0237Error("R0237 probe truth retained a self edge")
    if int(ids.min()) < 0 or int(ids.max()) >= ROWS:
        raise Round0237Error("R0237 probe truth holds an out-of-range id")
    if not np.isfinite(cosines).all():
        raise Round0237Error("R0237 probe truth cosines are not finite")
    if int((np.diff(cosines, axis=1) > 1e-5).sum()) != 0:
        raise Round0237Error("R0237 probe truth is not descending in cosine")

    rows_path = atomic_save_new_npy(
        os.path.join(output, "probe-query-rows.i64.npy"), probe_rows, immutable=True
    )
    ids_path = atomic_save_new_npy(
        os.path.join(output, "truth-k15-ids.i32.npy"), ids, immutable=True
    )
    cos_path = atomic_save_new_npy(
        os.path.join(output, "truth-k15-cos.f32.npy"), cosines, immutable=True
    )
    receipt = prompt_contract.seal(json_safe({
        "schema": TRUTH_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": manifest["release_sha"],
        "capability": TRUTH_CAPABILITY,
        "capabilities": [TRUTH_CAPABILITY],
        "rows": ROWS,
        "probe_rows": TRUTH_PROBE_ROWS,
        "probe_seed": TRUTH_PROBE_SEED,
        "probe_fraction": TRUTH_PROBE_ROWS / float(ROWS),
        "k": GRAPH_K,
        "method": TRUTH_METHOD,
        "population": RECALL_POPULATION,
        "affordability": TRUTH_AFFORDABILITY_NOTE,
        "query_block": TRUTH_QUERY_BLOCK,
        "database_chunk": TRUTH_DB_CHUNK,
        "database_block": TRUTH_DB_BLOCK,
        "outputs": {
            "query_rows": expected_input_signature(rows_path),
            "ids": expected_input_signature(ids_path),
            "cosines": expected_input_signature(cos_path),
        },
        "kth_cosine": {
            "mean": float(cosines[:, GRAPH_K - 1].mean()),
            "min": float(cosines[:, GRAPH_K - 1].min()),
            "max": float(cosines[:, GRAPH_K - 1].max()),
        },
        "performance": {
            "exact_search_s": search_s,
            "total_wall_s": time.monotonic() - started,
            "peak_allocated_bytes": peak_allocated,
        },
        "training_performed": False,
    }))
    atomic_write_new_json(
        os.path.join(output, "probe-k15-truth.json"), receipt, immutable=True
    )


# --------------------------------------------------------------------------- #
# node 3 — the replicate grid, the law, c selection, one build cell
# --------------------------------------------------------------------------- #
def _measure_imbalance_grid(
    *,
    substrate_path: str,
    output: str,
    repo_root: str,
    cache_root: str,
) -> tuple[dict[str, Any], dict[int, dict[int, dict[int, float]]]]:
    """The replicate grid: `seeds x nested N x c`, on THIS substrate.

    Runs as its own short child under the RAPIDS env, because k-means and the
    spill assignment live there. It builds no graph and hands cuVS nothing; the
    dataset it touches is a read-only memmap and the child asserts that before
    any k-means runs.
    """
    probe_dir = ensure_data_directory(os.path.join(output, "imbalance"))
    script = os.path.join(probe_dir, "probe.py")
    body = f'''
import json, os, sys, time
import numpy as np
sys.path.insert(0, {repo_root!r})
import cupy as cp
import cupyx
from basemap.round0226_cluster_spill_build import _assign, _kmeans

spill = {SPILL}
substrate = np.load({substrate_path!r}, mmap_mode="r")
assert isinstance(substrate, np.memmap) and not substrate.flags.writeable
cells = []
for rows in {tuple(int(v) for v in IMBALANCE_PROBE_ROWS)!r}:
    dataset = substrate[:rows]
    assert isinstance(dataset, np.memmap) and not dataset.flags.writeable
    for clusters in {tuple(int(v) for v in IMBALANCE_PROBE_CLUSTERS)!r}:
        for seed in {tuple(int(v) for v in IMBALANCE_REPLICATE_SEEDS)!r}:
            started = time.perf_counter()
            centroids = _kmeans(cp, cupyx, dataset, clusters=clusters, seed=seed)
            assignment = _assign(cp, dataset, centroids, rows=rows, spill=spill)
            sizes = np.bincount(assignment.ravel(), minlength=clusters).astype(np.int64)
            cells.append({{
                "rows": int(rows),
                "clusters": int(clusters),
                "seed": int(seed),
                "spill": spill,
                "min": int(sizes.min()),
                "max": int(sizes.max()),
                "mean": float(sizes.mean()),
                "median": float(np.median(sizes)),
                "empty_clusters": int((sizes == 0).sum()),
                "imbalance_max_over_mean": float(sizes.max() / sizes.mean()),
                "elapsed_s": float(time.perf_counter() - started),
            }})
            del centroids, assignment, sizes
            cp.get_default_memory_pool().free_all_blocks()
with open(os.path.join({probe_dir!r}, "imbalance.json"), "w") as handle:
    json.dump({{"spill": spill, "cells": cells}}, handle, indent=2, sort_keys=True)
'''
    with open(script, "w", encoding="utf-8") as handle:
        handle.write(body)
    started = time.monotonic()
    completed = subprocess.run(
        [CUML_LAUNCHER, script], cwd=repo_root,
        env=_child_environment(cache_root), capture_output=True, text=True,
        timeout=IMBALANCE_PROBE_TIMEOUT_S,
    )
    result_path = os.path.join(probe_dir, "imbalance.json")
    if completed.returncode != 0 or not os.path.exists(result_path):
        raise Round0237Error(
            f"R0237 imbalance grid failed ({completed.returncode}):\n"
            f"{completed.stdout[-2000:]}\n{completed.stderr[-2000:]}"
        )
    with open(result_path, encoding="utf-8") as handle:
        measured = json.load(handle)
    grid: dict[int, dict[int, dict[int, float]]] = {}
    for cell in measured["cells"]:
        grid.setdefault(int(cell["rows"]), {}).setdefault(
            int(cell["clusters"]), {}
        )[int(cell["seed"])] = float(cell["imbalance_max_over_mean"])
    measured.update({
        "method": (
            "R0226's _kmeans and _assign, imported unmodified, with 25 Lloyd "
            "iterations on a 1,000,000-row subsample; imbalance is "
            "max(bincount(assignment)) / (spill * N / c). The seed selects both "
            "the subsample and the initial centroids."
        ),
        "replicate_seeds": [int(value) for value in IMBALANCE_REPLICATE_SEEDS],
        "primary_seed": PRIMARY_IMBALANCE_SEED,
        "probed_rows": [int(value) for value in IMBALANCE_PROBE_ROWS],
        "probed_clusters": [int(value) for value in IMBALANCE_PROBE_CLUSTERS],
        "note": REPLICATE_NOTE,
        "wall_s": time.monotonic() - started,
        "summary": {
            str(rows): {
                str(clusters): replicate_summary(seeds)
                for clusters, seeds in cells.items()
            }
            for rows, cells in grid.items()
        },
    })
    return measured, grid


def _read_bound(
    job: Mapping[str, Any], key: str, *, label: str, sealed: bool
) -> tuple[str, dict[str, Any]]:
    """Read one hash-bound inherited artifact."""
    path = prompt_contract.verify_signature(dict(job[key]), label=label)
    if sealed:
        return path, prompt_contract.read_sealed(path, label=label)
    with open(path, encoding="utf-8") as handle:
        return path, json.load(handle)


def _inherited_law_points(job: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Every sealed `gd 64 / igd 256 / it 40` device point that exists.

    Read from sealed artifacts and tagged with the setting the receipt itself
    records, so `admit_law_point` can enforce homogeneity rather than trust a
    label (review-0233-01 D3).
    """
    points: list[dict[str, Any]] = []

    _sweep_path, sweep = _read_bound(
        job, "r0229_sweep", label="R0229 nnd quality sweep", sealed=False
    )
    for cell in sweep.get("cells") or []:
        memory = cell.get("memory") or {}
        points.append({
            "source": f"R0229 sweep {cell.get('cell')}",
            "cell": cell.get("cell"),
            "rows": sweep.get("rows"),
            "clusters": cell.get("clusters"),
            "spill": cell.get("spill"),
            "graph_degree": cell.get("graph_degree"),
            "intermediate_graph_degree": cell.get("intermediate_graph_degree"),
            "max_iterations": cell.get("max_iterations"),
            "max_cluster_rows": (cell.get("cluster_sizes") or {}).get("max"),
            "device_bytes": memory.get("device_wide_peak_over_baseline_bytes"),
        })

    _arm_path, arm = _read_bound(
        job, "r0229_arm", label="R0229 adopted arm build", sealed=True
    )
    build = arm.get("build") or {}
    points.append({
        "source": "R0229 adopted arm (c=200, s=8) @ 2M",
        "cell": "spill-lifted",
        "rows": build.get("rows"),
        "clusters": build.get("clusters"),
        "spill": build.get("spill"),
        "graph_degree": build.get("graph_degree"),
        "intermediate_graph_degree": build.get("intermediate_graph_degree"),
        "max_iterations": build.get("max_iterations"),
        "max_cluster_rows": (build.get("cluster_sizes") or {}).get("max"),
        "device_bytes": build.get("device_wide_peak_over_baseline_bytes"),
    })

    for key, label in (
        ("r0233_ladder", "R0233 build ladder"),
        ("r0235_ladder", "R0235 build ladder"),
        ("r0236_ladder", "R0236 build ladder"),
    ):
        _ladder_path, ladder = _read_bound(job, key, label=label, sealed=True)
        setting = ladder.get("nn_descent") or {}
        for entry in ladder.get("ladder") or []:
            cell_build = entry.get("build") or {}
            if not cell_build.get("fit"):
                continue
            points.append({
                "source": f"{label} {entry.get('cell')}",
                "cell": entry.get("cell"),
                "rows": ladder.get("rows"),
                "clusters": entry.get("clusters"),
                "spill": entry.get("spill"),
                "graph_degree": setting.get("graph_degree"),
                "intermediate_graph_degree": setting.get(
                    "intermediate_graph_degree"
                ),
                "max_iterations": setting.get("max_iterations"),
                "max_cluster_rows": (cell_build.get("cluster_sizes") or {}).get("max"),
                "device_bytes": cell_build.get(
                    "device_wide_peak_over_baseline_bytes"
                ),
            })
    return points


def _inherited_imbalance(job: Mapping[str, Any]) -> dict[str, Any]:
    """Sealed single-realisation `s = 8` imbalance at 2M (R0229)."""
    reach_path, reach = _read_bound(
        job, "r0229_reachability", label="R0229 spill reachability", sealed=False
    )
    at_2m = {
        int(cell["clusters"]): float(cell["realised_imbalance"])
        for cell in reach.get("cells") or []
        if int(cell.get("spill") or 0) == SPILL
        and cell.get("realised_imbalance") is not None
    }
    return {
        "by_rows": {int(reach["rows"]): at_2m},
        "sources": {
            str(int(reach["rows"])): expected_input_signature(reach_path),
        },
        "note": (
            "2,000,000 is the one N no substrate in this ladder holds as a "
            "prefix, so its cells stay single-realisation and are reported "
            "as such"
        ),
    }


def _inherited_replicate_grid(
    job: Mapping[str, Any],
) -> tuple[dict[int, dict[int, dict[int, float]]], dict[str, Any]]:
    """R0236's sealed 54-cell replicate grid at 6.25M / 12.5M / 25M.

    Read from the hash-bound ladder rather than recomputed. Review-0236-01
    released this artifact as a MEASUREMENT (blocking only the round's reading of
    it), so the cells are exactly as reusable as they are cheap to reuse: three
    seeds at each of three nested `N`, on prefixes of the very file this round
    extends. Recomputing them would cost GPU time and could not produce a
    different answer, because the substrate prefix is byte-identical.
    """
    path, ladder = _read_bound(job, "r0236_ladder", label="R0236 build ladder", sealed=True)
    grid: dict[int, dict[int, dict[int, float]]] = {}
    for cell in (ladder.get("measured_imbalance") or {}).get("cells") or []:
        grid.setdefault(int(cell["rows"]), {}).setdefault(
            int(cell["clusters"]), {}
        )[int(cell["seed"])] = float(cell["imbalance_max_over_mean"])
    return grid, {
        "source": expected_input_signature(path),
        "rows_measured": sorted(grid),
        "seeds": sorted({
            seed for cells in grid.values() for seeds in cells.values()
            for seed in seeds
        }),
        "note": (
            "R0236's sealed replicate grid, merged rather than recomputed. Its "
            "three N are byte-identical prefixes of this round's substrate, so "
            "the merged table is one file measured by one code path at four N."
        ),
    }


def _inherited_io_points(job: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Architectural substrate reads from the sealed 6.25M / 12.5M / 25M ladders.

    review-0236-01 F5: `substrate_read_bytes` is IDENTICALLY
    `substrate_passes x N x 1536` in every one of these points and none of them
    carries a physical measurement, so they describe the ceiling function on
    spill-group packing and nothing else. They are carried for continuity and
    are labelled `measured_physical_io: false`; no exponent fitted across them
    is cited here as a measured scaling.
    """
    points: list[dict[str, Any]] = []
    for key, label in (
        ("r0233_ladder", "R0233"),
        ("r0235_ladder", "R0235"),
        ("r0236_ladder", "R0236"),
    ):
        _path, ladder = _read_bound(job, key, label=f"{label} build ladder", sealed=True)
        for entry in ladder.get("ladder") or []:
            build = entry.get("build") or {}
            if not build.get("fit") or not build.get("substrate_read_bytes"):
                continue
            points.append({
                "source": f"{label} {entry.get('cell')}",
                "rows": int(ladder["rows"]),
                "clusters": int(entry["clusters"]),
                "spill": int(entry.get("spill") or SPILL),
                "substrate_passes": build.get("substrate_passes"),
                "spill_groups": build.get("spill_groups"),
                "substrate_read_bytes": int(build["substrate_read_bytes"]),
                "spill_write_bytes": build.get("spill_write_bytes"),
                "spill_write_seconds": (build.get("phases") or {}).get(
                    "spill_write_seconds"
                ),
                "builder_seconds": build.get("builder_seconds"),
                "measured_physical_io": False,
            })
    return points


def run_ladder(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    manifest = active["manifest"]
    repo_root = str(manifest["repo_root"])
    substrate_path, _sealed = _substrate_from_manifest(job)
    output = create_fresh_directory(str(job["outputs"][0]), label="R0237 build ladder")
    scratch_root = ensure_data_directory(str(job["scratch_root"]))
    cache_root = str(job["cache_root"])
    started = time.monotonic()

    inherited_points = _inherited_law_points(job)
    inherited_law = fit_device_law(inherited_points, label="all-sealed-gd64-igd256-it40")
    laws = [inherited_law]

    # R0236 lost a 48.7-minute queue to `too many values to unpack` here, and
    # `check_undefined_names.py` catches neither shadowing nor arity. Assert the
    # shape of what is unpacked, at the call site, before unpacking it.
    grid_result = _measure_imbalance_grid(
        substrate_path=substrate_path, output=output, repo_root=repo_root,
        cache_root=cache_root,
    )
    if not isinstance(grid_result, tuple) or len(grid_result) != 2:
        raise Round0237Error(
            "R0237 imbalance grid returned "
            f"{type(grid_result).__name__} of length "
            f"{len(grid_result) if isinstance(grid_result, tuple) else 'n/a'}; "
            "this call site unpacks exactly two values"
        )
    imbalance, grid = grid_result
    at_this_rung = grid.get(ROWS) or {}
    if not at_this_rung:
        raise Round0237Error(
            f"R0237 imbalance grid measured nothing at N = {ROWS}"
        )
    for clusters, seeds in at_this_rung.items():
        if len(seeds) != len(IMBALANCE_REPLICATE_SEEDS):
            raise Round0237Error(
                f"R0237 c = {clusters} has {len(seeds)} replicate realisations, "
                f"registered {len(IMBALANCE_REPLICATE_SEEDS)}"
            )
    measured = {
        clusters: float(seeds[PRIMARY_IMBALANCE_SEED])
        for clusters, seeds in at_this_rung.items()
        if PRIMARY_IMBALANCE_SEED in seeds
    }
    if not measured:
        raise Round0237Error("R0237 measured no primary-seed imbalance at this rung")
    #: The selection is taken on the WORST realisation across replicate seeds,
    #: never the primary alone and never the mean: five seeds now exist and using
    #: the best of them would be picking a favourable draw. Taking the worst of
    #: five is stricter than R0236's worst of three. The primary-seed column is
    #: published beside it so the ladder table stays like-for-like.
    worst = {
        clusters: float(max(seeds.values()))
        for clusters, seeds in at_this_rung.items()
    }
    selectable = {
        c: value for c, value in worst.items() if c in set(SELECTION_CANDIDATES)
    }
    selection = select_clusters(
        rows=ROWS, measured_imbalance=selectable, laws=laws,
        candidates=SELECTION_CANDIDATES, c_min=C_BUILD_MIN,
    )
    selected_clusters = int(selection["selected_clusters"])
    capacity_rows = int(admissible_max_cluster_rows(laws))

    statvfs = os.statvfs("/data")
    disk_free = int(statvfs.f_bavail) * int(statvfs.f_frsize)

    cells: list[dict[str, Any]] = []
    stopped: str | None = None
    for clusters in (selected_clusters,):
        cell_id = f"r0237-n{ROWS}-c{clusters}-s{SPILL}"
        guard = guard_decision(
            rows=ROWS, clusters=clusters, imbalance=worst[clusters],
            imbalance_source=(
                f"worst of {len(IMBALANCE_REPLICATE_SEEDS)} replicate seeds "
                "measured on this substrate at this N"
            ),
            laws=laws, disk_free_bytes=disk_free,
        )
        config = {
            "setting_id": cell_id,
            "cell": cell_id,
            "candidate": CANDIDATE,
            "rows": ROWS,
            "clusters": int(clusters),
            "spill": SPILL,
            "dimension": DIMENSION,
            "k": GRAPH_K,
            "substrate": substrate_path,
            "emit_graph": True,
            "scratch_root": scratch_root,
            "sample_interval_s": SAMPLE_INTERVAL_S,
            "graph_degree": GRAPH_DEGREE,
            "intermediate_graph_degree": INTERMEDIATE_GRAPH_DEGREE,
            "max_iterations": MAX_ITERATIONS,
            "cluster_capacity_rows": capacity_rows,
        }
        cell_dir = os.path.join(output, "builds", cell_id)
        config["abort_flag"] = os.path.join(cell_dir, "abort.flag")
        record = _run_child(
            config=config, out_dir=cell_dir, cache_root=cache_root,
            repo_root=repo_root, guard=guard,
            timeout_s=float(job["build_timeout_s"]),
        )
        ids_path = os.path.join(cell_dir, "graph-k15-ids.i32.npy")
        cells.append({
            "cell": cell_id, "clusters": int(clusters), "spill": SPILL,
            "role": "selected", "guard": guard, "run": True, "build": record,
            "graph_ids": (
                expected_input_signature(ids_path)
                if os.path.exists(ids_path) else None
            ),
            "graph_cos": (
                expected_input_signature(
                    os.path.join(cell_dir, "graph-k15-cos.f32.npy")
                )
                if os.path.exists(
                    os.path.join(cell_dir, "graph-k15-cos.f32.npy")
                ) else None
            ),
        })
        if not record.get("fit"):
            stopped = cell_id
    disk_free_after = os.statvfs("/data")
    receipt = prompt_contract.seal(json_safe({
        "schema": LADDER_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": manifest["release_sha"],
        "capability": LADDER_CAPABILITY,
        "capabilities": [LADDER_CAPABILITY, IMBALANCE_CAPABILITY],
        "rows": ROWS,
        "spill": SPILL,
        "nn_descent_setting": NN_DESCENT_SETTING,
        "nn_descent": {
            "graph_degree": GRAPH_DEGREE,
            "intermediate_graph_degree": INTERMEDIATE_GRAPH_DEGREE,
            "max_iterations": MAX_ITERATIONS,
        },
        "measured_imbalance": imbalance,
        "primary_seed_imbalance_at_this_rung": measured,
        "worst_seed_imbalance_at_this_rung": worst,
        "selection_imbalance_rule": (
            "c is selected on the WORST of the three replicate realisations at "
            "this N, never the best and never the mean; the primary-seed values "
            "are published beside it so the drift table stays like-for-like"
        ),
        "inherited_device_law": inherited_law,
        "cluster_capacity_rows": capacity_rows,
        "cluster_capacity_note": (
            "derived from the law fitted on every sealed gd64/igd256/it40 point "
            "plus the registered margins, and passed to the build child in its "
            "config.json where it is hash-bound (review-0233-01 D1)"
        ),
        "guard_note": GUARD_NOTE,
        "cluster_selection": selection,
        "c_build_min": C_BUILD_MIN,
        "c_build_min_note": C_BUILD_MIN_NOTE,
        "ladder_rule": LADDER_RULE,
        "ladder": cells,
        "ladder_stopped_at": stopped,
        "host_anon_field_note": HOST_ANON_FIELD_NOTE,
        "host_anon_authoritative_sampler": HOST_ANON_AUTHORITATIVE_SAMPLER,
        "io_instrument_note": IO_INSTRUMENT_NOTE,
        "mem_available_bytes": _mem_available_bytes(),
        "page_cache_budget_bytes": PAGE_CACHE_BUDGET_BYTES,
        "disk_free_bytes_before": disk_free,
        "disk_free_bytes_after": int(disk_free_after.f_bavail)
        * int(disk_free_after.f_frsize),
        "abort_policy": (
            "in-band cooperative flag only; no SIGTERM, no SIGKILL, no ptrace"
        ),
        "determinism_note": DETERMINISM_NOTE,
        "performance": {"total_wall_s": time.monotonic() - started},
        "training_performed": False,
    }))
    atomic_write_new_json(
        os.path.join(output, "build-ladder.json"), receipt, immutable=True
    )


# --------------------------------------------------------------------------- #
# node 4 — qualification, the fuzzy graph, the drift table, the I/O scaling
# --------------------------------------------------------------------------- #
def _candidate_cosines(
    *, ids: np.ndarray, host: np.ndarray, torch: Any, device: Any
) -> np.ndarray:
    """Recompute every candidate edge's cosine from the sealed substrate.

    Independent of the builder's own accumulator, which is the point: an in-node
    probe that shares the builder's arithmetic is not evidence (review-0216-01).
    """
    out = np.empty(ids.shape, dtype=np.float32)
    for begin in range(0, ROWS, COSINE_BLOCK):
        end = min(begin + COSINE_BLOCK, ROWS)
        anchor = torch.from_numpy(
            np.ascontiguousarray(host[begin:end])
        ).to(device)
        flat = ids[begin:end].reshape(-1).astype(np.int64)
        order = np.argsort(flat, kind="stable")
        gathered = np.empty((flat.size, DIMENSION), dtype=np.float32)
        gathered[order] = host[flat[order]]
        neighbours = torch.from_numpy(gathered).to(device).view(
            end - begin, GRAPH_K, DIMENSION
        )
        out[begin:end] = (
            torch.einsum("bd,bkd->bk", anchor, neighbours).cpu().numpy()
        )
        del anchor, neighbours, gathered, flat, order
    return out


def _score_probe(
    *,
    ids: np.ndarray,
    candidate_cos: np.ndarray,
    probe_rows: np.ndarray,
    truth_ids: np.ndarray,
    kth: np.ndarray,
    truth_best: np.ndarray,
) -> dict[str, Any]:
    """Strict and tie-aware containment on the registered uniform probe.

    `truth_best` is each probe row's exact BEST cosine and exists only for the
    zero-recall forensic: review-0235-01 explained tie-aware `0.0` as a duplicate
    family artefact and R0236 assumed the explanation carried. This verifies it.
    """
    probe_ids = np.ascontiguousarray(ids[probe_rows])
    probe_cos = np.ascontiguousarray(candidate_cos[probe_rows])
    strict = strict_containment_rows(probe_ids, truth_ids)
    tie = tie_aware_rows(probe_cos.astype(np.float64), probe_ids, kth)
    order = np.argsort(kth, kind="stable")
    size = probe_rows.size
    decile_tie = [
        float(tie[order[index * size // DENSITY_DECILES:
                       (index + 1) * size // DENSITY_DECILES]].mean())
        for index in range(DENSITY_DECILES)
    ]
    decile_strict = [
        float(strict[order[index * size // DENSITY_DECILES:
                           (index + 1) * size // DENSITY_DECILES]].mean())
        for index in range(DENSITY_DECILES)
    ]
    lost = np.rint((1.0 - strict) * GRAPH_K).astype(np.int16)
    summary = {
        "recall_population": RECALL_POPULATION,
        "rows_measured": int(size),
        "strict": summarize(strict, label="R0237 strict recall@15"),
        "tie_aware": summarize(tie, label="R0237 tie-aware recall@15"),
        "density_decile_tie_aware": decile_tie,
        "density_decile_strict": decile_strict,
        "density_decile_definition": (
            "deciles of the probe row's own exact 15th-best cosine; decile 0 "
            "sparsest"
        ),
        "rows_carrying_any_loss": int((lost > 0).sum()),
        "fraction_carrying_any_loss": float((lost > 0).mean()),
        "missing_true_edges": int(lost.sum()),
        "tie_aware_rows_below_one": int((tie < 1.0).sum()),
        "tie_aware_rows_at_zero": int((tie <= 0.0).sum()),
    }
    zero = np.flatnonzero(tie <= 0.0)
    summary["zero_recall_forensic"] = zero_recall_forensic(
        zero_rows=np.asarray(probe_rows)[zero],
        truth_kth_cosine=np.asarray(kth)[zero],
        truth_best_cosine=np.asarray(truth_best)[zero],
        candidate_best_cosine=(
            probe_cos[zero].max(axis=1) if zero.size else np.empty(0)
        ),
        candidate_worst_cosine=(
            probe_cos[zero].min(axis=1) if zero.size else np.empty(0)
        ),
    )
    del strict, tie, order, probe_ids, probe_cos
    return summary


def run_qualify(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    import torch

    manifest = active["manifest"]
    substrate_path, substrate_sealed = _substrate_from_manifest(job)
    truth_manifest_path, _ = _intra(job, "truth_reference", label="R0237 truth")
    truth = prompt_contract.read_sealed(truth_manifest_path, label="R0237 truth")
    truth_rows_path, _ = _intra_signature(
        dict(truth["outputs"]["query_rows"]), label="R0237 probe rows"
    )
    truth_ids_path, _ = _intra_signature(
        dict(truth["outputs"]["ids"]), label="R0237 truth ids"
    )
    truth_cos_path, _ = _intra_signature(
        dict(truth["outputs"]["cosines"]), label="R0237 truth cosines"
    )
    ladder_path, _ = _intra(job, "ladder_reference", label="R0237 build ladder")
    ladder = prompt_contract.read_sealed(ladder_path, label="R0237 build ladder")

    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0237 qualification"
    )
    started = time.monotonic()
    # At 50,000,000 rows the fuzzy symmetrisation is the largest host-memory
    # consumer in the queue and it has no watchdog. Sample it: peak anonymous
    # bytes and minimum MemAvailable, reported, never used to abort.
    node_sampler = _HostSampler()
    node_sampler.start()

    probe_rows = np.load(truth_rows_path, allow_pickle=False)
    truth_ids = np.load(truth_ids_path, allow_pickle=False)
    truth_cos = np.load(truth_cos_path, allow_pickle=False)
    if truth_ids.shape != (TRUTH_PROBE_ROWS, GRAPH_K):
        raise Round0237Error("R0237 truth arrays have the wrong shape")
    if not np.array_equal(
        probe_rows,
        truth_probe_query_rows(
            rows=ROWS, size=TRUTH_PROBE_ROWS, seed=TRUTH_PROBE_SEED
        ),
    ):
        raise Round0237Error(
            "R0237 sealed probe rows are not the registered draw at seed "
            f"{TRUTH_PROBE_SEED}"
        )
    kth = truth_cos[:, GRAPH_K - 1].astype(np.float64)
    truth_best = truth_cos[:, 0].astype(np.float64)
    del truth_cos

    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    host = _host_substrate(substrate_path)

    selected_clusters = int(ladder["cluster_selection"]["selected_clusters"])
    entry = next(
        item for item in ladder["ladder"]
        if item.get("run") and int(item["clusters"]) == selected_clusters
    )
    if not entry.get("graph_ids"):
        raise Round0237Error(
            f"R0237 selected c = {selected_clusters} emitted no graph"
        )
    ids_path = prompt_contract.verify_signature(
        dict(entry["graph_ids"]), label=f"R0237 {entry['cell']} graph"
    )
    ids = np.ascontiguousarray(
        np.load(ids_path, allow_pickle=False).astype(np.int32)
    )
    if ids.shape != (ROWS, GRAPH_K):
        raise Round0237Error(f"R0237 {entry['cell']} graph is {ids.shape}")

    candidate_cos = _candidate_cosines(
        ids=ids, host=host, torch=torch, device=device
    )
    selected = _score_probe(
        ids=ids, candidate_cos=candidate_cos, probe_rows=probe_rows,
        truth_ids=truth_ids, kth=kth, truth_best=truth_best,
    )
    selected.update({
        "clusters": selected_clusters,
        "spill": SPILL,
        "cell": entry["cell"],
    })

    # The builder's own cosines are never used for a recall claim, but comparing
    # them against the independent recompute is free and says whether the two
    # arithmetics agree at this rung.
    builder_agreement: dict[str, Any] | None = None
    if entry.get("graph_cos"):
        builder_cos_path = prompt_contract.verify_signature(
            dict(entry["graph_cos"]), label=f"R0237 {entry['cell']} builder cosines"
        )
        builder_cos = np.load(builder_cos_path, mmap_mode="r", allow_pickle=False)
        worst = 0.0
        for begin in range(0, ROWS, 2_000_000):
            end = min(begin + 2_000_000, ROWS)
            worst = max(worst, float(np.abs(
                np.asarray(builder_cos[begin:end], dtype=np.float64)
                - candidate_cos[begin:end].astype(np.float64)
            ).max()))
        builder_agreement = {
            "max_absolute_difference": worst,
            "note": (
                "the independent recompute against the builder's emitted "
                "cosines; the recompute is what every recall number uses"
            ),
        }
        del builder_cos

    structural = graph_validity(ids, rows=ROWS)
    if int(structural["zero_degree_rows"]) > MAX_ZERO_DEGREE_ROWS:
        raise Round0237Error(
            f"R0237 R0215 tripwire on the raw graph: "
            f"{structural['zero_degree_rows']} zero-degree rows"
        )
    selected["structural"] = structural
    selected["structural_population"] = STRUCTURAL_POPULATION
    selected["zero_degree_rows"] = int(structural["zero_degree_rows"])

    if float(selected["tie_aware"]["mean"]) < RECALL_MEAN_FLOOR:
        raise Round0237Error(
            f"R0237 tie-aware recall {selected['tie_aware']['mean']} is below "
            f"the registered {RECALL_MEAN_FLOOR} floor"
        )
    if float(selected["tie_aware"]["p10"]) < RECALL_P10_FLOOR:
        raise Round0237Error(
            f"R0237 tie-aware p10 {selected['tie_aware']['p10']} is below the "
            f"registered {RECALL_P10_FLOOR} floor"
        )
    del truth_ids
    torch.cuda.empty_cache()
    gc.collect()

    # ---- R0216's fuzzy law, unchanged, on the selected graph ----
    sort_order = np.argsort(-candidate_cos, axis=1, kind="stable")
    ids_sorted = np.take_along_axis(ids, sort_order, axis=1).astype(np.int32)
    cos_sorted = np.take_along_axis(candidate_cos, sort_order, axis=1)
    del sort_order, ids, candidate_cos
    gc.collect()
    dists = np.maximum((1.0 - cos_sorted).astype(np.float32), 0.0)
    if not np.isfinite(dists).all():
        raise Round0237Error("R0237 candidate distances are not finite")
    del cos_sorted
    gc.collect()

    import umap.umap_ as umap_api

    fuzzy_started = time.monotonic()
    graph, _sigmas, _rhos = umap_api.fuzzy_simplicial_set(
        host, n_neighbors=GRAPH_K,
        random_state=np.random.RandomState(FUZZY_RANDOM_STATE_SEED),
        metric="cosine", knn_indices=ids_sorted, knn_dists=dists,
    )
    coo = graph.tocoo()
    src = np.asarray(coo.row, dtype=np.int32)
    dst = np.asarray(coo.col, dtype=np.int32)
    wts = np.asarray(coo.data, dtype=np.float32)
    fuzzy_s = time.monotonic() - fuzzy_started
    del graph, coo, dists
    gc.collect()

    if not np.isfinite(wts).all() or wts.min() <= 0 or wts.max() > 1:
        raise Round0237Error("R0237 fuzzy weights are invalid")
    out_degree = np.bincount(src, minlength=ROWS)
    in_degree = np.bincount(dst, minlength=ROWS)
    degrees = {
        "population": STRUCTURAL_POPULATION,
        "out_degree_zero_rows": int((out_degree == 0).sum()),
        "in_degree_zero_rows": int((in_degree == 0).sum()),
        "zero_degree_rows": int(
            max((out_degree == 0).sum(), (in_degree == 0).sum())
        ),
        "out_degree_min": int(out_degree.min()),
        "in_degree_min": int(in_degree.min()),
        "median": float(np.median(out_degree)),
        "mean": float(out_degree.mean()),
        "max": int(out_degree.max()),
    }
    if degrees["zero_degree_rows"] > MAX_ZERO_DEGREE_ROWS:
        raise Round0237Error(
            f"R0237 R0215 tripwire: {degrees['zero_degree_rows']} zero-degree rows"
        )
    ids_out = atomic_save_new_npy(
        os.path.join(output, "graph-k15-ids.i32.npy"), ids_sorted, immutable=True
    )
    edges_path = atomic_save_new_npz(
        os.path.join(output, "edges-k15-fuzzy.npz"), immutable=True,
        compressed=False, sources=src, targets=dst, weights=wts,
        n_nodes=np.asarray(ROWS, dtype=np.int64),
        k=np.asarray(GRAPH_K, dtype=np.int64),
    )
    directed_edges = int(len(src))
    weight_min = float(wts.min())
    weight_max = float(wts.max())
    del src, dst, wts, ids_sorted, out_degree, in_degree
    gc.collect()

    # ---- the law, refitted with this round's own cell added ----
    inherited_law = ladder["inherited_device_law"]
    own_points: list[dict[str, Any]] = []
    instruments: list[dict[str, Any]] = []
    io_points: list[dict[str, Any]] = _inherited_io_points(job)
    setting = ladder.get("nn_descent") or {}
    for cell in ladder["ladder"]:
        build = cell.get("build") or {}
        largest = int((build.get("cluster_sizes") or {}).get("max") or 0)
        measured_bytes = build.get("device_wide_peak_over_baseline_bytes")
        passes = build.get("substrate_passes")
        instruments.append({
            "cell": cell["cell"],
            "clusters": int(cell["clusters"]),
            "role": cell.get("role"),
            "run": bool(cell.get("run")),
            "fit": build.get("fit"),
            "refused_a_priori": build.get("refused_a_priori"),
            "refusal_reasons": build.get("refusal_reasons") or [],
            "max_cluster_rows": largest or None,
            "imbalance_max_over_mean": (
                build.get("cluster_sizes") or {}
            ).get("imbalance_max_over_mean"),
            "device_wide_peak_bytes": build.get("device_wide_peak_bytes"),
            "device_wide_peak_over_baseline_bytes": measured_bytes,
            "nvidia_smi_per_process_peak_bytes": build.get(
                "nvidia_smi_per_process_peak_bytes"
            ),
            "child_device_peak_sampled_bytes": build.get(
                "child_device_peak_sampled_bytes"
            ),
            "rmm_peak_bytes": build.get("rmm_peak_bytes"),
            "rmm_bytes_per_max_cluster_row": (
                float(build["rmm_peak_bytes"]) / largest
                if largest and build.get("rmm_peak_bytes") else None
            ),
            "host_rss_peak_bytes": build.get("host_rss_peak_bytes"),
            "host_anon_peak_bytes_parent_watchdog": build.get(
                "host_anon_peak_bytes_parent_watchdog"
            ),
            "host_anon_peak_bytes_child_phase_sampled": build.get(
                "host_anon_peak_bytes_child_phase_sampled"
            ),
            "host_anon_authoritative_sampler": build.get(
                "host_anon_authoritative_sampler"
            ),
            "host_vmhwm_bytes": build.get("host_vmhwm_bytes"),
            "system_swap_growth_bytes": build.get("system_swap_growth_bytes"),
            "peak_scratch_bytes": build.get("peak_scratch_bytes"),
            "spill_groups": build.get("spill_groups"),
            "substrate_passes": passes,
            "substrate_read_bytes": build.get("substrate_read_bytes"),
            "spill_write_bytes": build.get("spill_write_bytes"),
            "child_io_read_bytes": build.get("child_io_read_bytes"),
            "child_io_write_bytes": build.get("child_io_write_bytes"),
            "child_io_rchar_bytes": build.get("child_io_rchar_bytes"),
            "child_io_wchar_bytes": build.get("child_io_wchar_bytes"),
            "data_block_device_read_bytes": build.get(
                "data_block_device_read_bytes"
            ),
            "data_block_device_write_bytes": build.get(
                "data_block_device_write_bytes"
            ),
            "mem_available_bytes_before": build.get("mem_available_bytes_before"),
            "builder_seconds": build.get("builder_seconds"),
            "watchdog_escalations": build.get("watchdog_escalations"),
            "signal_handler_installed": build.get("signal_handler_installed"),
            "cuvs_inputs_asserted_memmap": build.get("cuvs_inputs_asserted_memmap"),
            "cluster_capacity_rows": build.get("cluster_capacity_rows"),
            "phases": build.get("phases"),
        })
        if build.get("fit") and passes:
            io_points.append({
                "source": f"R0237 {cell['cell']}",
                "rows": ROWS,
                "clusters": int(cell["clusters"]),
                "spill": SPILL,
                "substrate_passes": int(passes),
                "spill_groups": build.get("spill_groups"),
                "substrate_read_bytes": int(build["substrate_read_bytes"]),
                "spill_write_bytes": build.get("spill_write_bytes"),
                "spill_write_seconds": (build.get("phases") or {}).get(
                    "spill_write_seconds"
                ),
                "builder_seconds": build.get("builder_seconds"),
                "measured_physical_io": True,
                "child_io_read_bytes": build.get("child_io_read_bytes"),
                "child_io_write_bytes": build.get("child_io_write_bytes"),
                "data_block_device_read_bytes": build.get(
                    "data_block_device_read_bytes"
                ),
                "data_block_device_write_bytes": build.get(
                    "data_block_device_write_bytes"
                ),
            })
        if not build.get("fit") or largest <= 0 or not measured_bytes:
            continue
        own_points.append({
            "source": f"R0237 ladder {cell['cell']}",
            "cell": cell["cell"],
            "rows": ROWS,
            "clusters": int(cell["clusters"]),
            "spill": SPILL,
            "graph_degree": setting.get("graph_degree"),
            "intermediate_graph_degree": setting.get("intermediate_graph_degree"),
            "max_iterations": setting.get("max_iterations"),
            "max_cluster_rows": largest,
            "device_bytes": float(measured_bytes),
        })

    combined_law = fit_device_law(
        list(inherited_law["points"]) + own_points,
        label="all-sealed-gd64-igd256-it40-with-r0236",
    )
    laws = [combined_law]

    # ---- the imbalance table: this round's five seeds at 50M, merged with
    # ---- R0236's sealed three seeds at 6.25M / 12.5M / 25M ----
    inherited_imbalance = _inherited_imbalance(job)
    inherited_grid, inherited_grid_source = _inherited_replicate_grid(job)
    grid: dict[int, dict[int, dict[int, float]]] = {
        int(rows): {
            int(clusters): dict(seeds) for clusters, seeds in cells.items()
        }
        for rows, cells in inherited_grid.items()
    }
    own_rows: set[int] = set()
    for cell in ladder["measured_imbalance"]["cells"]:
        own_rows.add(int(cell["rows"]))
        grid.setdefault(int(cell["rows"]), {}).setdefault(
            int(cell["clusters"]), {}
        )[int(cell["seed"])] = float(cell["imbalance_max_over_mean"])
    sources_by_rows = {
        int(rows): (
            f"measured in R0237, {len(IMBALANCE_REPLICATE_SEEDS)} seeds"
            if int(rows) in own_rows
            else "measured in R0236, 3 seeds, sealed and hash-bound"
        )
        for rows in grid
    }
    drift = replicate_grid_table(
        grid, sources_by_rows=sources_by_rows,
        inherited=inherited_imbalance["by_rows"],
    )
    drift["merged_from"] = inherited_grid_source
    drift["drift_note"] = (
        "R0236 settled the direction of travel and review-0236-01 sharpened it: "
        "analysed at the (N, seed) level the movement is a statistically "
        "significant -1.749% per doubling (t = -2.686, df = 5, p = 0.0435; "
        "permutation p = 0.0463), about 8x smaller than R0235's single-"
        "realisation -13.42% and pointing in the SAFE direction. This round does "
        "not re-litigate it: the table is descriptive and the round's weight is "
        "on the within-N spread, which is what a guard margin is exposed to."
    )

    at_this_rung = grid.get(ROWS) or {}
    primary_now = {
        clusters: float(seeds[PRIMARY_IMBALANCE_SEED])
        for clusters, seeds in at_this_rung.items()
        if PRIMARY_IMBALANCE_SEED in seeds
    }
    worst_now = {
        clusters: float(max(seeds.values()))
        for clusters, seeds in at_this_rung.items()
    }

    # ---- per-rung re-derivation, with the tolerance stated ----
    rungs: dict[str, Any] = {}
    for rung in PHASE2_RUNGS:
        with_margin = rung_derivation(
            rung=int(rung), imbalance_by_c=worst_now,
            imbalance_source=(
                f"worst of {len(IMBALANCE_REPLICATE_SEEDS)} replicate seeds at "
                f"N = {ROWS:,}, s = 8, on this round's substrate"
            ),
            laws=laws, apply_margin=True,
        )
        tolerances = {
            str(int(clusters)): imbalance_tolerance(
                rung=int(rung), clusters=int(clusters),
                imbalance=float(value), laws=laws,
            )
            for clusters, value in sorted(worst_now.items())
        }
        rungs[str(rung)] = {
            "with_margin": with_margin,
            "point_estimate_no_margin": rung_derivation(
                rung=int(rung), imbalance_by_c=worst_now,
                imbalance_source=(
                    f"worst of {len(IMBALANCE_REPLICATE_SEEDS)} replicate seeds "
                    f"at N = {ROWS:,}, s = 8"
                ),
                laws=laws, apply_margin=False,
            ),
            "with_margin_primary_seed": rung_derivation(
                rung=int(rung), imbalance_by_c=primary_now,
                imbalance_source=(
                    f"primary seed {PRIMARY_IMBALANCE_SEED} at N = {ROWS:,}, s = 8"
                ),
                laws=laws, apply_margin=True,
            ),
            "imbalance_tolerance_by_c": tolerances,
            "carry": carry_distance(measured_at_rows=ROWS, rung=int(rung)),
            "selected_tolerance": (
                tolerances.get(str(int(with_margin["selected_clusters"])))
                if with_margin.get("selected_clusters") is not None else None
            ),
        }

    # ---- the 100M verdict: confirm c = 400, do not re-open the question ----
    reachability_path, reachability = _read_bound(
        job, "reachability_reference", label="R0237 high-c reachability",
        sealed=True,
    )
    if str(reachability.get("round_id")) != ROUND_ID:
        raise Round0237Error("R0237 reachability artifact is not this round's")
    reachability_by_c = {
        int(key): float(value)
        for key, value in (
            reachability.get("strict_ceiling_by_clusters") or {}
        ).items()
    }
    hundred_m = hundred_m_verdict(
        imbalance_by_c=worst_now, laws=laws,
        reachability_by_c=reachability_by_c,
    )
    hundred_m["imbalance_source"] = (
        f"worst of {len(IMBALANCE_REPLICATE_SEEDS)} replicate seeds measured at "
        f"N = {ROWS:,} on this round's substrate"
    )
    hundred_m["reachability_source"] = expected_input_signature(reachability_path)
    hundred_m["reachability_measured_at_rows"] = int(reachability["rows"])
    hundred_m["carry"] = carry_distance(measured_at_rows=ROWS, rung=100_000_000)
    hundred_m["code_corpus_headroom"] = {
        "target_rows_at_100m": 10_000_000,
        "corpus_rows_available": (
            (substrate_sealed.get("sources") or {})
            .get("starcoderdata-code-chunked-120-all-MiniLM-L6-v2", {})
            .get("corpus_rows")
        ),
        "reserve_rows_withheld": RESERVE_ROWS_PER_CORPUS,
        "note": (
            "an independent 100M constraint this round can see and no earlier "
            "rung could: the code corpus is exactly 10,000,000 rows and the "
            "plan's 100M composition asks for 10,000,000 of them, while "
            "50,000 are permanently withheld as the eval reserve. The final "
            "rung cannot be assembled at the confirmed shares without either "
            "more code chunks or a share change. This is a composition finding, "
            "not a device finding, and it is reported rather than acted on."
        ),
    }

    # ---- the I/O scaling, measured and projected ----
    selected_build = next(
        cell["build"] for cell in ladder["ladder"]
        if cell.get("run") and int(cell["clusters"]) == selected_clusters
    )
    realised_passes = int(selected_build.get("substrate_passes") or 1)
    spill_write_seconds = float(
        (selected_build.get("phases") or {}).get("spill_write_seconds") or 0.0
    )
    architecture = architectural_io(rows=ROWS, substrate_passes=realised_passes)
    measured_read_rate = (
        substrate_sealed.get("substrate_read_measurement") or {}
    ).get("bytes_per_s")
    spill_phase_rate = (
        (architecture["substrate_read_bytes"] + architecture["spill_write_bytes"])
        / spill_write_seconds if spill_write_seconds > 0 else None
    )
    scaling = io_scaling_fit(io_points)
    scaling["scope_correction"] = (
        "review-0236-01 F5: substrate_read_bytes is IDENTICALLY "
        "substrate_passes x N x 1536 in every fitted point, and only this "
        "round's and R0236's carry a physical measurement at all. The exponent "
        "below describes the ceiling function on whole-cluster spill packing, "
        "NOT a measured physical scaling, and this round does not cite it as "
        "one. The finding this round reports is the MEASURED one: child "
        "read_bytes and /data device reads against the architectural volume, "
        "with mincore page-cache residency beside them."
    )
    # ---- the regime, measured rather than inferred from a budget ----
    measured_reads = selected_build.get("child_io_read_bytes")
    device_reads = selected_build.get("data_block_device_read_bytes")
    architectural_reads = int(architecture["substrate_read_bytes"])
    cache_before = selected_build.get("substrate_page_cache_before") or {}
    cache_after = selected_build.get("substrate_page_cache_after") or {}
    substrate_bytes = int(architecture["substrate_bytes"])
    observed_regime = {
        "rows": ROWS,
        "clusters": selected_clusters,
        "substrate_bytes": substrate_bytes,
        "substrate_passes_measured": realised_passes,
        "architectural_substrate_read_bytes": architectural_reads,
        "child_io_read_bytes": measured_reads,
        "data_block_device_read_bytes": device_reads,
        "child_io_write_bytes": selected_build.get("child_io_write_bytes"),
        "data_block_device_write_bytes": selected_build.get(
            "data_block_device_write_bytes"
        ),
        "architectural_spill_write_bytes": int(architecture["spill_write_bytes"]),
        "child_read_fraction_of_architecture": (
            float(measured_reads) / architectural_reads
            if measured_reads is not None and architectural_reads else None
        ),
        "device_read_fraction_of_architecture": (
            float(device_reads) / architectural_reads
            if device_reads is not None and architectural_reads else None
        ),
        "child_read_multiple_of_one_pass": (
            float(measured_reads) / substrate_bytes
            if measured_reads is not None and substrate_bytes else None
        ),
        "substrate_page_cache_before": cache_before,
        "substrate_page_cache_after": cache_after,
        "page_cache_budget_bytes_registered": PAGE_CACHE_BUDGET_BYTES,
        "mem_available_bytes_before_cell": selected_build.get(
            "mem_available_bytes_before"
        ),
        "mem_available_min_bytes_sampled": selected_build.get(
            "mem_available_min_bytes_sampled"
        ),
        "regime_observed": (
            None if measured_reads is None or not architectural_reads else (
                "page-cache-resident"
                if float(measured_reads) < 0.5 * architectural_reads
                else "page-cache-thrashing"
            )
        ),
        "regime_predicted": (
            "page-cache-resident"
            if substrate_bytes <= PAGE_CACHE_BUDGET_BYTES
            else "page-cache-thrashing"
        ),
        "note": (
            "the regime is stated from the measured block-layer read volume and "
            "the mincore residency of the substrate file, not from the "
            "registered page-cache budget. review-0236-01 F5: the ~52M-row flip "
            "point is exactly page_cache_budget / 1536, an assumption no rung "
            "before this one had power to test. This substrate is 76.8 GB "
            "against that 80 GB budget, a 4% margin, so this is the test."
        ),
    }
    projections: dict[str, Any] = {}
    for rung in PHASE2_RUNGS:
        derivation = rungs[str(rung)]["with_margin"]
        clusters = derivation.get("selected_clusters")
        if clusters is None:
            projections[str(rung)] = {
                "rung": int(rung), "feasible": False,
                "note": "no admissible c, so no I/O term is defined",
            }
            continue
        imbalance = float(worst_now[int(clusters)])
        passes = (
            realised_passes if int(rung) == ROWS
            else predicted_substrate_passes(
                rows=int(rung), clusters=int(clusters), imbalance=imbalance
            )
        )
        prediction = physical_io_prediction(rows=int(rung), substrate_passes=passes)
        rates = {
            "measured_this_rung_cold_read": measured_read_rate,
            "measured_this_rung_spill_phase": spill_phase_rate,
            "review_0232_fragmented": DATA_READ_FRAGMENTED_BYTES_PER_S,
            "review_0232_contiguous": DATA_READ_CONTIGUOUS_BYTES_PER_S,
        }
        projections[str(rung)] = {
            "rung": int(rung),
            "clusters": int(clusters),
            "imbalance_carried": imbalance,
            "substrate_passes": int(passes),
            "substrate_passes_measured": bool(int(rung) == ROWS),
            "architectural": architectural_io(
                rows=int(rung), substrate_passes=int(passes)
            ),
            "physical_prediction": prediction,
            "io_hours_architectural": {
                name: io_hours(
                    read_bytes=prediction["total_read_bytes"],
                    write_bytes=prediction["total_write_bytes"],
                    read_bytes_per_s=float(rate),
                    write_bytes_per_s=DATA_WRITE_BYTES_PER_S,
                )
                for name, rate in rates.items() if rate
            },
            "io_hours_physical_prediction": {
                name: io_hours(
                    read_bytes=prediction["predicted_physical_read_bytes"],
                    write_bytes=prediction["predicted_physical_write_bytes"],
                    read_bytes_per_s=float(rate),
                    write_bytes_per_s=DATA_WRITE_BYTES_PER_S,
                )
                for name, rate in rates.items() if rate
            },
        }

    io_term = io_projection(rows=ROWS, substrate_passes=realised_passes)
    io_contiguous = io_projection(
        rows=ROWS, substrate_passes=realised_passes,
        read_bytes_per_s=DATA_READ_CONTIGUOUS_BYTES_PER_S,
    )

    node_sampler.halt()
    node_host = node_sampler.readings()
    receipt = prompt_contract.seal(json_safe({
        "schema": GRAPH_SCHEMA,
        "law_schema": LAW_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": manifest["release_sha"],
        "capability": GRAPH_CAPABILITY,
        "capabilities": [
            GRAPH_CAPABILITY, IMBALANCE_CAPABILITY, IO_CAPABILITY,
            FEASIBILITY_CAPABILITY,
        ],
        "rows": ROWS,
        "k": GRAPH_K,
        "spill": SPILL,
        "selected_clusters": selected_clusters,
        "selected_cell": entry["cell"],
        "cluster_selection": ladder["cluster_selection"],
        "recall_population": RECALL_POPULATION,
        "structural_population": STRUCTURAL_POPULATION,
        "truth_method": TRUTH_METHOD,
        "truth_affordability": TRUTH_AFFORDABILITY_NOTE,
        "probe_rows": TRUTH_PROBE_ROWS,
        "probe_seed": TRUTH_PROBE_SEED,
        "selected_graph": selected,
        "builder_cosine_agreement": builder_agreement,
        "floors": {
            "tie_aware_mean": RECALL_MEAN_FLOOR,
            "tie_aware_p10": RECALL_P10_FLOOR,
            "zero_degree_rows": MAX_ZERO_DEGREE_ROWS,
        },
        "degrees": degrees,
        "directed_edges": directed_edges,
        "edges_per_row": directed_edges / float(ROWS),
        "fuzzy_weight_range": [weight_min, weight_max],
        "graph": expected_input_signature(edges_path),
        "neighbour_ids": expected_input_signature(ids_out),
        "nesting": substrate_sealed.get("nesting"),
        "device_law_inherited": inherited_law,
        "device_law_combined": combined_law,
        "device_law_homogeneity_note": LAW_HOMOGENEITY_NOTE,
        "device_law_setting": {
            "graph_degree": LAW_GRAPH_DEGREE,
            "intermediate_graph_degree": LAW_INTERMEDIATE_GRAPH_DEGREE,
            "max_iterations": LAW_MAX_ITERATIONS,
        },
        "guard": {
            "note": GUARD_NOTE,
            "imbalance_margin": GUARD_IMBALANCE_MARGIN,
            "law_residual_margin": LAW_RESIDUAL_MARGIN,
            "device_budget_bytes": int(GUARD_DEVICE_BUDGET_BYTES),
            "admissible_max_cluster_rows": float(admissible_max_cluster_rows(laws)),
            "cluster_capacity_rows_used": ladder.get("cluster_capacity_rows"),
        },
        "imbalance_table": drift,
        "imbalance_drift": drift,
        "imbalance_sources": inherited_imbalance.get("sources"),
        "imbalance_replicate_seeds": [int(v) for v in IMBALANCE_REPLICATE_SEEDS],
        "primary_seed_imbalance_at_this_rung": primary_now,
        "worst_seed_imbalance_at_this_rung": worst_now,
        "per_rung_derivation": rungs,
        "hundred_m_verdict": hundred_m,
        "reachability_high_c": {
            "source": expected_input_signature(reachability_path),
            "rows": int(reachability["rows"]),
            "strict_ceiling_by_clusters": reachability_by_c,
            "concern_floor": REACHABILITY_CONCERN_FLOOR,
            "control": reachability.get("control"),
            "r0229_2m_s8_reference": reachability.get("r0229_2m_s8_reference"),
            "note": REACHABILITY_NOTE,
        },
        "imbalance_margin_note": MARGIN_NOTE,
        "io_observed_regime": observed_regime,
        "io_scaling": scaling,
        "io_regime_note": IO_REGIME_NOTE,
        "io_instrument_note": IO_INSTRUMENT_NOTE,
        "io_projection_by_rung": projections,
        "io_measured_rates": {
            "substrate_cold_read_bytes_per_s": measured_read_rate,
            "spill_phase_combined_bytes_per_s": spill_phase_rate,
            "spill_write_seconds": spill_write_seconds,
            "note": (
                "the spill phase reads the substrate once per group and writes "
                "the whole spill volume, so its combined rate is the honest "
                "throughput at the file layouts this round wrote"
            ),
        },
        "io_term_fragmented": io_term,
        "io_term_contiguous": io_contiguous,
        "substrate_read_measurement": substrate_sealed.get(
            "substrate_read_measurement"
        ),
        "substrate_write_measurement": substrate_sealed.get(
            "substrate_write_measurement"
        ),
        "per_cell_instruments": instruments,
        "host_anon_field_note": HOST_ANON_FIELD_NOTE,
        "host_anon_authoritative_sampler": HOST_ANON_AUTHORITATIVE_SAMPLER,
        "determinism_note": DETERMINISM_NOTE,
        "performance": {
            "fuzzy_s": fuzzy_s,
            "total_wall_s": time.monotonic() - started,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            **node_host,
        },
        "training_performed": False,
        "gate_registered": False,
        "adoption_claimed": False,
        "hundred_m_built": False,
        "hundred_m_registered_here": False,
    }))
    atomic_write_new_json(
        os.path.join(output, "qualified-graph.json"), receipt, immutable=True
    )


# --------------------------------------------------------------------------- #
# intra-queue references
# --------------------------------------------------------------------------- #
def _intra_signature(
    signature: Mapping[str, Any], *, label: str
) -> tuple[str, dict[str, Any]]:
    path = str(signature.get("canonical_path") or "")
    if not path or not os.path.exists(path):
        raise Round0237Error(f"{label} is absent at {path!r}")
    observed = expected_input_signature(path)
    if signature.get("sha256") and observed.get("sha256") != signature.get("sha256"):
        raise Round0237Error(f"{label} bytes changed")
    return path, observed


def _intra(
    job: Mapping[str, Any], key: str, *, label: str
) -> tuple[str, dict[str, Any]]:
    reference = dict(job[key])
    return _intra_signature(reference, label=label)


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    action = str(job.get("action") or "")
    if action == REACHABILITY_ACTION:
        run_reachability(active, job)
    elif action == ASSEMBLE_ACTION:
        run_assemble(active, job)
    elif action == TRUTH_ACTION:
        run_truth(active, job)
    elif action == LADDER_ACTION:
        run_ladder(active, job)
    elif action == QUALIFY_ACTION:
        run_qualify(active, job)
    else:
        raise Round0237Error(f"R0237 does not authorize action {action!r}")


__all__ = [
    "ASSEMBLE_ACTION",
    "LADDER_ACTION",
    "QUALIFY_ACTION",
    "REACHABILITY_ACTION",
    "TRUTH_ACTION",
    "run_job",
]
