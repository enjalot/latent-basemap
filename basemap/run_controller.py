"""Single fail-closed GPU run-controller with an atomic, crash-safe lease (P0-D).

Guarantees one GPU job at a time and, critically, keeps the lease held for the
LIFETIME OF THE GPU CHILD even if the controller process is killed:

- the lease is an exclusive `flock` on an open file description; the controller
  passes that FD (inheritable) into the child and starts the child in its own
  process group, so the OS lock persists until the child exits even if the
  controller dies uncatchably. A new controller then cannot overlap the orphan.
- jobs are idempotent via an explicit controller-written `.done.json` completion
  record (NOT the output file), created only after exit-0 AND declared-output
  validation. A stale/partial output never counts as done.
- the chain STOPS on nonzero exit, missing outputs, or an invalid completion
  record, unless a job sets `continue_on_failure`.
- before a launch the controller enforces a co-tenant policy: unknown compute
  PIDs or insufficient free VRAM fail (or wait), rather than launching anyway.

Route ALL GPU entry points (training, scoring, graph/index validation, benches,
chains) through here; direct launches are unsafe.
"""
from __future__ import annotations
import os, sys, json, time, uuid, fcntl, subprocess, datetime, dataclasses, hashlib
import importlib
import signal
import secrets
import stat
import struct
import select
import socket
import ctypes
import re
import tempfile
from typing import Optional

from .artifact_identity import (canonical_json, expected_input_signature,
                                git_checkout_state, path_signature, sha256_bytes,
                                sha256_file)
from .output_safety import (atomic_copy_new, atomic_write_new_json,
                            refuse_existing)

_DEFAULT_LEASE = "/data/latent-basemap/.gpu_lease"
_CHILD_CAPABILITY_FD_ENV = "BASEMAP_ROUND0005_CAPABILITY_FD"
_CHILD_CAPABILITY_NONCE_ENV = "BASEMAP_ROUND0005_LAUNCH_NONCE"
_ACTIVE_CHILD_ADMISSION: dict | None = None
_GENUINE_CHILD_ADMISSION_SEALS: dict[int, str] = {}
_NVIDIA_SMI_EXECUTABLE = "/usr/bin/nvidia-smi"
_NVIDIA_SMI_ENVIRONMENT = {"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"}
_ROUND0005_GSV_GPU_NAME = "NVIDIA GeForce RTX 5090"
_CHILD_CONTRACT_FIELDS = {
    "schema", "launch_nonce", "controller_id", "controller_pid",
    "controller_starttime_ticks", "parent_pid", "child_pid", "node_id",
    "job_index", "deps", "argv", "environment", "manifest_sha256",
    "gate_identity", "lease_identity", "deadline_utc", "gpu_hours_cap",
    "gpu_memory_cap_mb", "p90_wall_s", "telemetry_interval_s",
    "watchdog_channel", "delegated", "script", "descendant_limit",
    "descendant_argv", "delegated_ordinal", "cache_policy", "controller_claim",
}


def _lease_path() -> str:
    """Resolve the lease path at CALL time so BASEMAP_GPU_LEASE set after import
    (e.g. in tests, or per-launch) is honoured."""
    return os.environ.get("BASEMAP_GPU_LEASE", _DEFAULT_LEASE)


LEASE_PATH = _lease_path()   # back-compat module constant (import-time value)


def _utcnow() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _atomic_write_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)   # atomic


def _output_sig(path: str) -> Optional[dict]:
    """Readable full SHA-256 signature of an output, or None if absent."""
    try:
        return path_signature(path)
    except FileNotFoundError:
        return None


def _git_state():
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return git_checkout_state(root)
    except Exception:
        return {"head": None, "clean": None, "detached": None,
                "dirty_tree_digest": None, "error": "nogit"}


def _job_spec_digest(job) -> str:
    """Bind a done record to the job's FULL identity (P1/L0.3): argv + cwd +
    declared outputs + the repo commit/dirty state + the content hashes of every
    declared input_path (config, scorer/trainer code, input artifacts) + the
    dependency edges, canary identity, predicted wall, and resource/verdict policy.
    A changed scorer, config, dependency edge, or resource policy at the same argv
    therefore invalidates a stale done marker."""
    payload = json.dumps({
        "argv": list(job.argv), "cwd": job.cwd or "", "outputs": sorted(job.outputs),
        "code": _git_state(),
        "inputs": {p: _output_sig(p) for p in sorted(getattr(job, "input_paths", []) or [])},
        # L0.3: bind the execution contract, not just argv/inputs.
        "deps": sorted(getattr(job, "deps", []) or []),
        "canary_dep": getattr(job, "canary_dep", None),
        "predicted_wall_s": getattr(job, "predicted_wall_s", 0.0),
        "required_free_gb": getattr(job, "required_free_gb", 0.0),
        "require_passing_verdict": getattr(job, "require_passing_verdict", None),
        "scientific_rows": getattr(job, "scientific_rows", 0),
        "performance_gate_path": getattr(job, "performance_gate_path", None),
        "certifying": getattr(job, "certifying", True),
    }, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


_OWNED_LEASE_FDS: dict[int, str] = {}


def _ofd_flock_bytes(lock_type: int) -> bytes:
    return struct.pack("hhqqi", lock_type, os.SEEK_SET, 0, 0, 0)


def _ofd_set_lock(fd: int, lock_type: int, *, blocking: bool = False) -> None:
    command = (fcntl.F_OFD_SETLKW if blocking else fcntl.F_OFD_SETLK)
    fcntl.fcntl(fd, command, _ofd_flock_bytes(lock_type))


def _ofd_conflict_type(fd: int) -> int:
    value = fcntl.fcntl(fd, fcntl.F_OFD_GETLK, _ofd_flock_bytes(fcntl.F_WRLCK))
    return struct.unpack("hhqqi", value[:struct.calcsize("hhqqi")])[0]


def _open_parent_directory_no_symlinks(path: str) -> tuple[int, str]:
    """Open the existing parent through O_NOFOLLOW at every component."""
    absolute = os.path.abspath(path)
    parent = os.path.dirname(absolute) or "/"
    pieces = [piece for piece in parent.split(os.sep) if piece]
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        current = "/"
        for piece in pieces:
            next_fd = os.open(
                piece, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=fd)
            os.close(fd)
            fd = next_fd
            current = os.path.join(current, piece)
        return fd, parent
    except BaseException:
        try:
            os.close(fd)
        except BaseException:
            pass
        raise


def _lease_payload(fd: int) -> dict:
    raw = os.pread(fd, 4096, 0)
    try:
        value = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise RuntimeError("GPU lease owner payload is missing or malformed") from exc
    required = {"schema", "controller_id", "controller_pid",
                "controller_starttime_ticks", "token", "acquired_at"}
    if not isinstance(value, dict) or set(value) != required:
        raise RuntimeError("GPU lease owner payload fields are invalid")
    return value


def _prove_fd_owns_ofd_lock(fd: int, path: str) -> dict:
    """Prove ``fd`` owns the lock; an arbitrary open fd cannot satisfy this."""
    path_state = os.lstat(path)
    fd_state = os.fstat(fd)
    if (stat.S_ISLNK(path_state.st_mode) or not stat.S_ISREG(path_state.st_mode) or
            not stat.S_ISREG(fd_state.st_mode) or path_state.st_nlink != 1 or
            (path_state.st_dev, path_state.st_ino) !=
            (fd_state.st_dev, fd_state.st_ino)):
        raise RuntimeError("GPU lease fd/path inode identity changed or was unlinked")
    # A different open file description holding the lock is visible to GETLK.
    # The owner's own OFD lock is not.  Then an independent probe must conflict;
    # if it succeeds, the candidate fd was merely an unlocked descriptor.
    if _ofd_conflict_type(fd) != fcntl.F_UNLCK:
        raise RuntimeError("inherited GPU lease fd is not the lock-owning open description")
    probe = os.open(path, os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        try:
            _ofd_set_lock(probe, fcntl.F_WRLCK)
        except (BlockingIOError, OSError) as exc:
            if isinstance(exc, BlockingIOError) or getattr(exc, "errno", None) in {11, 13}:
                pass
            else:
                raise
        else:
            _ofd_set_lock(probe, fcntl.F_UNLCK)
            raise RuntimeError("GPU lease fd is open but no active OFD lock exists")
    finally:
        os.close(probe)
    return _lease_payload(fd)


def require_active_lease(path: str = None) -> dict:
    """Refuse to run a GPU entry point unless THIS process OWNS or INHERITED the
    GPU lease (P1 — the old any-lock-holder-passes check was ownership-blind, so a
    stray direct process passed while a controller job held the lease). Proof is:
      (a) an in-process GpuLease this process acquired (registered fd whose inode
          matches the lease file), or
      (b) an inherited lease fd the controller passed via BASEMAP_GPU_LEASE_FD
          (open in this process, inode matches the lease file).
    There is deliberately no environment-variable bypass in certifying code."""
    if os.environ.get("BASEMAP_UNSAFE_NO_LEASE") is not None:
        raise RuntimeError("BASEMAP_UNSAFE_NO_LEASE is forbidden; no lease bypass is accepted")
    path = os.path.abspath(path or _lease_path())
    candidates: list[tuple[int, str | None]] = list(_OWNED_LEASE_FDS.items())
    envfd = os.environ.get("BASEMAP_GPU_LEASE_FD")
    if envfd is not None:
        try:
            candidates.append((int(envfd), os.environ.get("BASEMAP_GPU_LEASE_TOKEN")))
        except ValueError:
            pass
    for fd, expected_token in candidates:
        try:
            payload = _prove_fd_owns_ofd_lock(fd, path)
            if expected_token and payload["token"] != expected_token:
                raise RuntimeError("inherited GPU lease token differs from controller owner")
            if fd in _OWNED_LEASE_FDS and payload["token"] != _OWNED_LEASE_FDS[fd]:
                raise RuntimeError("in-process GPU lease token changed")
            return payload
        except (OSError, RuntimeError):
            if fd in _OWNED_LEASE_FDS:
                _OWNED_LEASE_FDS.pop(fd, None)
    raise RuntimeError(
        f"no owned/inherited GPU lease for {path} — refuse to run a GPU entry point "
        f"(P1). Launch via run_controller (which passes BASEMAP_GPU_LEASE_FD) or hold "
        f"an in-process GpuLease; a lease held by another process does NOT count.")


def _process_argv() -> list[str]:
    return [os.path.abspath(sys.executable), os.path.realpath(sys.argv[0]), *sys.argv[1:]]


def _capability_socket_from_environment() -> socket.socket:
    raw = os.environ.get(_CHILD_CAPABILITY_FD_ENV)
    try:
        fd = int(raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Round 0005 child has no inherited launch capability") from exc
    if fd < 0:
        raise RuntimeError("Round 0005 child launch capability descriptor is invalid")
    try:
        return socket.socket(fileno=fd)
    except OSError as exc:
        raise RuntimeError("Round 0005 child launch capability is closed/stale") from exc


def _load_child_control_json(path: str, *, label: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return value


def _controller_process_record(pid: int) -> dict:
    raw_stat = open(f"/proc/{pid}/stat", encoding="utf-8").read()
    fields = raw_stat[raw_stat.rfind(")") + 2:].split()
    cmdline = open(f"/proc/{pid}/cmdline", "rb").read()
    argv = [part.decode(errors="surrogateescape")
            for part in cmdline.rstrip(b"\0").split(b"\0")]
    executable = os.path.realpath(f"/proc/{pid}/exe")
    return {
        "pid": int(pid),
        "proc_starttime_ticks": int(fields[19]),
        "argv": argv,
        "cmdline_sha256": sha256_bytes(cmdline),
        "executable": expected_input_signature(executable),
    }


def _validate_controller_claim(manifest: dict, manifest_path: str,
                               response: dict) -> dict:
    """Verify the exact QueueAdmission claim and canonical Round-0005 CLI."""
    claim = response.get("controller_claim")
    fields = {
        "schema", "admission_id", "admission_nonce", "claim_nonce",
        "controller_id", "controller_pid", "controller_starttime_ticks",
        "controller_process", "fixture_only", "manifest",
        "construction_receipt", "ordered_job_ids", "entry_gate_sha256",
        "identity_sha256",
    }
    if not isinstance(claim, dict) or set(claim) != fields:
        raise RuntimeError("child controller claim fields are incomplete")
    body = {key: claim[key] for key in claim if key != "identity_sha256"}
    controller_pid = response.get("controller_pid")
    if (claim.get("schema") != "round0005_queue_controller_claim.v1" or
            claim.get("fixture_only") is not False or
            not isinstance(claim.get("admission_id"), int) or
            isinstance(claim.get("admission_id"), bool) or
            claim["admission_id"] <= 0 or
            not re.fullmatch(r"[0-9a-f]{64}", str(claim.get("admission_nonce", ""))) or
            not re.fullmatch(r"[0-9a-f]{64}", str(claim.get("claim_nonce", ""))) or
            not re.fullmatch(r"[0-9a-f]{64}", str(claim.get("entry_gate_sha256", ""))) or
            claim.get("controller_id") != response.get("controller_id") or
            claim.get("controller_pid") != controller_pid or
            claim.get("controller_starttime_ticks") !=
            response.get("controller_starttime_ticks") or
            claim.get("ordered_job_ids") != [job.get("id") for job in manifest["jobs"]] or
            claim.get("manifest") != expected_input_signature(manifest_path) or
            claim.get("identity_sha256") != sha256_bytes(canonical_json(body))):
        raise RuntimeError("child controller claim is not the issued production claim")
    current_process = _controller_process_record(controller_pid)
    if claim.get("controller_process") != current_process:
        raise RuntimeError("child controller process identity changed or was fabricated")
    with open(manifest["environment_manifest"], encoding="utf-8") as handle:
        environment = json.load(handle)
    expected_python = os.path.realpath(os.path.join(environment["venv_path"], "bin", "python"))
    argv = current_process["argv"]
    if (len(argv) != 4 or os.path.realpath(argv[0]) != expected_python or
            argv[1:] != ["-m", "basemap.run_controller", manifest_path]):
        raise RuntimeError("production child parent is not the canonical queue controller CLI")
    construction = claim.get("construction_receipt")
    if (not isinstance(construction, dict) or
            expected_input_signature(construction.get("canonical_path", "")) != construction):
        raise RuntimeError("QueueAdmission construction receipt is stale")
    return claim


def _clean_watchdog_verdict(value: dict, *, job: str,
                            controller_pid: int,
                            controller_claim_sha256: str) -> bool:
    if not isinstance(value, dict):
        return False
    identity = value.get("identity_sha256")
    body = {key: value[key] for key in value if key != "identity_sha256"}
    return (
        value.get("schema") == "round0005_watchdog_verdict.v1" and
        value.get("job") == job and
        value.get("controller_pid") == controller_pid and
        value.get("controller_claim_sha256") == controller_claim_sha256 and
        value.get("status") == "clean" and value.get("error") is None and
        identity == sha256_bytes(canonical_json(body)))


def _validate_completed_child_predecessors(manifest: dict, response: dict) -> None:
    """Reopen the exact completed prefix before an admitted child can work."""
    jobs = manifest.get("jobs")
    index = response.get("job_index")
    if (not isinstance(jobs, list) or not isinstance(index, int) or
            isinstance(index, bool) or index < 0 or index >= len(jobs) or
            jobs[index].get("id") != response.get("node_id")):
        raise RuntimeError("child job index/order is not the canonical manifest order")
    expected_deps = [] if index == 0 else [jobs[index - 1]["id"]]
    if jobs[index].get("deps") != expected_deps or response.get("deps") != expected_deps:
        raise RuntimeError("child dependency edge is not the completed fail-stop prefix")
    claim_sha = response["controller_claim"]["identity_sha256"]

    for raw in jobs[:index]:
        output_map = {
            path: expected_input_signature(path) for path in raw["outputs"]
        }
        controller = _load_child_control_json(
            raw["manifest"], label=f"{raw['id']} controller receipt")
        done = _load_child_control_json(
            raw["done_marker"], label=f"{raw['id']} done marker")
        record = controller.get("record")
        watchdog = record.get("watchdog_verdict") if isinstance(record, dict) else None
        predecessor_capability = controller.get("child_capability")
        predecessor_ack = controller.get("child_capability_ack")
        if (controller.get("schema") != "round0005_controller_job.v3" or
                controller.get("fixture_only") is not False or
                controller.get("controller_id") != response["controller_id"] or
                controller.get("job") != raw["id"] or
                controller.get("manifest_sha256") != response["manifest_sha256"] or
                controller.get("controller_claim_sha256") != claim_sha or
                not isinstance(predecessor_capability, dict) or
                set(predecessor_capability) != _CHILD_CONTRACT_FIELDS or
                predecessor_capability.get("controller_claim") !=
                response["controller_claim"] or
                predecessor_capability.get("node_id") != raw["id"] or
                predecessor_ack != {
                    "schema": "round0005_child_capability_ack.v1",
                    "launch_nonce": predecessor_capability.get("launch_nonce"),
                    "pid": predecessor_capability.get("child_pid"),
                    "capability_sha256": sha256_bytes(canonical_json(
                        predecessor_capability)),
                } or
                controller.get("runtime_contract") != raw or
                controller.get("post_child_integrity") is not True or
                controller.get("watchdog_clean") is not True or
                not isinstance(record, dict) or record.get("status") != "ok" or
                record.get("child_capability_ack") != predecessor_ack or
                record.get("exit_code") != 0 or
                record.get("output_signatures") != output_map or
                not _clean_watchdog_verdict(
                    watchdog, job=raw["id"],
                    controller_pid=response["controller_pid"],
                    controller_claim_sha256=claim_sha)):
            raise RuntimeError(
                f"completed predecessor {raw['id']} controller evidence is invalid")
        if (done.get("schema") != "round0005_controller_done.v3" or
                done.get("fixture_only") is not False or
                done.get("status") != "ok" or done.get("job") != raw["id"] or
                done.get("manifest_sha256") != response["manifest_sha256"] or
                done.get("controller_claim_sha256") != claim_sha or
                done.get("runtime_contract_sha256") !=
                sha256_bytes(canonical_json(raw)) or
                done.get("output_signatures") != output_map or
                done.get("post_child_integrity") is not True or
                done.get("watchdog_verdict_sha256") !=
                watchdog.get("identity_sha256")):
            raise RuntimeError(
                f"completed predecessor {raw['id']} done evidence is invalid")
        expected_input_signature(raw["log"])
        expected_input_signature(raw["manifest"])
        expected_input_signature(raw["done_marker"])

    # No current/future canonical result or control leaf may predate this child.
    delegated_ordinal = response.get("delegated_ordinal")
    allowed_existing: set[str] = set()
    if response.get("delegated") is True:
        if (response.get("node_id") != "scalar_equivalence" or
                delegated_ordinal not in {0, 1}):
            raise RuntimeError("child descendant ordinal is not controller-authorized")
        if delegated_ordinal == 1:
            # The first scorer descendant has completed before the second one
            # can be issued.  Its four private outputs are the only current-job
            # leaves allowed to exist at this boundary.
            allowed_existing.update(jobs[index]["outputs"][1:5])
    elif delegated_ordinal is not None:
        raise RuntimeError("direct child capability carries a descendant ordinal")
    for raw in jobs[index:]:
        paths = [*raw["outputs"], raw["done_marker"], raw["log"], raw["manifest"]]
        present = [path for path in paths
                   if os.path.lexists(path) and path not in allowed_existing]
        if present:
            raise RuntimeError(
                f"child launch is stale/out of DAG order; future evidence exists: {present}")
    for path in sorted(allowed_existing):
        expected_input_signature(path)


def _validate_live_controller_journal(manifest: dict, response: dict) -> None:
    """Require the immutable real-controller prefix that precedes capability issue."""
    root = os.path.realpath(manifest["controller_checkpoints_dir"])
    if (not root.startswith("/data/") or not os.path.isdir(root) or
            os.path.islink(root)):
        raise RuntimeError("child controller checkpoint root is invalid")
    all_names = sorted(os.listdir(root))
    temporary_logs = [name for name in all_names
                      if name.startswith(f".{response['node_id']}.child-log.")]
    if len(temporary_logs) != 1:
        raise RuntimeError("child controller has no unique live private log")
    temporary_status = os.lstat(os.path.join(root, temporary_logs[0]))
    if (not stat.S_ISREG(temporary_status.st_mode) or
            temporary_status.st_nlink != 1):
        raise RuntimeError("child controller private log has an unsupported identity")
    telemetry_temps = [name for name in all_names if re.fullmatch(
        r"\.\d{6}-gpu-telemetry\.json\.tmp\.[0-9a-f]{32}", name)]
    names = [name for name in all_names
             if name not in temporary_logs and name not in telemetry_temps]
    records = []
    claim_sha = response["controller_claim"]["identity_sha256"]
    for sequence, name in enumerate(names):
        path = os.path.join(root, name)
        expected_input_signature(path)
        record = _load_child_control_json(path, label="controller checkpoint")
        event = record.get("event")
        if (record.get("schema") != "round0005_controller_checkpoint.v1" or
                record.get("sequence") != sequence or
                not isinstance(event, str) or
                name != f"{sequence:06d}-{event}.json" or
                record.get("controller_id") != response["controller_id"] or
                record.get("controller_claim_sha256") != claim_sha):
            raise RuntimeError("child controller checkpoint sequence is forged")
        records.append(record)
    if any(value["event"] in {"exception", "runtime-violation", "terminal"}
           for value in records):
        raise RuntimeError("child controller journal is failed, terminal, or stale")
    core = [value for value in records if value["event"] != "gpu-telemetry"]
    expected = ["admission", "lease-acquired"]
    jobs = manifest["jobs"]
    index = response["job_index"]
    for raw in jobs[:index]:
        expected.extend(("boundary", "launch", "completion", "cumulative-registry"))
    expected.append("boundary")
    if response.get("delegated") is True:
        expected.append("launch")
    if [value["event"] for value in core] != expected:
        raise RuntimeError("child controller checkpoint DAG prefix is incomplete/reordered")
    admission_checkpoint, lease_checkpoint = core[:2]
    construction_path = admission_checkpoint.get("construction_receipt")
    construction = _load_child_control_json(
        construction_path, label="admission construction receipt")
    construction_body = {
        key: construction[key] for key in construction if key != "identity_sha256"
    }
    if (admission_checkpoint.get("manifest_sha256") !=
            response["manifest_sha256"] or
            admission_checkpoint.get("controller_claim") != response["controller_claim"] or
            admission_checkpoint.get("controller_entry_gate_sha256") !=
            response["controller_claim"]["entry_gate_sha256"] or
            sha256_bytes(canonical_json(
                admission_checkpoint.get("controller_entry_gate"))) !=
            response["controller_claim"]["entry_gate_sha256"] or
            admission_checkpoint.get("runtime_jobs_match_manifest") is not True or
            construction.get("schema") != "round0005_integrity_receipt.v3" or
            construction.get("phase") != "admission-construction" or
            construction.get("status") != "matched" or
            construction.get("expected") != construction.get("observed") or
            construction.get("original_manifest_sha256") !=
            response["manifest_sha256"] or
            construction.get("identity_sha256") !=
            sha256_bytes(canonical_json(construction_body)) or
            expected_input_signature(construction_path).get("kind") != "file" or
            lease_checkpoint.get("lease_owner") != response["lease_identity"] or
            lease_checkpoint.get("controller_claim_sha256") != claim_sha):
        raise RuntimeError("child controller admission/lease attestation is invalid")
    current_boundary = core[-2] if response.get("delegated") is True else core[-1]
    gate_receipt = current_boundary.get("gate_receipt")
    receipt_path = gate_receipt.get("receipt_path") if isinstance(gate_receipt, dict) else None
    receipt = _load_child_control_json(receipt_path, label="controller gate receipt")
    receipt_body = {key: receipt[key] for key in receipt if key != "identity_sha256"}
    if (current_boundary.get("job") != response["node_id"] or
            not isinstance(gate_receipt, dict) or
            gate_receipt.get("comprehensive_integrity", {}).get(
                "integrity_match") is not True or
            receipt.get("schema") != "round0005_integrity_receipt.v3" or
            receipt.get("status") != "matched" or
            receipt.get("expected") != receipt.get("observed") or
            receipt.get("identity_sha256") !=
            sha256_bytes(canonical_json(receipt_body)) or
            expected_input_signature(receipt_path).get("kind") != "file"):
        raise RuntimeError("child controller current boundary evidence is invalid")


def _validate_child_live_authority(manifest: dict, manifest_path: str,
                                   token: str, response: dict) -> dict:
    """Independently reopen release/source/gate authority in the actual child."""
    from .gate_preparation import validate_gate_preparation_receipt
    from .roundwatch_gate import RoundwatchGateAuthority

    if manifest.get("round_id") == "0017":
        validate_exact_program = importlib.import_module(
            ".round0017_program", __package__).validate_exact_program
    elif manifest.get("round_id") == "0016":
        validate_exact_program = importlib.import_module(
            ".round0016_program", __package__).validate_exact_program
    elif manifest.get("round_id") == "0015":
        from .round0015_program import validate_exact_program
    elif manifest.get("round_id") == "0014":
        from .round0014_program import validate_exact_program
    else:
        from .round0005_program import validate_exact_program

    validate_exact_program(
        manifest, manifest_path=manifest_path,
        repo_root=os.path.realpath(manifest["repo_root"]))
    validate_gate_preparation_receipt(
        manifest["gate_preparation_receipt"], manifest_path=manifest_path,
        manifest=manifest)
    current_gate = RoundwatchGateAuthority().check(
        manifest=manifest, manifest_path=manifest_path,
        manifest_sha256=token)
    stable_fields = {
        "instance_id", "gate_id", "authority", "round_event",
        "gate_prepared_event", "control_event",
    }
    current = current_gate.get("event_identity", {})
    bound = response.get("gate_identity", {})
    if ({key: current.get(key) for key in stable_fields} !=
            {key: bound.get(key) for key in stable_fields}):
        raise RuntimeError("child current Roundwatch control differs from capability")
    _validate_live_controller_journal(manifest, response)
    _validate_completed_child_predecessors(manifest, response)
    return current_gate


def _require_child_cache_policy(manifest: dict, expected: dict) -> dict:
    from .queue_admission import observe_round0005_cache_policy

    observed = observe_round0005_cache_policy(manifest)
    if not isinstance(expected, dict) or observed != expected:
        raise RuntimeError(
            "child cache policy differs from the controller-sealed launch boundary")
    return observed


def require_round0005_child_admission(expected_script: str) -> dict:
    """Consume one controller/parent launch capability before output or CUDA."""
    global _ACTIVE_CHILD_ADMISSION
    if _ACTIVE_CHILD_ADMISSION is not None:
        if _ACTIVE_CHILD_ADMISSION.get("script") != expected_script:
            raise RuntimeError("active child admission is bound to another script")
        return _ACTIVE_CHILD_ADMISSION
    token = os.environ.get("BASEMAP_ROUND0005_ADMISSION")
    manifest_path = os.environ.get("BASEMAP_ROUND0005_MANIFEST")
    node_id = os.environ.get("BASEMAP_ROUND0005_NODE")
    if (not token or not re.fullmatch(r"[0-9a-f]{64}", token) or
            not manifest_path or not node_id):
        raise RuntimeError("Round 0005 executable requires controller admission identity")
    manifest_path = os.path.realpath(manifest_path)
    if sha256_file(manifest_path) != token:
        raise RuntimeError("Round 0005 child manifest differs from controller admission")
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    matches = [job for job in manifest.get("jobs", []) if job.get("id") == node_id]
    if len(matches) != 1:
        raise RuntimeError("Round 0005 child node is not in the exact admitted manifest")
    job = matches[0]
    canonical_script = os.path.realpath(os.path.join(manifest["repo_root"], expected_script))
    current_script = os.path.realpath(sys.argv[0])
    if current_script != canonical_script:
        raise RuntimeError("Round 0005 executable/script is not canonical")
    launch_nonce = os.environ.get(_CHILD_CAPABILITY_NONCE_ENV)
    if not launch_nonce or not re.fullmatch(r"[0-9a-f]{64}", launch_nonce):
        raise RuntimeError("Round 0005 child launch nonce is missing")
    channel = _capability_socket_from_environment()
    hello = {
        "schema": "round0005_child_hello.v1", "launch_nonce": launch_nonce,
        "pid": os.getpid(), "ppid": os.getppid(), "node_id": node_id,
        "script": expected_script, "argv": _process_argv(),
        "manifest_sha256": token,
    }
    try:
        channel.sendall(canonical_json(hello))
        channel.settimeout(180.0)
        response = json.loads(channel.recv(1 << 20).decode("utf-8"))
    except Exception as exc:
        channel.close()
        raise RuntimeError("controller did not issue a live child launch verdict") from exc
    if not isinstance(response, dict) or set(response) != _CHILD_CONTRACT_FIELDS:
        channel.close()
        raise RuntimeError("child launch capability contract is incomplete")
    delegated = response["delegated"] is True
    delegated_ordinal = response.get("delegated_ordinal")
    expected_direct_argv = job["argv"]
    descendant_argv = response.get("descendant_argv")
    expected_descendants = None
    if delegated and node_id == "scalar_equivalence":
        from experiments.compare_panel_cache import scalar_equivalence_descendant_argvs
        expected_descendants = scalar_equivalence_descendant_argvs(
            list(job["argv"]), repo_root=manifest["repo_root"])
    delegation_valid = (
        (not delegated and response.get("descendant_limit") in {0, 2} and
         isinstance(descendant_argv, list) and delegated_ordinal is None) or
        (delegated and node_id == "scalar_equivalence" and
         expected_script == "experiments/score_complete_panel.py" and
         isinstance(descendant_argv, list) and _process_argv() in descendant_argv and
         delegated_ordinal in {0, 1} and expected_descendants is not None and
         _process_argv() == expected_descendants[delegated_ordinal] and
         response.get("descendant_limit") == 0))
    if (response["schema"] != "round0005_child_launch_capability.v1" or
            response["launch_nonce"] != launch_nonce or
            response["child_pid"] != os.getpid() or response["parent_pid"] != os.getppid() or
            response["node_id"] != node_id or response["script"] != expected_script or
            response["manifest_sha256"] != token or response["argv"] != _process_argv() or
            (not delegated and response["argv"] != expected_direct_argv) or
            not delegation_valid or
            response["job_index"] != [item["id"] for item in manifest["jobs"]].index(node_id) or
            response["deps"] != job["deps"] or
            response["deadline_utc"] != manifest["deadline_utc"] or
            float(response["gpu_hours_cap"]) != float(manifest["gpu_hours_cap"]) or
            int(response["gpu_memory_cap_mb"]) != int(job["node_policy"]["gpu_memory_cap_mb"]) or
            float(response["p90_wall_s"]) != float(job["p90_wall_s"]) or
            not isinstance(response["gate_identity"], dict) or
            not isinstance(response["watchdog_channel"], dict) or
            not isinstance(response["cache_policy"], dict)):
        channel.close()
        raise RuntimeError("child launch capability differs from exact job/current gate")
    _validate_controller_claim(manifest, manifest_path, response)
    current_environment = {key: os.environ.get(key) for key in response["environment"]}
    if current_environment != response["environment"] or \
            set(os.environ) != set(response["environment"]):
        channel.close()
        raise RuntimeError("child environment differs from the exact launch capability")
    _require_child_cache_policy(manifest, response["cache_policy"])
    payload = require_active_lease(manifest["lease_path"])
    if (payload != response["lease_identity"] or
            payload["controller_pid"] != response["controller_pid"] or
            (not delegated and payload["controller_pid"] != os.getppid())):
        channel.close()
        raise RuntimeError("child lease/controller identity differs from launch capability")
    for signature in manifest["global_input_registry"]:
        if expected_input_signature(signature["canonical_path"]) != signature:
            raise RuntimeError(
                f"Round 0005 child input changed before executable start: "
                f"{signature['canonical_path']}")
    _validate_child_live_authority(manifest, manifest_path, token, response)
    accepted_sha = sha256_bytes(canonical_json(response))
    try:
        channel.sendall(canonical_json({
            "schema": "round0005_child_capability_ack.v1",
            "launch_nonce": launch_nonce, "pid": os.getpid(),
            "capability_sha256": accepted_sha,
        }))
    finally:
        channel.close()
        os.environ.pop(_CHILD_CAPABILITY_FD_ENV, None)
        os.environ.pop(_CHILD_CAPABILITY_NONCE_ENV, None)
    _ACTIVE_CHILD_ADMISSION = {
        **response, "script": expected_script, "capability_sha256": accepted_sha,
        "manifest": manifest, "job": job, "descendants_consumed": 0,
    }
    _GENUINE_CHILD_ADMISSION_SEALS[id(_ACTIVE_CHILD_ADMISSION)] = accepted_sha
    return _ACTIVE_CHILD_ADMISSION


def require_active_round0005_child_admission() -> dict:
    """Prove the current process owns a genuine, still-current child capability.

    This is the device-selection boundary used by scoring libraries.  Environment
    strings, a copied dict, a fixture admission, or a self-acquired public lease
    are deliberately insufficient.
    """
    active = _ACTIVE_CHILD_ADMISSION
    if type(active) is not dict:
        raise RuntimeError("CUDA scoring requires a genuine controller child capability")
    try:
        contract = {field: active[field] for field in _CHILD_CONTRACT_FIELDS}
        contract_sha = sha256_bytes(canonical_json(contract))
    except Exception as exc:
        raise RuntimeError("active child capability is incomplete") from exc
    if (_GENUINE_CHILD_ADMISSION_SEALS.get(id(active)) != contract_sha or
            active.get("capability_sha256") != contract_sha or
            active.get("child_pid") != os.getpid() or
            active.get("manifest_sha256") != os.environ.get("BASEMAP_ROUND0005_ADMISSION")):
        raise RuntimeError("active child capability is forged, copied, or stale")
    manifest_path = os.path.realpath(
        os.environ.get("BASEMAP_ROUND0005_MANIFEST", ""))
    if (not manifest_path or sha256_file(manifest_path) != active["manifest_sha256"] or
            active.get("manifest", {}).get("fixture_only") is True):
        raise RuntimeError("active child capability is not bound to a production manifest")
    if _proc_starttime_ticks(active["controller_pid"]) != active["controller_starttime_ticks"]:
        raise RuntimeError("active child capability controller is no longer genuine/live")
    lease = require_active_lease(active["manifest"]["lease_path"])
    if lease != active["lease_identity"]:
        raise RuntimeError("active child capability lease identity changed")
    current = _descendant_current_gate(active)["event_identity"]
    bound = active["gate_identity"]
    stable_fields = {
        "instance_id", "gate_id", "authority", "round_event",
        "gate_prepared_event", "control_event",
    }
    if ({key: current.get(key) for key in stable_fields} !=
            {key: bound.get(key) for key in stable_fields}):
        raise RuntimeError("active child capability gate/control identity changed")
    return active


def _new_child_capability_channel() -> tuple[socket.socket, socket.socket]:
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    parent.set_inheritable(False)
    child.set_inheritable(True)
    return parent, child


def _receive_child_hello(channel: socket.socket, process: subprocess.Popen, *,
                         launch_nonce: str, job, script: str) -> dict:
    channel.settimeout(180.0)
    try:
        hello = json.loads(channel.recv(1 << 20).decode("utf-8"))
    except Exception as exc:
        raise RuntimeError("child did not consume its inherited launch capability") from exc
    expected_fields = {
        "schema", "launch_nonce", "pid", "ppid", "node_id", "script", "argv",
        "manifest_sha256",
    }
    if (not isinstance(hello, dict) or set(hello) != expected_fields or
            hello.get("schema") != "round0005_child_hello.v1" or
            hello.get("launch_nonce") != launch_nonce or
            hello.get("pid") != process.pid or hello.get("ppid") != os.getpid() or
            hello.get("node_id") != job.name or hello.get("script") != script or
            hello.get("argv") != job.argv):
        raise RuntimeError("child hello differs from the exact Popen/job identity")
    return hello


def _issue_child_capability(channel: socket.socket, *, contract: dict,
                            process: subprocess.Popen) -> dict:
    channel.sendall(canonical_json(contract))
    channel.settimeout(180.0)
    try:
        ack = json.loads(channel.recv(1 << 20).decode("utf-8"))
    except Exception as exc:
        raise RuntimeError("child did not authenticate the launch capability") from exc
    expected = {
        "schema": "round0005_child_capability_ack.v1",
        "launch_nonce": contract["launch_nonce"], "pid": process.pid,
        "capability_sha256": sha256_bytes(canonical_json(contract)),
    }
    if ack != expected:
        raise RuntimeError("child capability acknowledgement is forged/incomplete")
    return ack


def _cache_policy_from_comprehensive_boundary(boundary: dict) -> dict:
    """Return only the exact cache policy proven by a comprehensive boundary."""
    required = {"phase", "expected", "observed", "integrity_match"}
    if (not isinstance(boundary, dict) or set(boundary) != required or
            boundary.get("integrity_match") is not True or
            not isinstance(boundary.get("expected"), dict) or
            not isinstance(boundary.get("observed"), dict) or
            boundary["expected"] != boundary["observed"]):
        raise RuntimeError("production launch integrity boundary is incomplete or unmatched")
    expected = boundary["expected"].get("cache_policy")
    observed = boundary["observed"].get("cache_policy")
    if not isinstance(observed, dict) or not observed or observed != expected:
        raise RuntimeError("production launch cache policy is absent or mismatched")
    return observed


def _build_round0005_child_contract(*, launch_nonce: str, controller_id: str,
                                    child_pid: int, job, jobs: list,
                                    child_environment: dict, manifest: dict,
                                    manifest_sha256: str, gate_identity: dict,
                                    lease_identity: dict, telemetry_interval_s: float,
                                    watchdog_pid: int, watchdog_nonce: str,
                                    launch_integrity: dict,
                                    controller_claim: dict) -> dict:
    """Build the genuine non-fixture capability from observed boundary state."""
    script = job.node_policy["canonical_script"]
    return {
        "schema": "round0005_child_launch_capability.v1",
        "launch_nonce": launch_nonce, "controller_id": controller_id,
        "controller_pid": os.getpid(),
        "controller_starttime_ticks": _proc_starttime_ticks(os.getpid()),
        "parent_pid": os.getpid(), "child_pid": child_pid,
        "node_id": job.name,
        "job_index": [value.name for value in jobs].index(job.name),
        "deps": list(job.deps), "argv": list(job.argv),
        "environment": dict(child_environment),
        "manifest_sha256": manifest_sha256,
        "gate_identity": gate_identity,
        "lease_identity": lease_identity,
        "deadline_utc": manifest["deadline_utc"],
        "gpu_hours_cap": manifest["gpu_hours_cap"],
        "gpu_memory_cap_mb": job.gpu_memory_cap_mb,
        "p90_wall_s": job.p90_wall_s,
        "telemetry_interval_s": float(telemetry_interval_s),
        "watchdog_channel": {
            "pid": watchdog_pid, "nonce": watchdog_nonce,
            "authenticated_result_required": True,
        },
        "delegated": False, "delegated_ordinal": None,
        "cache_policy": _cache_policy_from_comprehensive_boundary(
            launch_integrity),
        "controller_claim": controller_claim,
        "script": script,
        "descendant_limit": (2 if job.name == "scalar_equivalence" else 0),
        "descendant_argv": (
            __import__(
                "experiments.compare_panel_cache",
                fromlist=["scalar_equivalence_descendant_argvs"]
            ).scalar_equivalence_descendant_argvs(
                list(job.argv), repo_root=manifest["repo_root"])
            if job.name == "scalar_equivalence" else []),
    }


def _descendant_current_gate(active: dict) -> dict:
    manifest = active["manifest"]
    manifest_path = os.environ["BASEMAP_ROUND0005_MANIFEST"]
    token = os.environ["BASEMAP_ROUND0005_ADMISSION"]
    if sha256_file(manifest_path) != token:
        raise RuntimeError("descendant parent manifest changed")
    from .gate_preparation import validate_gate_preparation_receipt
    from .release_preflight import validate_release_preflight_receipt
    from .roundwatch_gate import RoundwatchGateAuthority
    validate_gate_preparation_receipt(
        manifest["gate_preparation_receipt"], manifest_path=manifest_path,
        manifest=manifest)
    release_signatures = [
        entry["signature"]
        for entry in manifest.get("program_inputs", [])
        if entry.get("role") == "release_preflight_receipt"]
    if len(release_signatures) != 1:
        raise RuntimeError("descendant parent has no unique release receipt")
    validate_release_preflight_receipt(
        release_signatures[0]["canonical_path"],
        expected_identity_sha256=manifest["release_preflight_identity"],
        expected_signature=release_signatures[0])
    for signature in manifest["global_input_registry"]:
        if expected_input_signature(signature["canonical_path"]) != signature:
            raise RuntimeError("descendant parent observed signed-input drift")
    _require_child_cache_policy(manifest, active["cache_policy"])
    return RoundwatchGateAuthority().check(
        manifest=manifest, manifest_path=manifest_path, manifest_sha256=token)


def run_round0005_admitted_descendant(argv: list[str], *, cwd: str,
                                       text: bool = True,
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE) -> subprocess.CompletedProcess:
    """Launch one exact scorer descendant from an active genuine controller child."""
    global _ACTIVE_CHILD_ADMISSION
    active = _ACTIVE_CHILD_ADMISSION
    if (not isinstance(active, dict) or active.get("node_id") != "scalar_equivalence" or
            active.get("delegated") is not False):
        raise RuntimeError("scorer descendant requires the genuine scalar-equivalence child")
    ordinal = int(active.get("descendants_consumed", 0))
    allowed = active.get("descendant_argv")
    if (not isinstance(allowed, list) or ordinal >= active.get("descendant_limit", 0) or
            ordinal >= len(allowed) or list(argv) != allowed[ordinal]):
        raise RuntimeError("scorer descendant argv/order is not controller-authorized")
    if os.path.realpath(cwd) != os.path.realpath(active["manifest"]["repo_root"]):
        raise RuntimeError("scorer descendant cwd differs from the admitted checkout")
    active["descendants_consumed"] = ordinal + 1
    _descendant_current_gate(active)
    lease_identity = _prove_fd_owns_ofd_lock(
        int(os.environ["BASEMAP_GPU_LEASE_FD"]), active["manifest"]["lease_path"])
    parent_channel, child_channel = _new_child_capability_channel()
    launch_nonce = secrets.token_hex(32)
    environment = dict(os.environ)
    environment[_CHILD_CAPABILITY_FD_ENV] = str(child_channel.fileno())
    environment[_CHILD_CAPABILITY_NONCE_ENV] = launch_nonce
    parent_pid = os.getpid()
    process = None
    try:
        process = subprocess.Popen(
            _parent_death_exec_argv(list(argv), expected_parent=parent_pid),
            cwd=cwd, text=text, stdout=stdout, stderr=stderr,
            close_fds=True,
            pass_fds=(int(os.environ["BASEMAP_GPU_LEASE_FD"]), child_channel.fileno()),
            env=environment)
        child_channel.close(); child_channel = None
        parent_channel.settimeout(20.0)
        hello = json.loads(parent_channel.recv(1 << 20).decode("utf-8"))
        if (not isinstance(hello, dict) or hello.get("pid") != process.pid or
                hello.get("ppid") != parent_pid or hello.get("argv") != list(argv) or
                hello.get("launch_nonce") != launch_nonce or
                hello.get("script") != "experiments/score_complete_panel.py"):
            raise RuntimeError("scorer descendant hello is not exact")
        gate = _descendant_current_gate(active)
        contract = {
            "schema": "round0005_child_launch_capability.v1",
            "launch_nonce": launch_nonce, "controller_id": active["controller_id"],
            "controller_pid": active["controller_pid"],
            "controller_starttime_ticks": active["controller_starttime_ticks"],
            "parent_pid": parent_pid, "child_pid": process.pid,
            "node_id": active["node_id"], "job_index": active["job_index"],
            "deps": active["deps"], "argv": list(argv), "environment": environment,
            "manifest_sha256": active["manifest_sha256"],
            "gate_identity": gate["event_identity"], "lease_identity": lease_identity,
            "deadline_utc": active["deadline_utc"],
            "gpu_hours_cap": active["gpu_hours_cap"],
            "gpu_memory_cap_mb": active["gpu_memory_cap_mb"],
            "p90_wall_s": active["p90_wall_s"],
            "telemetry_interval_s": active["telemetry_interval_s"],
            "watchdog_channel": active["watchdog_channel"],
            "delegated": True, "script": "experiments/score_complete_panel.py",
            "descendant_limit": 0, "descendant_argv": [list(argv)],
            "delegated_ordinal": ordinal,
            "cache_policy": active["cache_policy"],
            "controller_claim": active["controller_claim"],
        }
        _issue_child_capability(parent_channel, contract=contract, process=process)
        parent_channel.close(); parent_channel = None
        out, err = process.communicate()
        return subprocess.CompletedProcess(argv, process.returncode, out, err)
    except Exception:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait(timeout=5)
        raise
    finally:
        for channel in (parent_channel, child_channel):
            if channel is not None:
                try:
                    channel.close()
                except OSError:
                    pass


def gpu_snapshot(*, strict: bool = False, manifest: dict | None = None) -> dict:
    expected_uuid = expected_name = expected_driver = None
    if manifest is not None:
        try:
            with open(manifest["environment_manifest"], encoding="utf-8") as handle:
                sealed = json.load(handle)
            expected_uuid = sealed["gpu_uuid"]
            expected_name = sealed["gpu_name"]
            expected_driver = sealed["gpu_driver"]
            if (expected_name != _ROUND0005_GSV_GPU_NAME or
                    manifest["child_environment"]["CUDA_VISIBLE_DEVICES"] !=
                    expected_uuid):
                raise RuntimeError("runtime telemetry manifest GPU binding is invalid")
        except Exception as exc:
            if strict:
                raise RuntimeError("runtime telemetry cannot reopen sealed GPU identity") from exc
    elif strict:
        raise RuntimeError("strict Round 0005 GPU telemetry requires its queue manifest")

    observer_before = None

    def q(kind, fields):
        nonlocal observer_before
        try:
            current = expected_input_signature(_NVIDIA_SMI_EXECUTABLE)
            if observer_before is None:
                observer_before = current
            elif current != observer_before:
                raise RuntimeError("sealed nvidia-smi identity changed between queries")
            out = subprocess.check_output(
                [_NVIDIA_SMI_EXECUTABLE, f"--query-{kind}={fields}",
                 "--format=csv,noheader,nounits"],
                text=True, timeout=15, env=dict(_NVIDIA_SMI_ENVIRONMENT))
            if expected_input_signature(_NVIDIA_SMI_EXECUTABLE) != observer_before:
                raise RuntimeError("sealed nvidia-smi identity changed during query")
            return [l.strip() for l in out.strip().splitlines() if l.strip()]
        except Exception as e:
            if strict:
                raise RuntimeError(f"GPU telemetry query failed closed: {kind}/{fields}: {e}") from e
            return [f"err:{e}"]
    gpu = q(
        "gpu",
        "uuid,name,driver_version,memory.free,memory.used,memory.total,"
        "utilization.gpu,power.draw")
    apps = q("compute-apps", "gpu_uuid,pid,used_gpu_memory")
    pids = []
    app_records = []
    for a in apps:
        try:
            pieces = [piece.strip() for piece in a.split(",")]
            gpu_uuid = pieces[0]
            pid = int(pieces[1]); used_mb = float(pieces[2])
            if expected_uuid is not None and gpu_uuid != expected_uuid:
                raise RuntimeError("compute process row belongs to another GPU UUID")
            pids.append(pid)
            app_records.append({
                "gpu_uuid": gpu_uuid, "pid": pid, "used_memory_mb": used_mb})
        except Exception as exc:
            if strict:
                raise RuntimeError(f"GPU compute-app telemetry is malformed: {a!r}") from exc
    gpu_uuid = gpu_name = gpu_driver = None
    free_mb = used_mb = total_mb = None
    try:
        if len(gpu) != 1:
            raise RuntimeError("Round 0005 runtime telemetry requires exactly one GPU row")
        pieces = [piece.strip() for piece in gpu[0].split(",")]
        gpu_uuid, gpu_name, gpu_driver = pieces[:3]
        free_mb, used_mb, total_mb = map(float, pieces[3:6])
        if expected_uuid is not None and (
                gpu_uuid != expected_uuid or gpu_name != expected_name or
                gpu_driver != expected_driver):
            raise RuntimeError("live runtime GPU differs from the sealed manifest UUID")
    except Exception as exc:
        if strict:
            raise RuntimeError(f"GPU memory telemetry is malformed: {gpu!r}") from exc
    return {"at": _utcnow(), "gpu": gpu[0] if gpu else None, "compute_apps": apps,
            "compute_app_records": app_records, "compute_pids": pids,
            "free_mb": free_mb, "used_mb": used_mb, "total_mb": total_mb,
            "n_co_tenants": len(pids), "observer": observer_before,
            "gpu_uuid": gpu_uuid, "gpu_name": gpu_name, "gpu_driver": gpu_driver}


def _proc_starttime_ticks(pid: int) -> int:
    # The comm field may contain spaces and ')'; split only after its final ')'.
    raw = open(f"/proc/{pid}/stat", encoding="utf-8").read()
    fields = raw[raw.rfind(")") + 2:].split()
    return int(fields[19])  # field 22 overall, field 3 is fields[0]


def process_identity(pid: int, *, marker: str, gpu_memory_budget_mb: int,
                     service_identity: str | None = None) -> dict:
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        raise ValueError("allowed service PID must be a positive integer")
    cmdline_bytes = open(f"/proc/{pid}/cmdline", "rb").read()
    cmdline = cmdline_bytes.replace(b"\x00", b" ").decode(errors="replace")
    if not cmdline_bytes or marker not in cmdline:
        raise RuntimeError(f"PID {pid} command line does not contain service marker {marker!r}")
    return {
        "pid": pid,
        "proc_starttime_ticks": _proc_starttime_ticks(pid),
        "cmdline_sha256": sha256_bytes(cmdline_bytes),
        "service_identity": service_identity or marker,
        "marker": marker,
        "gpu_memory_budget_mb": int(gpu_memory_budget_mb),
    }


def validate_allowed_processes(expected: list[dict], *, snapshot: dict | None = None,
                               allow_pids=()) -> dict:
    """Re-probe PID identity and current VRAM immediately before launch."""
    observed = []
    for item in expected:
        current = process_identity(
            item["pid"], marker=item["marker"],
            gpu_memory_budget_mb=item["gpu_memory_budget_mb"],
            service_identity=item["service_identity"])
        if current != item:
            raise RuntimeError(
                f"allowed service identity drift/PID reuse for {item['pid']}: "
                f"expected={item!r} observed={current!r}")
        observed.append(current)
    snap = snapshot or gpu_snapshot(strict=True)
    expected_by_pid = {item["pid"]: item for item in expected}
    additionally_allowed = {int(pid) for pid in allow_pids}
    unknown = [record for record in snap["compute_app_records"]
               if record["pid"] not in expected_by_pid and
               record["pid"] not in additionally_allowed]
    if unknown:
        raise RuntimeError(f"unknown GPU compute processes fail admission: {unknown}")
    over_budget = [record for record in snap["compute_app_records"]
                   if record["pid"] in expected_by_pid and
                   record["used_memory_mb"] >
                   expected_by_pid[record["pid"]]["gpu_memory_budget_mb"]]
    if over_budget:
        raise RuntimeError(f"allow-listed GPU service exceeded its memory budget: {over_budget}")
    return {"expected": expected, "observed": observed, "gpu_snapshot": snap}


def _process_record(pid: int) -> dict:
    raw = open(f"/proc/{pid}/stat", encoding="utf-8").read()
    fields = raw[raw.rfind(")") + 2:].split()
    cmdline = open(f"/proc/{pid}/cmdline", "rb").read()
    return {
        "pid": int(pid), "ppid": int(fields[1]),
        "proc_starttime_ticks": int(fields[19]),
        "cmdline_sha256": sha256_bytes(cmdline),
    }


def _process_tree(root_pid: int) -> list[dict]:
    records: dict[int, dict] = {}
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            record = _process_record(int(name))
        except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
            continue
        records[record["pid"]] = record
    if root_pid not in records:
        raise RuntimeError(f"launched child PID {root_pid} vanished from /proc")
    selected = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, record in records.items():
            if pid not in selected and record["ppid"] in selected:
                selected.add(pid)
                changed = True
    return [records[pid] for pid in sorted(selected)]


class RuntimeGpuViolation(RuntimeError):
    def __init__(self, message: str, *, snapshot: dict):
        super().__init__(message)
        self.snapshot = snapshot


class RuntimeGpuAccountant:
    """Revalidate exact service/tree identities and memory accounting."""

    def __init__(self, *, manifest: dict, job: "Job", root_pid: int,
                 lease: GpuLease):
        self.manifest = manifest
        self.job = job
        self.root_pid = int(root_pid)
        self.lease = lease
        self.identities: dict[int, dict] = {}

    def snapshot(self, *, supplied_gpu: dict | None = None) -> dict:
        lease_owner = self.lease.verify_current()
        tree = _process_tree(self.root_pid)
        for record in tree:
            previous = self.identities.get(record["pid"])
            if previous is not None and previous != record:
                snapshot = {"process_tree": tree, "previous_identity": previous,
                            "observed_identity": record}
                raise RuntimeGpuViolation(
                    f"process PID/start-time/cmdline identity drift for {record['pid']}",
                    snapshot=snapshot)
            self.identities.setdefault(record["pid"], record)
        gpu = supplied_gpu or gpu_snapshot(strict=True, manifest=self.manifest)
        try:
            allowed = validate_allowed_processes(
                self.manifest["allowed_processes"], snapshot=gpu,
                allow_pids={record["pid"] for record in tree})
        except Exception as exc:
            snapshot = {
                "schema": "round0005_runtime_gpu_telemetry.v2",
                "at": _utcnow(), "job": self.job.name,
                "root_pid": self.root_pid, "lease_owner": lease_owner,
                "process_tree": tree,
                "known_process_identities": [self.identities[pid]
                                             for pid in sorted(self.identities)],
                "gpu_snapshot": gpu,
                "errors": [f"{type(exc).__name__}: {exc}"],
            }
            raise RuntimeGpuViolation(
                f"runtime GPU process/accounting validation failed: {exc}",
                snapshot=snapshot) from exc
        service_by_pid = {item["pid"]: item for item in self.manifest["allowed_processes"]}
        tree_pids = {record["pid"] for record in tree}
        unknown = [record for record in gpu["compute_app_records"]
                   if record["pid"] not in tree_pids and
                   record["pid"] not in service_by_pid]
        job_records = [record for record in gpu["compute_app_records"]
                       if record["pid"] in tree_pids]
        service_records = [record for record in gpu["compute_app_records"]
                           if record["pid"] in service_by_pid]
        job_used = sum(float(record["used_memory_mb"]) for record in job_records)
        service_used = sum(float(record["used_memory_mb"]) for record in service_records)
        service_reserved = sum(int(item["gpu_memory_budget_mb"])
                               for item in self.manifest["allowed_processes"])
        job_reserved = int(self.job.gpu_memory_cap_mb)
        device_total = float(gpu.get("total_mb") or 0.0)
        accounting = {
            "job_allocated_accounted_mb": job_used,
            "job_reserved_cap_mb": job_reserved,
            "service_allocated_accounted_mb": service_used,
            "service_reserved_cap_mb": service_reserved,
            "cumulative_allocated_accounted_mb": job_used + service_used,
            "cumulative_reserved_cap_mb": job_reserved + service_reserved,
            "device_total_mb": device_total,
        }
        record = {
            "schema": "round0005_runtime_gpu_telemetry.v2",
            "at": _utcnow(), "job": self.job.name,
            "root_pid": self.root_pid, "lease_owner": lease_owner,
            "process_tree": tree,
            "known_process_identities": [self.identities[pid]
                                         for pid in sorted(self.identities)],
            "allowed_services": allowed,
            "gpu_snapshot": gpu, "unknown_gpu_processes": unknown,
            "job_gpu_processes": job_records,
            "service_gpu_processes": service_records,
            "memory_accounting": accounting,
        }
        errors = []
        if unknown:
            errors.append(f"unknown runtime GPU processes: {unknown}")
        if job_reserved <= 0:
            errors.append("canonical GPU job has no positive memory cap")
        if job_used > job_reserved:
            errors.append(
                f"job GPU memory {job_used} MiB exceeds cap {job_reserved} MiB")
        if device_total <= 0:
            errors.append("GPU total-memory telemetry is unavailable")
        elif accounting["cumulative_reserved_cap_mb"] > device_total:
            errors.append(
                "job plus service reserved memory caps exceed device capacity")
        if accounting["cumulative_allocated_accounted_mb"] > device_total:
            errors.append("cumulative allocated GPU memory exceeds device capacity")
        if errors:
            record["errors"] = errors
            raise RuntimeGpuViolation("; ".join(errors), snapshot=record)
        record["errors"] = []
        return record


class GpuLease:
    """Exclusive, atomic, crash-safe GPU lease. The lock is held by the OPEN
    FILE DESCRIPTION, so passing the fd to a child keeps it held past controller
    death. ``timeout=0`` fails fast, ``None`` blocks, ``N`` retries for N s."""
    def __init__(self, path: str = None, timeout: Optional[float] = 0,
                 controller_id: Optional[str] = None):
        self.path = path or _lease_path()
        self.timeout = timeout
        self.controller_id = controller_id or f"ctl-{uuid.uuid4().hex[:8]}"
        self._fd = None
        self._guard_fd = None
        self._token = None
        self._inode = None

    def acquire(self):
        self.path = os.path.abspath(self.path)
        self._guard_fd, _parent = _open_parent_directory_no_symlinks(self.path)
        deadline = None if self.timeout is None else time.time() + self.timeout
        while True:
            try:
                # Locking the existing parent directory prevents a cooperating
                # contender from bypassing serialization by unlinking/recreating
                # the leaf while the original inode is held.
                fcntl.flock(self._guard_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._fd = os.open(
                    os.path.basename(self.path),
                    os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW,
                    0o600, dir_fd=self._guard_fd)
                state = os.fstat(self._fd)
                if not stat.S_ISREG(state.st_mode) or state.st_nlink != 1:
                    raise RuntimeError("GPU lease path is not one linked regular file")
                _ofd_set_lock(self._fd, fcntl.F_WRLCK)
                self._token = uuid.uuid4().hex
                payload = {
                    "schema": "round0005_gpu_lease.v2",
                    "controller_id": self.controller_id,
                    "controller_pid": os.getpid(),
                    "controller_starttime_ticks": _proc_starttime_ticks(os.getpid()),
                    "token": self._token,
                    "acquired_at": _utcnow(),
                }
                os.ftruncate(self._fd, 0)
                os.write(self._fd, json.dumps(
                    payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
                os.fsync(self._fd)
                self._inode = (state.st_dev, state.st_ino)
                os.set_inheritable(self._fd, True)
                os.set_inheritable(self._guard_fd, True)
                _OWNED_LEASE_FDS[self._fd] = self._token
                self.verify_current()
                return self
            except BlockingIOError as exc:
                if self._token is not None:
                    try:
                        self.release()
                    except BaseException as cleanup_exc:
                        try:
                            exc.add_note(
                                f"GPU lease verification cleanup also failed: "
                                f"{cleanup_exc!r}")
                        except AttributeError:
                            pass
                    raise
                if self._fd is not None:
                    contended_fd = self._fd
                    self._fd = None
                    _OWNED_LEASE_FDS.pop(contended_fd, None)
                    try:
                        os.close(contended_fd)
                    except BaseException as exc:
                        try:
                            self.release()
                        except BaseException as cleanup_exc:
                            try:
                                exc.add_note(
                                    f"GPU lease guard cleanup also failed: {cleanup_exc!r}")
                            except AttributeError:
                                pass
                        raise
                if deadline is not None and time.time() >= deadline:
                    held = "parent-directory serialization guard"
                    probe = None
                    try:
                        probe = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW)
                        held = os.pread(probe, 512, 0).decode(errors="replace")
                    except OSError:
                        pass
                    finally:
                        if probe is not None:
                            try:
                                os.close(probe)
                            except OSError:
                                pass
                    failure = RuntimeError(
                        f"GPU lease held by [{held}]; not launching (P0-D).")
                    try:
                        self.release()
                    except BaseException as cleanup_exc:
                        failure.add_note(
                            f"GPU lease timeout cleanup also failed: {cleanup_exc!r}")
                    raise failure
                time.sleep(0.1)
            except BaseException as exc:
                # Integrity failures and verify_current races are just as
                # important as lock-contention errors: no descriptor or lock
                # may survive a failed acquisition attempt in this process.
                try:
                    self.release()
                except BaseException as cleanup_exc:
                    try:
                        exc.add_note(
                            f"GPU lease acquisition cleanup also failed: {cleanup_exc!r}")
                    except AttributeError:
                        pass
                raise

    def fileno(self):
        return self._fd

    def pass_fds(self) -> tuple[int, ...]:
        return tuple(fd for fd in (self._fd, self._guard_fd) if fd is not None)

    @property
    def token(self) -> str:
        if self._token is None:
            raise RuntimeError("GPU lease is not acquired")
        return self._token

    def verify_current(self) -> dict:
        if self._fd is None or self._token is None:
            raise RuntimeError("GPU lease is not acquired")
        payload = _prove_fd_owns_ofd_lock(self._fd, self.path)
        if payload["token"] != self._token:
            raise RuntimeError("GPU lease owner token changed")
        if self._inode != (os.fstat(self._fd).st_dev, os.fstat(self._fd).st_ino):
            raise RuntimeError("GPU lease inode changed")
        return payload

    def release(self):
        fd, guard_fd = self._fd, self._guard_fd
        self._fd = None
        self._guard_fd = None
        self._token = None
        self._inode = None
        first_error = None
        if fd is not None:
            _OWNED_LEASE_FDS.pop(fd, None)
            try:
                _ofd_set_lock(fd, fcntl.F_UNLCK)
            except BaseException as exc:
                first_error = exc
            finally:
                try:
                    os.close(fd)
                except BaseException as exc:
                    if first_error is None:
                        first_error = exc
        if guard_fd is not None:
            try:
                fcntl.flock(guard_fd, fcntl.LOCK_UN)
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
            finally:
                try:
                    os.close(guard_fd)
                except BaseException as exc:
                    if first_error is None:
                        first_error = exc
        if first_error is not None:
            raise first_error

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *exc):
        self.release()
        return False


def _round0015_terminal_identity(summary: dict) -> dict:
    """Bind the terminal facts that exist before the lease can be released."""
    body = {
        "schema": "round0015-controller-terminal-identity-v1",
        "controller_id": summary["controller_id"],
        "controller_pid": summary["controller_pid"],
        "controller_starttime_ticks": summary["controller_starttime_ticks"],
        "queue_manifest_path": summary["queue_manifest_path"],
        "queue_manifest_sha256": summary["queue_manifest_sha256"],
        "queue_release_sha": summary["queue_release_sha"],
        "started": summary["started"],
        "finished": summary["finished"],
        "terminal_verdict": summary["terminal_verdict"],
        "stop_reason": summary.get("stop_reason"),
        "required_jobs": summary["required_jobs"],
        "completed_jobs": summary["completed_jobs"],
        "gpu_elapsed_s": summary["gpu_elapsed_s"],
    }
    return {**body, "identity_sha256": sha256_bytes(canonical_json(body))}


def _round0016_terminal_identity(summary: dict) -> dict:
    """Reuse the exact terminal fields with a Round-0016 target seal."""
    prior = _round0015_terminal_identity(summary)
    body = {key: value for key, value in prior.items()
            if key != "identity_sha256"}
    body["schema"] = "round0016-controller-terminal-identity-v1"
    return {**body, "identity_sha256": sha256_bytes(canonical_json(body))}


def _round0017_terminal_identity(summary: dict) -> dict:
    """Reuse the exact terminal fields with a Round-0017 target seal."""
    prior = _round0015_terminal_identity(summary)
    body = {key: value for key, value in prior.items()
            if key != "identity_sha256"}
    body["schema"] = "round0017-controller-terminal-identity-v1"
    return {**body, "identity_sha256": sha256_bytes(canonical_json(body))}


def _round0015_parent_guard_is_owned(path: str) -> dict:
    """Prove a separate parent open description conflicts before release."""
    probe, parent = _open_parent_directory_no_symlinks(path)
    try:
        try:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as exc:
            if isinstance(exc, BlockingIOError) or getattr(exc, "errno", None) in {11, 13}:
                return {"parent": parent, "independent_probe_conflicted": True}
            raise
        else:
            fcntl.flock(probe, fcntl.LOCK_UN)
            raise RuntimeError("Round 0015 parent serialization guard is not owned")
    finally:
        os.close(probe)


def _round0015_prove_released(path: str) -> dict:
    """Independently acquire both exact locks nonblocking after close."""
    guard_fd, parent = _open_parent_directory_no_symlinks(path)
    leaf_fd = None
    try:
        fcntl.flock(guard_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        leaf_fd = os.open(
            os.path.basename(path),
            os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=guard_fd)
        _ofd_set_lock(leaf_fd, fcntl.F_WRLCK)
        state = os.fstat(leaf_fd)
        proof = {
            "proved_at": _utcnow(),
            "parent_path": parent,
            "parent_serialization_lock_nonblocking_acquired": True,
            "lease_file_ofd_lock_nonblocking_acquired": True,
            "lease_device": int(state.st_dev),
            "lease_inode": int(state.st_ino),
            "lease_links": int(state.st_nlink),
        }
        _ofd_set_lock(leaf_fd, fcntl.F_UNLCK)
        os.close(leaf_fd); leaf_fd = None
        fcntl.flock(guard_fd, fcntl.LOCK_UN)
        return proof
    finally:
        if leaf_fd is not None:
            try:
                _ofd_set_lock(leaf_fd, fcntl.F_UNLCK)
            except BaseException:
                pass
            try:
                os.close(leaf_fd)
            except BaseException:
                pass
        try:
            fcntl.flock(guard_fd, fcntl.LOCK_UN)
        except BaseException:
            pass
        os.close(guard_fd)


def _round0015_release_with_receipt(
        lease: GpuLease, *, receipt_path: str,
        terminal_identity: dict, _fixture_only: bool = False,
        _round0016: bool = False, _round0017: bool = False) -> dict:
    """Neutralize, release, and prove both locks for Rounds 0015--0017."""
    round_id = "0017" if _round0017 else ("0016" if _round0016 else "0015")
    round_label = f"Round {round_id}"
    expected = f"/data/latent-basemap/runs/round-{round_id}/queue/lease-release.json"
    if (os.path.realpath(receipt_path) != receipt_path or
            (not _fixture_only and receipt_path != expected) or
            os.path.lexists(receipt_path)):
        raise RuntimeError(f"{round_label} lease-release receipt destination changed")
    if _fixture_only and os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError(f"{round_label} lease-release fixture requires CUDA hidden")
    acquisition = lease.verify_current()
    _round0015_parent_guard_is_owned(lease.path)
    before = os.pread(lease._fd, 4096, 0)
    before_state = os.fstat(lease._fd)
    if json.loads(before.decode("utf-8")) != acquisition:
        raise RuntimeError(f"{round_label} lease payload differs from acquisition identity")
    released_at = _utcnow()
    neutral = {
        "schema": f"round{round_id}_gpu_lease_released.v1",
        "state": "released",
        "controller_id": acquisition["controller_id"],
        "controller_pid": acquisition["controller_pid"],
        "controller_starttime_ticks": acquisition["controller_starttime_ticks"],
        "token": acquisition["token"],
        "acquired_at": acquisition["acquired_at"],
        "released_at": released_at,
        "terminal_identity_sha256": terminal_identity["identity_sha256"],
    }
    neutral_bytes = canonical_json(neutral)
    release_error = None
    proof = None
    try:
        # Both the leaf OFD lock and parent flock are still owned here.
        lease.verify_current()
        _round0015_parent_guard_is_owned(lease.path)
        os.ftruncate(lease._fd, 0)
        written = os.pwrite(lease._fd, neutral_bytes, 0)
        if written != len(neutral_bytes):
            raise RuntimeError(f"{round_label} neutral lease payload write was short")
        os.fsync(lease._fd)
        if os.pread(lease._fd, len(neutral_bytes) + 1, 0) != neutral_bytes:
            raise RuntimeError(f"{round_label} neutral lease payload did not persist")
    except BaseException as exc:
        release_error = f"{type(exc).__name__}: {exc}"
    try:
        lease.release()
    except BaseException as exc:
        detail = f"{type(exc).__name__}: {exc}"
        release_error = detail if release_error is None else f"{release_error}; {detail}"
    try:
        proof = _round0015_prove_released(lease.path)
    except BaseException as exc:
        detail = f"{type(exc).__name__}: {exc}"
        release_error = detail if release_error is None else f"{release_error}; {detail}"
    after = open(lease.path, "rb").read()
    body = {
        "schema": f"round{round_id}-terminal-lease-release-receipt-v1",
        "status": "passed" if release_error is None else "failed",
        "lease_path": lease.path,
        "lease_device": int(before_state.st_dev),
        "lease_inode": int(before_state.st_ino),
        "payload_before_sha256": sha256_bytes(before),
        "payload_after_sha256": sha256_bytes(after),
        "payload_before": acquisition,
        "payload_after": neutral if after == neutral_bytes else None,
        "neutral_payload_sha256": sha256_bytes(neutral_bytes),
        "controller_acquisition_identity": acquisition,
        "terminal_identity": terminal_identity,
        "released_at": released_at,
        "lock_proof": proof,
        "release_error": release_error,
    }
    receipt = {**body, "identity_sha256": sha256_bytes(canonical_json(body))}
    atomic_write_new_json(receipt_path, receipt, immutable=True)
    if release_error is not None:
        raise RuntimeError(
            f"{round_label} terminal lease release proof failed: {release_error}")
    return receipt


DEFAULT_SERVICE_MARKERS = ("ls-serve", "moonshine-web")   # known always-on viewers


def known_service_pids(markers=DEFAULT_SERVICE_MARKERS) -> list:
    """Compute PIDs whose /proc cmdline matches a KNOWN background-service marker
    (P0-5). Unlike snapshotting every observed PID, this allow-lists only named
    services by identity — an unknown training process is NOT auto-tolerated."""
    out = []
    for pid in gpu_snapshot()["compute_pids"]:
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmd = f.read().replace(b"\x00", b" ").decode(errors="replace")
            if any(m in cmd for m in markers):
                out.append(int(pid))
        except Exception:
            pass
    return out


def check_co_tenants(required_free_gb: float, allowed_pids=(), wait_s: float = 0) -> dict:
    """Enforce the co-tenant policy before a GPU launch. Fails (or waits up to
    ``wait_s``) if an unknown compute PID is present or free VRAM < requirement.
    ``allowed_pids`` are tolerated background services (e.g. a viewer)."""
    deadline = time.time() + wait_s
    allowed = set(int(p) for p in allowed_pids)
    while True:
        snap = gpu_snapshot()
        unknown = [p for p in snap["compute_pids"] if p not in allowed]
        free_gb = (snap["free_mb"] or 0) / 1024.0
        ok = (not unknown) and free_gb >= required_free_gb
        if ok or time.time() >= deadline:
            if not ok:
                raise RuntimeError(
                    f"co-tenant policy blocked launch: unknown GPU PIDs {unknown}, "
                    f"free {free_gb:.1f} GB < required {required_free_gb} GB (P0-D). snapshot={snap['gpu']}")
            return snap
        time.sleep(2.0)


@dataclasses.dataclass
class Job:
    name: str
    argv: list
    outputs: list                    # declared output paths; ALL must exist for success
    done_marker: str                 # controller-written completion record (.done.json)
    deps: list = dataclasses.field(default_factory=list)   # names that must be done first
    cwd: Optional[str] = None
    log: Optional[str] = None
    manifest: Optional[str] = None
    required_free_gb: float = 0.0
    continue_on_failure: bool = False
    input_paths: list = dataclasses.field(default_factory=list)   # P1: files whose
    expected_inputs: list = dataclasses.field(default_factory=list)
    # content binds the job identity (config, scorer/trainer code, input artifacts).
    # A change to any of them (or the repo commit) invalidates a stale done marker.
    certifying: bool = True   # S1: a certifying job MUST declare outputs; outputs=[]
    # is only allowed for an explicitly non-certifying job (certifying=False).
    predicted_wall_s: float = 0.0   # S2: predicted wall time; >CANARY_REQUIRED_WALL_S
    p90_wall_s: float = 0.0
    # forces a passing canary dependency before the run is admitted.
    canary_dep: Optional[str] = None   # S2: name of the perf-canary job this long
    # run depends on. Must also appear in `deps` so a sub-floor canary blocks it.
    require_passing_verdict: Optional[str] = None   # L0.3: path to a verdict JSON
    # that must parse to {"passed": true} before this job launches. Binds the
    # canary→train edge by CONTENT — a `touch`-only job named `perf_canary` cannot
    # release training because it never writes a passing verdict.
    scientific_rows: int = 0   # Round 0005: >=8M requires a pre-gate signed gate.
    performance_gate_path: Optional[str] = None
    scale_policy: Optional[dict] = None
    gpu_memory_cap_mb: int = 0
    node_policy: Optional[dict] = None

    def manifest_contract(self) -> dict:
        """Serialize every runtime-relevant field for exact manifest comparison."""
        if self.node_policy is not None:
            return {
                "argv": list(self.argv), "cwd": self.cwd, "deps": list(self.deps),
                "done_marker": self.done_marker, "expected_inputs": list(self.expected_inputs),
                "id": self.name, "inputs": list(self.input_paths), "log": self.log,
                "manifest": self.manifest, "node_policy": dict(self.node_policy),
                "outputs": list(self.outputs), "p90_wall_s": self.p90_wall_s,
                "predicted_wall_s": self.predicted_wall_s,
            }
        return {
            "argv": list(self.argv), "canary_dep": self.canary_dep,
            "certifying": self.certifying, "continue_on_failure": self.continue_on_failure,
            "cwd": self.cwd, "deps": list(self.deps), "done_marker": self.done_marker,
            "expected_inputs": list(self.expected_inputs), "id": self.name,
            "inputs": list(self.input_paths), "log": self.log, "manifest": self.manifest,
            "outputs": list(self.outputs), "p90_wall_s": self.p90_wall_s,
            "performance_gate_path": self.performance_gate_path,
            "predicted_wall_s": self.predicted_wall_s,
            "require_passing_verdict": self.require_passing_verdict,
            "required_free_gb": self.required_free_gb,
            "scale_policy": self.scale_policy, "scientific_rows": self.scientific_rows,
        }


CANARY_REQUIRED_WALL_S = 600.0   # S2: >10 predicted minutes ⇒ canary is mandatory
SCALE_PERFORMANCE_ROWS = 8_000_000


def require_scale_performance_gate(report_path: Optional[str], *,
                                   scientific_rows: int,
                                   scale_policy: Optional[dict] = None,
                                   row_derivation: Optional[dict] = None,
                                   release_sha: Optional[str] = None) -> Optional[dict]:
    """Use the one strict scale validator shared by launchers and controller."""
    from experiments.round0005_performance_gate import (
        require_scale_performance_gate as require_strict_scale_gate,
    )

    return require_strict_scale_gate(
        report_path, scientific_rows=scientific_rows,
        row_derivation=row_derivation, release_sha=release_sha,
        scale_policy=scale_policy)


def _run_jobs_legacy(jobs: list, controller_id: Optional[str] = None, allowed_pids=(),
                     summary_path: Optional[str] = None, admission=None,
                     launch_edge_hook=None) -> dict:
    cid = controller_id or f"ctl-{uuid.uuid4().hex[:8]}"
    launch_checkout = _git_state()
    summary = {"controller_id": cid, "controller_pid": os.getpid(), "started": _utcnow(),
               "launch_checkout": launch_checkout, "jobs": []}
    if admission is not None:
        summary["queue_manifest_path"] = admission.manifest_path
        summary["queue_manifest_sha256"] = admission.manifest_sha256
        summary["queue_release_sha"] = admission.manifest["release_sha"]
        summary["expected_input_signatures"] = admission.expected_inputs
        summary["initial_observed_input_signatures"] = admission.initial_inputs
    done = set()
    with GpuLease(controller_id=cid, timeout=0) as lease:
        for job in jobs:
            rec = {"name": job.name}
            # S1: a certifying job MUST declare outputs (exit-0 alone can't certify);
            # and every declared input MUST exist (fail on missing input).
            if job.certifying and not job.outputs:
                rec["status"] = "config_error:certifying_job_without_outputs"
                summary["jobs"].append(rec)
                if not job.continue_on_failure:
                    summary["stop_reason"] = f"{job.name}: certifying job declares no outputs (S1)"; break
                continue
            missing_inputs = [p for p in (job.input_paths or []) if not os.path.exists(p)]
            if missing_inputs:
                rec["status"] = f"missing_inputs:{missing_inputs}"
                summary["jobs"].append(rec)
                if not job.continue_on_failure:
                    summary["stop_reason"] = f"{job.name}: missing declared inputs {missing_inputs} (S1)"; break
                continue
            # S2: a certifying run predicted to exceed 10 minutes MUST depend on a
            # passing performance canary in this same batch (critique #4: nothing
            # made the canary a mandatory dependency). A sub-floor canary exits
            # non-zero → not in `done` → the deps check below blocks this run.
            if (job.certifying and job.predicted_wall_s > CANARY_REQUIRED_WALL_S
                    and (not job.canary_dep or job.canary_dep not in (job.deps or []))):
                rec["status"] = "config_error:long_run_without_canary_dep"
                rec["predicted_wall_s"] = job.predicted_wall_s
                summary["jobs"].append(rec)
                if not job.continue_on_failure:
                    summary["stop_reason"] = (f"{job.name}: predicted {job.predicted_wall_s:.0f}s "
                                              f"> {CANARY_REQUIRED_WALL_S:.0f}s needs a canary_dep "
                                              f"in deps (S2)"); break
                continue
            spec_digest = _job_spec_digest(job)
            # idempotency: skip only if the done record matches THIS job's spec
            # digest AND every output still matches its recorded signature (P0-5:
            # a changed argv/config or a mutated output invalidates the skip).
            if os.path.exists(job.done_marker):
                try:
                    m = json.load(open(job.done_marker))
                    sigs_ok = all(_output_sig(o) == (m.get("output_sigs") or {}).get(o)
                                  and _output_sig(o) is not None for o in job.outputs)
                    if (m.get("status") == "ok" and m.get("spec_digest") == spec_digest
                            and sigs_ok):
                        rec["status"] = "skipped_done"; done.add(job.name)
                        summary["jobs"].append(rec); continue
                except Exception:
                    pass  # partial/corrupt marker → re-run
            # dependencies
            missing = [d for d in job.deps if d not in done]
            if missing:
                rec["status"] = f"blocked_deps:{missing}"
                summary["jobs"].append(rec)
                if not job.continue_on_failure:
                    summary["stop_reason"] = f"unsatisfied deps for {job.name}"; break
                continue
            # An 8M/30M launch is impossible unless its performance certificate
            # already existed and was part of the signed input set before gate.
            try:
                if job.performance_gate_path and job.performance_gate_path not in job.input_paths:
                    raise RuntimeError("performance gate path is not a declared job input")
                require_scale_performance_gate(
                    job.performance_gate_path, scientific_rows=int(job.scientific_rows))
            except Exception as e:
                rec["status"] = "blocked:performance_gate"
                rec["error"] = str(e)
                summary["jobs"].append(rec)
                if not job.continue_on_failure:
                    summary["stop_reason"] = f"{job.name}: {e}"; break
                continue
            # L0.3: content-based canary gate — the referenced verdict JSON must
            # exist and parse to passed=true. A job named `perf_canary` that only
            # `touch`es a file writes no such verdict, so it cannot release a train.
            rpv = getattr(job, "require_passing_verdict", None)
            if rpv:
                if admission is not None and rpv not in job.input_paths:
                    rec["status"] = "blocked:verdict_not_pre_gate_signed"
                    rec["verdict_path"] = rpv
                    summary["jobs"].append(rec)
                    if not job.continue_on_failure:
                        summary["stop_reason"] = (
                            f"{job.name}: required verdict is not a signed pre-gate input"); break
                    continue
                v = None
                try:
                    v = json.load(open(rpv))
                except Exception as e:
                    v = {"_load_error": repr(e)}
                if not (isinstance(v, dict) and v.get("passed") is True):
                    rec["status"] = "blocked:verdict_not_passing"
                    rec["verdict_path"] = rpv
                    summary["jobs"].append(rec)
                    if not job.continue_on_failure:
                        summary["stop_reason"] = (f"{job.name}: required verdict {rpv} is not "
                                                  f"passed=true (L0.3)"); break
                    continue
            # co-tenant policy — enforced only for jobs that declare a GPU need
            # (required_free_gb>0); required_free_gb==0 marks a CPU job.
            try:
                snap_pre = (check_co_tenants(job.required_free_gb, allowed_pids)
                            if job.required_free_gb > 0 else gpu_snapshot())
            except RuntimeError as e:
                rec["status"] = "co_tenant_block"; rec["error"] = str(e)
                summary["jobs"].append(rec)
                if not job.continue_on_failure:
                    summary["stop_reason"] = str(e); break
                continue
            # Gate-check is intentionally here: after dependency/idempotency and
            # co-tenant work, immediately before the child launch.  A second
            # local prelaunch hash closes the gate-response -> Popen mutation
            # window without learning a new baseline.
            if admission is not None:
                try:
                    rec["gate_receipt"] = admission.boundary(job.name)
                except Exception as e:
                    rec["status"] = "boundary_rejected"
                    rec["error"] = str(e)
                    summary["jobs"].append(rec)
                    summary["stop_reason"] = f"{job.name}: queue boundary rejected: {e}"
                    break
            rec["gpu_pre"] = snap_pre
            input_sigs = {p: _output_sig(p) for p in sorted(job.input_paths or [])}
            if job.manifest:
                _atomic_write_json(job.manifest, {"controller_id": cid, "job": job.name,
                                                  "argv": job.argv, "gpu_pre": snap_pre,
                                                  "status": "running", "started": _utcnow(),
                                                  "launch_checkout": launch_checkout,
                                                  "queue_manifest_sha256": (admission.manifest_sha256
                                                                            if admission else None),
                                                  "input_signatures": input_sigs})
            # snapshot pre-existing output signatures — an exit-0 no-op that
            # leaves a stale output unchanged must NOT be certified (P0-5).
            pre_sigs = {o: _output_sig(o) for o in job.outputs}
            logf = open(job.log, "x") if job.log else None
            t0 = time.time()
            # child in its own process group. pass_fds keeps the inherited lease
            # fd open (lock survives controller death); close_fds stays default
            # True so no other fds leak (the old close_fds=False warned every run).
            # P1: pass the inherited lease fd number so the child's
            # require_active_lease() can PROVE ownership (inode match), not merely
            # observe the global lock is held.
            child_env = dict(os.environ)
            if admission is not None:
                child_env.update({str(k): str(v) for k, v in
                                  admission.manifest["cache_environment"].items()})
            if lease.fileno() is not None:
                child_env["BASEMAP_GPU_LEASE_FD"] = str(lease.fileno())
            try:
                # Keep these statements adjacent. All hashing, receipt/manifest/log
                # writes, output snapshots, and environment construction are above.
                # The comparison has no side effect and the very next operation is
                # the operating-system child launch.
                if admission is not None:
                    admission.final_expected_input_comparison(job.name, after_comparison_hook=(lambda _comparison: launch_edge_hook(job)) if launch_edge_hook is not None else None)
                p = subprocess.Popen(job.argv, cwd=job.cwd, stdout=logf, stderr=subprocess.STDOUT,
                                     start_new_session=True, env=child_env,
                                     pass_fds=(lease.fileno(),) if lease.fileno() is not None else ())
            except Exception as e:
                if logf:
                    logf.close()
                recorder = (getattr(admission, "record_final_comparison_failure", None)
                            if admission is not None else None)
                if recorder is not None:
                    try:
                        rec["launch_input_failure_receipt"] = recorder(job.name, e)
                    except Exception as receipt_error:
                        rec["launch_input_failure_receipt_error"] = repr(receipt_error)
                rec["status"] = "launch_edge_rejected"
                rec["error"] = str(e)
                summary["jobs"].append(rec)
                summary["stop_reason"] = f"{job.name}: launch edge rejected: {e}"
                break
            rec["child_pid"] = p.pid; rec["pgid"] = os.getpgid(p.pid)
            rc = p.wait()
            if logf:
                logf.close()
            post_sigs = {o: _output_sig(o) for o in job.outputs}
            outs_ok = all(post_sigs[o] is not None for o in job.outputs)
            # every declared output must be freshly produced or changed vs pre-run
            fresh_ok = all(post_sigs[o] is not None and post_sigs[o] != pre_sigs[o]
                           for o in job.outputs)
            stale = outs_ok and not fresh_ok
            success = (rc == 0) and outs_ok and fresh_ok
            rec["status"] = ("ok" if success else
                             ("exit_%d" % rc if rc != 0 else
                              "stale_outputs" if stale else "missing_outputs"))
            rec["seconds"] = round(time.time() - t0, 1)
            rec["outputs_present"] = outs_ok
            snap_post = gpu_snapshot()
            final = {"controller_id": cid, "job": job.name, "argv": job.argv,
                     "status": rec["status"], "exit_code": rc, "seconds": rec["seconds"],
                     "spec_digest": spec_digest, "output_sigs": post_sigs,
                     "input_signatures": input_sigs,
                     "launch_checkout": launch_checkout,
                     "queue_manifest_sha256": (admission.manifest_sha256 if admission else None),
                     "gpu_pre": snap_pre, "gpu_post": snap_post, "finished": _utcnow()}
            if job.manifest:
                _atomic_write_json(job.manifest, final)
            if success:
                # completion record is controller-written, AFTER validation, and
                # bound to the spec digest + output signatures.
                _atomic_write_json(job.done_marker, {"status": "ok", "job": job.name,
                                                     "finished": _utcnow(), "exit_code": 0,
                                                     "spec_digest": spec_digest,
                                                     "input_signatures": input_sigs,
                                                     "launch_checkout": launch_checkout,
                                                     "queue_manifest_sha256": (admission.manifest_sha256
                                                                               if admission else None),
                                                     "output_sigs": post_sigs})
                done.add(job.name)
            summary["jobs"].append(rec)
            if not success and not job.continue_on_failure:
                summary["stop_reason"] = f"{job.name} failed ({rec['status']}) — chain stopped"; break
    summary.setdefault("stop_reason", "completed")
    summary["finished"] = _utcnow()
    if summary_path:
        _atomic_write_json(summary_path, summary)
    return summary


class _CheckpointWriter:
    """Append-only immutable controller event journal for one admitted queue."""

    def __init__(self, root: str, *, controller_claim_sha256: str | None = None):
        self.root = os.path.realpath(root)
        if (not self.root.startswith("/data/") or not os.path.isdir(self.root) or
                os.path.islink(root)):
            raise RuntimeError("controller checkpoint root must be a regular /data directory")
        if os.listdir(self.root):
            raise FileExistsError(f"controller checkpoint root is not fresh: {self.root}")
        self.sequence = 0
        self.controller_claim_sha256 = controller_claim_sha256

    def write(self, event: str, payload: dict) -> str:
        record = {"schema": "round0005_controller_checkpoint.v1",
                  "sequence": self.sequence, "event": event, "at": _utcnow(),
                  **({"controller_claim_sha256": self.controller_claim_sha256}
                     if self.controller_claim_sha256 is not None else {}),
                  **payload}
        destination = os.path.join(self.root, f"{self.sequence:06d}-{event}.json")
        atomic_write_new_json(destination, record, immutable=True)
        self.sequence += 1
        return destination


def _deadline(value: str) -> datetime.datetime:
    parsed = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise RuntimeError("queue deadline has no UTC offset")
    return parsed.astimezone(datetime.timezone.utc)


def _require_budget_fit(manifest: dict, job: Job, *, gpu_elapsed_s: float,
                        phase: str) -> dict:
    now = datetime.datetime.now(datetime.timezone.utc)
    deadline = _deadline(manifest["deadline_utc"])
    remaining_deadline_s = (deadline - now).total_seconds()
    remaining_cap_s = float(manifest["gpu_hours_cap"]) * 3600.0 - gpu_elapsed_s
    registered_s = max(float(job.predicted_wall_s), float(job.p90_wall_s))
    required_s = registered_s * 1.15
    if remaining_deadline_s < required_s:
        raise RuntimeError(
            f"deadline blocks {job.name} during {phase}: need p90/predicted+15% "
            f"{required_s:.3f}s, have {remaining_deadline_s:.3f}s")
    if job.required_free_gb > 0 and remaining_cap_s < required_s:
        raise RuntimeError(
            f"cumulative GPU-hour cap blocks {job.name} during {phase}: need "
            f"{required_s:.3f}s, have {remaining_cap_s:.3f}s")
    return {"phase": phase, "registered_wall_s": registered_s,
            "required_with_margin_s": required_s,
            "remaining_deadline_s": remaining_deadline_s,
            "remaining_gpu_cap_s": remaining_cap_s, "gpu_elapsed_s": gpu_elapsed_s}


def _fresh_job_paths(job: Job) -> list[str]:
    paths = [*job.outputs, job.done_marker, job.log, job.manifest]
    present = [path for path in paths if os.path.lexists(path)]
    if present:
        raise FileExistsError(
            f"job {job.name} refuses pre-existing scientific/control paths: {present}")
    return paths


def _strict_output_signature(path: str) -> Optional[dict]:
    if not os.path.lexists(path):
        return None
    return expected_input_signature(path)


def _immutable_runtime_signature(path: str) -> dict:
    """Content plus filesystem identity for completed controller evidence."""
    canonical = os.path.realpath(path)
    signature = expected_input_signature(canonical)

    def identity(member: str) -> dict:
        status = os.lstat(member)
        if stat.S_ISLNK(status.st_mode):
            raise RuntimeError(f"cumulative evidence contains a symlink: {member}")
        if stat.S_ISREG(status.st_mode):
            kind = "file"
            if status.st_nlink != 1:
                raise RuntimeError(f"cumulative evidence is hard linked: {member}")
        elif stat.S_ISDIR(status.st_mode):
            kind = "directory"
        else:
            raise RuntimeError(f"cumulative evidence has unsupported kind: {member}")
        return {
            "kind": kind, "device": int(status.st_dev), "inode": int(status.st_ino),
            "mode": int(status.st_mode), "links": int(status.st_nlink),
            "bytes": int(status.st_size), "mtime_ns": int(status.st_mtime_ns),
            "ctime_ns": int(status.st_ctime_ns),
        }

    identities = {".": identity(canonical)}
    if signature["kind"] == "directory":
        for member in signature["members"]:
            identities[member["relative_path"]] = identity(
                os.path.join(canonical, member["relative_path"]))
    return {"signature": signature, "filesystem_identities": identities}


def _flat_controller_evidence(root: str) -> list[str]:
    """List a flat evidence directory while rejecting every surprising member."""
    paths = []
    with os.scandir(root) as entries:
        for entry in sorted(entries, key=lambda value: value.name):
            status = os.lstat(entry.path)
            if (entry.is_symlink() or not stat.S_ISREG(status.st_mode) or
                    status.st_nlink != 1):
                raise RuntimeError(
                    f"controller evidence directory contains unsupported member: {entry.path}")
            paths.append(os.path.realpath(entry.path))
    return paths


def _verify_cumulative_registry(registry: dict[str, dict]) -> dict[str, dict]:
    observed = {}
    for path in sorted(registry):
        if not os.path.lexists(path):
            raise RuntimeError(f"completed predecessor evidence was unlinked: {path}")
        observed[path] = _immutable_runtime_signature(path)
    if observed != registry:
        raise RuntimeError(
            "completed predecessor output/log/done/control evidence changed")
    return observed


def _extend_cumulative_registry(registry: dict[str, dict], *, job: Job,
                                checkpoint_root: str,
                                gate_receipts_root: str) -> dict[str, dict]:
    _verify_cumulative_registry(registry)
    paths = [*job.outputs, job.done_marker, job.log, job.manifest]
    paths.extend(_flat_controller_evidence(checkpoint_root))
    paths.extend(_flat_controller_evidence(gate_receipts_root))
    for path in sorted(set(os.path.realpath(value) for value in paths)):
        current = _immutable_runtime_signature(path)
        if path in registry and registry[path] != current:
            raise RuntimeError(f"cumulative evidence changed while extending: {path}")
        registry[path] = current
    return _verify_cumulative_registry(registry)


def _cumulative_runtime_state(lease: GpuLease, registry: dict[str, dict]) -> dict:
    return {"lease": lease.verify_current(),
            "cumulative_registry": _verify_cumulative_registry(registry)}


def _terminate_process_group(process: subprocess.Popen, *, grace_s: float = 5.0) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_s)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_s)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"could not terminate process group {process.pid}") from exc


def _kill_process_group_id(pgid: int) -> None:
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _child_parent_death_setup(expected_parent: int) -> None:
    """Arm parent-death handling inside a freshly execed, single-threaded shim."""
    libc = ctypes.CDLL(None, use_errno=True)
    PR_SET_PDEATHSIG = 1
    if libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    if os.getppid() != expected_parent:
        os.kill(os.getpid(), signal.SIGKILL)


def _parent_death_exec_argv(argv: list[str], *, expected_parent: int) -> list[str]:
    if (not isinstance(argv, list) or len(argv) < 2 or
            not os.path.isabs(argv[0]) or not os.path.isfile(argv[0])):
        raise RuntimeError("parent-death exec requires an absolute Python job argv")
    return [
        argv[0], "-m", "basemap.run_controller",
        "--round0005-internal-parent-death-exec", str(expected_parent), "--",
        *argv,
    ]


class _InheritedLeaseView:
    """Read-only lease verifier used by the separately execed watchdog."""

    def __init__(self, *, fd: int, path: str, token: str):
        self._fd = int(fd)
        self.path = path
        self._token = token

    def verify_current(self) -> dict:
        payload = _prove_fd_owns_ofd_lock(self._fd, self.path)
        if payload.get("token") != self._token:
            raise RuntimeError("watchdog inherited lease token changed")
        return payload


def _write_watchdog_emergency(root: str, *, job: Job, child_pid: int,
                              error: str, snapshot: dict | None) -> None:
    body = {
        "schema": "round0005_watchdog_emergency.v1", "at": _utcnow(),
        "job": job.name, "child_pid": child_pid, "pgid": child_pid,
        "error": error, "snapshot": snapshot,
    }
    body["identity_sha256"] = sha256_bytes(canonical_json(body))
    try:
        atomic_write_new_json(
            os.path.join(root, f"watchdog-{job.name}-{child_pid}.json"),
            body, immutable=True)
    except Exception:
        # The kill guarantee must not depend on evidence publication succeeding.
        pass


def _watchdog_loop(*, read_fd: int, result_fd: int, result_nonce: str,
                   controller_pid: int, checkpoint_root: str, manifest: dict,
                   job: Job, child_pid: int, lease: GpuLease,
                   deadline_epoch: float, runtime_deadline_epoch: float,
                   gpu_cap_deadline_epoch: float,
                   controller_claim_sha256: str) -> None:
    accountant = (RuntimeGpuAccountant(
        manifest=manifest, job=job, root_pid=child_pid, lease=lease)
        if job.node_policy and job.node_policy.get("gpu_required") else None)
    error = None
    snapshot = None
    result_channel = socket.socket(fileno=result_fd)
    try:
        while True:
            readable, _, _ = select.select([read_fd], [], [], 1.0)
            if readable:
                message = os.read(read_fd, 1)
                if message == b"D":
                    break
                if message == b"":
                    error = "controller process died; watchdog terminated process group"
                    break
            now = time.time()
            if now >= deadline_epoch:
                error = "watchdog enforced queue deadline after controller loss/stall"
                break
            if now >= runtime_deadline_epoch:
                error = "watchdog enforced registered p90 +15% runtime deadline"
                break
            if now >= gpu_cap_deadline_epoch:
                error = "watchdog enforced cumulative GPU-hour cap"
                break
            if accountant is not None:
                try:
                    snapshot = accountant.snapshot()
                except RuntimeGpuViolation as exc:
                    snapshot = exc.snapshot
                    error = f"watchdog GPU policy violation: {exc}"
                    break
                except Exception as exc:
                    error = f"watchdog GPU telemetry failed closed: {type(exc).__name__}: {exc}"
                    break
    except BaseException as exc:
        error = f"watchdog internal failure: {type(exc).__name__}: {exc}"
    finally:
        if error is not None:
            _kill_process_group_id(child_pid)
            _write_watchdog_emergency(
                checkpoint_root, job=job, child_pid=child_pid,
                error=error, snapshot=snapshot)
        verdict = {
            "schema": "round0005_watchdog_verdict.v1",
            "nonce": result_nonce, "watchdog_pid": os.getpid(),
            "controller_pid": controller_pid, "job": job.name,
            "controller_claim_sha256": controller_claim_sha256,
            "child_pid": child_pid,
            "status": ("clean" if error is None else "emergency"),
            "error": error, "snapshot": snapshot, "at": _utcnow(),
        }
        verdict["identity_sha256"] = sha256_bytes(canonical_json(verdict))
        try:
            result_channel.sendall(canonical_json(verdict))
        except Exception:
            pass
        result_channel.close()
        os.close(read_fd)


@dataclasses.dataclass
class _WatchdogHandle:
    process: subprocess.Popen
    pid: int
    control_fd: int
    result_channel: socket.socket
    nonce: str
    controller_pid: int
    job: str
    child_pid: int
    controller_claim_sha256: str


def _start_watchdog(*, read_fd: int, write_fd: int, checkpoint_root: str,
                    manifest: dict, job: Job, child_pid: int, lease: GpuLease,
                    deadline_epoch: float, runtime_deadline_epoch: float,
                    gpu_cap_deadline_epoch: float,
                    result_nonce: str,
                    controller_claim_sha256: str) -> _WatchdogHandle:
    result_parent, result_child = socket.socketpair(
        socket.AF_UNIX, socket.SOCK_SEQPACKET)
    config_parent, config_child = socket.socketpair(
        socket.AF_UNIX, socket.SOCK_SEQPACKET)
    result_parent.set_inheritable(False)
    result_child.set_inheritable(False)
    config_parent.set_inheritable(False)
    config_child.set_inheritable(False)
    controller_pid = os.getpid()
    config = {
        "schema": "round0005_watchdog_spawn.v1",
        "result_nonce": result_nonce, "controller_pid": controller_pid,
        "checkpoint_root": checkpoint_root,
        "manifest": {
            "allowed_processes": manifest["allowed_processes"],
            "environment_manifest": manifest["environment_manifest"],
            "child_environment": manifest["child_environment"],
        },
        "job": {
            "name": job.name, "node_policy": job.node_policy,
            "gpu_memory_cap_mb": job.gpu_memory_cap_mb,
        },
        "child_pid": child_pid,
        "lease_path": lease.path, "lease_token": lease.token,
        "deadline_epoch": deadline_epoch,
        "runtime_deadline_epoch": runtime_deadline_epoch,
        "gpu_cap_deadline_epoch": gpu_cap_deadline_epoch,
        "controller_claim_sha256": controller_claim_sha256,
    }
    command = [
        job.argv[0], "-m", "basemap.run_controller",
        "--round0005-internal-watchdog", str(config_child.fileno()),
        str(read_fd), str(result_child.fileno()), str(lease.fileno()),
    ]
    try:
        process = subprocess.Popen(
            command, cwd=manifest["repo_root"],
            env=dict(manifest["child_environment"]),
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, start_new_session=True,
            close_fds=True,
            pass_fds=(config_child.fileno(), read_fd, result_child.fileno(),
                      lease.fileno()))
        config_child.close()
        result_child.close()
        os.close(read_fd)
        config_parent.sendall(canonical_json(config))
        config_parent.close()
    except BaseException:
        for channel in (config_parent, config_child, result_parent, result_child):
            try:
                channel.close()
            except OSError:
                pass
        try:
            os.close(read_fd)
        except OSError:
            pass
        raise
    return _WatchdogHandle(
        process=process, pid=process.pid, control_fd=write_fd,
        result_channel=result_parent,
        nonce=result_nonce, controller_pid=controller_pid,
        job=job.name, child_pid=child_pid,
        controller_claim_sha256=controller_claim_sha256)


def _stop_watchdog(handle: _WatchdogHandle | None, *,
                   require_clean: bool = True) -> dict | None:
    if handle is None:
        return None
    try:
        try:
            os.write(handle.control_fd, b"D")
        except (BrokenPipeError, OSError):
            pass
        try:
            os.close(handle.control_fd)
        except OSError:
            pass
        handle.result_channel.settimeout(15.0)
        try:
            raw = handle.result_channel.recv(1 << 20)
            verdict = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            raise RuntimeError("watchdog omitted its authenticated result verdict") from exc
    finally:
        handle.result_channel.close()
    status = handle.process.wait(timeout=15.0)
    identity = verdict.get("identity_sha256") if isinstance(verdict, dict) else None
    body = dict(verdict) if isinstance(verdict, dict) else {}
    body.pop("identity_sha256", None)
    expected_fields = {
        "schema", "nonce", "watchdog_pid", "controller_pid", "job",
        "child_pid", "controller_claim_sha256", "status", "error", "snapshot",
        "at", "identity_sha256",
    }
    if (status != 0 or not isinstance(verdict, dict) or
            set(verdict) != expected_fields or
            verdict.get("schema") != "round0005_watchdog_verdict.v1" or
            verdict.get("nonce") != handle.nonce or
            verdict.get("watchdog_pid") != handle.pid or
            verdict.get("controller_pid") != handle.controller_pid or
            verdict.get("controller_claim_sha256") !=
            handle.controller_claim_sha256 or
            verdict.get("job") != handle.job or
            verdict.get("child_pid") != handle.child_pid or
            identity != sha256_bytes(canonical_json(body)) or
            verdict.get("status") not in {"clean", "emergency"} or
            (verdict.get("status") == "clean") != (verdict.get("error") is None)):
        raise RuntimeError("watchdog result verdict identity is invalid")
    if require_clean and verdict["status"] != "clean":
        raise RuntimeError(f"watchdog emergency forbids job success: {verdict['error']}")
    return verdict


def _validate_scale_before_launch(job: Job, *, release_sha: str) -> Optional[dict]:
    policy = job.scale_policy or {}
    if job.scientific_rows >= SCALE_PERFORMANCE_ROWS:
        from experiments.round0005_performance_gate import derive_scale_rows

        declared_derivation = policy.get("row_derivation")
        if not isinstance(declared_derivation, dict):
            raise RuntimeError(f"scale job {job.name} lacks an exact row derivation")
        embedding_input = declared_derivation.get("embedding_input") or {}
        embedding_path = embedding_input.get("canonical_path")
        dimensions = declared_derivation.get("dimensions")
        observed_derivation = derive_scale_rows(
            embedding_path, dimensions=dimensions)
        if observed_derivation != declared_derivation:
            raise RuntimeError(
                f"scale row derivation changed before launch for {job.name}: "
                f"declared={declared_derivation!r} observed={observed_derivation!r}")
        return require_scale_performance_gate(
            job.performance_gate_path, scientific_rows=job.scientific_rows,
            scale_policy=policy, row_derivation=observed_derivation,
            release_sha=release_sha)

    evidence = policy.get("row_evidence")
    if job.scientific_rows > 0:
        path = evidence["canonical_path"]
        kind = evidence["kind"]
        if kind == "npy_shape":
            import numpy as np
            observed_rows = int(np.load(path, mmap_mode="r").shape[0])
        elif kind == "parquet_rows":
            import pyarrow.parquet as pq
            observed_rows = int(pq.ParquetFile(path).metadata.num_rows)
        elif kind == "seal_rows":
            payload = json.load(open(path, encoding="utf-8"))
            candidates = [payload.get(field) for field in
                          ("rows", "row_count", "expected_rows") if field in payload]
            if len(candidates) != 1:
                raise RuntimeError(
                    f"row seal {path} must expose exactly one structural row count")
            observed_rows = int(candidates[0])
        else:
            raise RuntimeError(f"unsupported structural row evidence kind: {kind}")
        if observed_rows != job.scientific_rows:
            raise RuntimeError(
                f"scientific_rows lie/default rejected for {job.name}: "
                f"declared={job.scientific_rows} derived={observed_rows}")
    return require_scale_performance_gate(
        job.performance_gate_path, scientific_rows=job.scientific_rows,
        scale_policy=policy, release_sha=release_sha)


def _strict_gpu_preflight(manifest: dict, job: Job) -> dict:
    if job.required_free_gb <= 0:
        return {"gpu_required": False, "cuda_visible_devices":
                manifest["child_environment"]["CUDA_VISIBLE_DEVICES"]}
    snap = gpu_snapshot(strict=True, manifest=manifest)
    allowed = validate_allowed_processes(manifest["allowed_processes"], snapshot=snap)
    free_gb = float(snap["free_mb"]) / 1024.0
    if free_gb < float(job.required_free_gb):
        raise RuntimeError(
            f"free GPU memory {free_gb:.3f} GB is below {job.required_free_gb:.3f} GB")
    return {"gpu_required": True, "free_gb": free_gb, "allowed_processes": allowed}


def _run_admitted_jobs(jobs: Optional[list], *, controller_id: Optional[str], admission,
                       controller_capability: dict, fixture_only: bool,
                       launch_edge_hook=None, post_child_integrity_hook=None,
                       terminal_integrity_hook=None,
                       telemetry_interval_s: float = 5.0,
                       telemetry_provider=None) -> dict:
    # This is deliberately the first operation: fake/copied/stale capabilities
    # cannot create checkpoints, logs, output parents, leases, or children.
    from .queue_admission import QueueAdmission
    QueueAdmission._verify_controller_capability(
        admission, controller_capability, fixture_only=fixture_only)
    cid = controller_id or f"ctl-{uuid.uuid4().hex[:8]}"
    controller_claim = controller_capability["controller_claim"]
    controller_claim_sha256 = controller_claim["identity_sha256"]
    if controller_claim.get("controller_id") != cid:
        raise RuntimeError("controller ID differs from the issued QueueAdmission claim")
    manifest = admission.manifest
    writer = _CheckpointWriter(
        manifest["controller_checkpoints_dir"],
        controller_claim_sha256=controller_claim_sha256)
    summary_path = manifest["controller_terminal_summary"]
    refuse_existing(summary_path, label="controller terminal summary")
    summary = {
        "schema": "round0005_controller_terminal.v3", "controller_id": cid,
        "fixture_only": fixture_only,
        "controller_pid": os.getpid(),
        "controller_starttime_ticks": _proc_starttime_ticks(os.getpid()),
        "started": _utcnow(), "queue_manifest_path": admission.manifest_path,
        "queue_manifest_sha256": admission.manifest_sha256,
        "queue_release_sha": manifest["release_sha"], "jobs": [],
        "controller_claim": controller_claim,
        "controller_claim_sha256": controller_claim_sha256,
        "gpu_elapsed_s": 0.0, "terminal_verdict": "running",
    }
    round0015 = manifest.get("round_id") == "0015"
    round0016 = manifest.get("round_id") == "0016"
    round0017 = manifest.get("round_id") == "0017"
    terminal_release_round = round0015 or round0016 or round0017
    done: set[str] = set()
    cumulative_registry: dict[str, dict] = {}
    lease = None
    active_process = None
    watchdog_handle = None
    terminal_published = False
    try:
        if jobs is None:
            jobs = QueueAdmission.runtime_jobs(admission)
        QueueAdmission.assert_runtime_jobs(admission, jobs)
        if not jobs or [job.name for job in jobs] != [
                raw["id"] for raw in manifest["jobs"]]:
            raise RuntimeError("controller requires the exact nonempty ordered manifest jobs")
        aggregate_registered = sum(float(job.p90_wall_s) * 1.15 for job in jobs)
        if aggregate_registered > float(manifest["gpu_hours_cap"]) * 3600.0:
            raise RuntimeError(
                "registered six-node p90 +15% aggregate exceeds the queue GPU cap")
        writer.write("admission", {
            "controller_id": cid, "manifest_sha256": admission.manifest_sha256,
            "controller_claim": controller_claim,
            "controller_entry_gate": controller_capability["entry_gate"],
            "controller_entry_gate_sha256": controller_claim["entry_gate_sha256"],
            "construction_receipt": admission.construction_receipt_path,
            "gate_preparation_receipt": admission.gate_preparation_signature,
            "runtime_jobs_match_manifest": True,
            "aggregate_registered_with_margin_s": aggregate_registered,
        })
        first_fit = _require_budget_fit(
            manifest, jobs[0], gpu_elapsed_s=0.0, phase="lease-acquire")
        lease_timeout = max(0.0, min(
            5.0, first_fit["remaining_deadline_s"] - first_fit["required_with_margin_s"]))
        lease = GpuLease(
            path=manifest["lease_path"], timeout=lease_timeout,
            controller_id=cid).acquire()
        lease_identity = lease.verify_current()
        writer.write("lease-acquired", {
            "controller_id": cid, "lease_path": manifest["lease_path"],
            "lease_owner": lease_identity,
        })
        for job in jobs:
            _verify_cumulative_registry(cumulative_registry)
            record = {"name": job.name, "status": "preparing"}
            summary["jobs"].append(record)
            missing_deps = [dependency for dependency in job.deps if dependency not in done]
            if missing_deps:
                raise RuntimeError(f"{job.name} has unsatisfied dependencies {missing_deps}")
            fit = _require_budget_fit(
                manifest, job, gpu_elapsed_s=summary["gpu_elapsed_s"],
                phase="job-boundary")
            _fresh_job_paths(job)
            target_scale_round = manifest.get("round_id") in {
                "0014", "0015", "0016", "0017",
            }
            if job.scientific_rows >= SCALE_PERFORMANCE_ROWS and not target_scale_round:
                raise RuntimeError("issued Round 0005 program unexpectedly contains a scale node")
            if not job.node_policy:
                raise RuntimeError("runtime node lacks its canonical derived policy")
            if target_scale_round:
                if round0017:
                    NODE_BY_ID = importlib.import_module(
                        ".round0017_program", __package__).NODE_BY_ID
                    canary_schema = "round0017-canary-verdict-v1"
                    round_label = "Round 0017"
                elif round0016:
                    NODE_BY_ID = importlib.import_module(
                        ".round0016_program", __package__).NODE_BY_ID
                    canary_schema = "round0016-canary-verdict-v1"
                    round_label = "Round 0016"
                elif round0015:
                    from .round0015_program import NODE_BY_ID
                    canary_schema = "round0015-canary-verdict-v1"
                    round_label = "Round 0015"
                else:
                    from .round0014_program import NODE_BY_ID
                    canary_schema = "round0014-canary-verdict-v1"
                    round_label = "Round 0014"

                expected_training = NODE_BY_ID[job.name].training_performed
                if job.node_policy.get("training_performed") is not expected_training:
                    raise RuntimeError(f"{round_label} runtime training policy changed")
                if job.name == "train_seed42_30m":
                    canary_output = manifest["jobs"][0]["outputs"][0]
                    verdict_path = os.path.join(canary_output, "verdict.json")
                    with open(verdict_path, encoding="utf-8") as handle:
                        verdict = json.load(handle)
                    verdict_body = {key: verdict[key] for key in verdict
                                    if key != "identity_sha256"}
                    evidence = verdict.get("evidence")
                    if (verdict.get("schema") != canary_schema or
                            verdict.get("passed") is not True or
                            verdict.get("optimizer_updates") != 0 or
                            verdict.get("pipeline") != "device_uniform" or
                            verdict.get("sampling") !=
                            "uniform-over-directed-edges" or
                            not isinstance(verdict.get("headroom_gib"), (int, float)) or
                            verdict["headroom_gib"] < 1.5 or
                            verdict.get("scorer_scalar_equivalence") is not True or
                            verdict.get("semantic_render_alignment") is not True or
                            not isinstance(verdict.get(
                                "registered_p90_plus_margin_seconds"), (int, float)) or
                            verdict["registered_p90_plus_margin_seconds"] >
                            float(manifest["gpu_hours_cap"]) * 3600.0 or
                            not isinstance(evidence, dict) or
                            expected_input_signature(
                                evidence.get("canonical_path", "")) != evidence or
                            verdict.get("identity_sha256") !=
                            sha256_bytes(canonical_json(verdict_body))):
                        raise RuntimeError(
                            f"{round_label} training predecessor is not the passing exact canary")
            elif job.node_policy.get("training_performed") is not False:
                raise RuntimeError("runtime node lacks canonical derived no-training policy")
            preflight = _strict_gpu_preflight(manifest, job)
            lease_identity = lease.verify_current()
            child_env = dict(manifest["child_environment"])
            if any(key.startswith("BASEMAP_UNSAFE") for key in child_env):
                raise RuntimeError("unsafe BASEMAP bypass variable in sealed child environment")
            child_env["BASEMAP_GPU_LEASE_FD"] = str(lease.fileno())
            child_env["BASEMAP_GPU_LEASE_TOKEN"] = lease.token
            child_env["BASEMAP_ROUND0005_ADMISSION"] = admission.manifest_sha256
            child_env["BASEMAP_ROUND0005_MANIFEST"] = admission.manifest_path
            child_env["BASEMAP_ROUND0005_NODE"] = job.name
            capability_parent = capability_child = None
            child_capability_contract = None
            launch_nonce = None
            watchdog_channel_nonce = uuid.uuid4().hex
            if not fixture_only:
                capability_parent, capability_child = _new_child_capability_channel()
                launch_nonce = secrets.token_hex(32)
                child_env[_CHILD_CAPABILITY_FD_ENV] = str(capability_child.fileno())
                child_env[_CHILD_CAPABILITY_NONCE_ENV] = launch_nonce
            controller_parent_pid = os.getpid()
            log_fd, temporary_log = tempfile.mkstemp(
                prefix=f".{job.name}.child-log.", dir=writer.root)
            log_handle = os.fdopen(log_fd, "wb")
            watchdog_read_fd, watchdog_write_fd = os.pipe2(os.O_CLOEXEC)
            launched = False
            certified = False
            started_mono = time.monotonic()
            try:
                gate_receipt = QueueAdmission.boundary(admission, job.name)
                writer.write("boundary", {
                    "controller_id": cid, "job": job.name, "budget": fit,
                    "derived_node_policy": job.node_policy,
                    "gpu_preflight": preflight, "gate_receipt": gate_receipt,
                    "lease_owner": lease_identity,
                })
                # This comprehensive comparison is immediately adjacent to
                # Popen.  The child remains blocked on its inherited capability;
                # a second live control comparison below is the work-launch edge.
                # includes the manifest, checkout, environment/freeze, venv
                # executable, release/gate-preparation receipts, Roundwatch
                # implementation, every future/current queue input, outputs,
                # and the controller-owned lease.  Popen is the next operation.
                QueueAdmission.comprehensive_integrity_boundary(
                    admission,
                    job.name, phase="gate-response-to-Popen",
                    include_output_absence=True,
                    after_comparison_hook=((lambda _state: launch_edge_hook(job))
                                           if launch_edge_hook is not None else None),
                    runtime_expected=_cumulative_runtime_state(
                        lease, cumulative_registry),
                    runtime_probe=lambda: _cumulative_runtime_state(
                        lease, cumulative_registry))
                active_process = subprocess.Popen(
                    _parent_death_exec_argv(
                        list(job.argv), expected_parent=controller_parent_pid),
                    cwd=job.cwd, stdout=log_handle, stderr=subprocess.STDOUT,
                    start_new_session=True, env=child_env,
                    pass_fds=(*lease.pass_fds(), *((capability_child.fileno(),)
                              if capability_child is not None else ())))
                if capability_child is not None:
                    capability_child.close()
                    capability_child = None
                launched = True
                child_pid = active_process.pid
                record.update({"status": "running", "child_pid": child_pid,
                               "pgid": child_pid, "started": _utcnow()})
                hard_runtime_s = float(job.p90_wall_s) * 1.15
                now_epoch = time.time()
                queue_deadline_epoch = _deadline(manifest["deadline_utc"]).timestamp()
                cap_remaining_s = (float(manifest["gpu_hours_cap"]) * 3600.0 -
                                   summary["gpu_elapsed_s"])
                watchdog_handle = _start_watchdog(
                    read_fd=watchdog_read_fd, write_fd=watchdog_write_fd,
                    checkpoint_root=writer.root, manifest=manifest, job=job,
                    child_pid=child_pid, lease=lease,
                    deadline_epoch=queue_deadline_epoch,
                    runtime_deadline_epoch=now_epoch + hard_runtime_s,
                    gpu_cap_deadline_epoch=now_epoch + cap_remaining_s,
                    result_nonce=watchdog_channel_nonce,
                    controller_claim_sha256=controller_claim_sha256)
                watchdog_write_fd = None
                capability_ack = None
                launch_gate_receipt = None
                accountant = (RuntimeGpuAccountant(
                    manifest=manifest, job=job, root_pid=child_pid, lease=lease)
                    if job.node_policy.get("gpu_required") else None)
                last_telemetry = started_mono - max(float(telemetry_interval_s), 0.0)
                last_gpu = {"gpu_required": False}
                if not fixture_only:
                    script = job.node_policy["canonical_script"]
                    _receive_child_hello(
                        capability_parent, active_process, launch_nonce=launch_nonce,
                        job=job, script=script)
                    # The process exists but is still blocked before argument
                    # parsing, output creation, CUDA, or scientific work.  Check
                    # current control and all mutable state at the release edge.
                    launch_gate_receipt = QueueAdmission.boundary(admission, job.name)
                    launch_integrity = QueueAdmission.comprehensive_integrity_boundary(
                        admission,
                        job.name, phase="child-capability-release",
                        include_output_absence=True,
                        runtime_expected=_cumulative_runtime_state(
                            lease, cumulative_registry),
                        runtime_probe=lambda: _cumulative_runtime_state(
                            lease, cumulative_registry))
                    # Capture one independent controller sample while the real
                    # child is still blocked on its capability.  Every node
                    # consequently has authenticated telemetry even if its
                    # scientific phase exits before the first periodic tick.
                    if accountant is not None:
                        last_gpu = accountant.snapshot()
                        writer.write("gpu-telemetry", {
                            "controller_id": cid, "job": job.name,
                            "snapshot": last_gpu,
                        })
                        last_telemetry = time.monotonic()
                    gate_identity = launch_gate_receipt["gate"]["event_identity"]
                    contract = _build_round0005_child_contract(
                        launch_nonce=launch_nonce, controller_id=cid,
                        child_pid=child_pid, job=job, jobs=jobs,
                        child_environment=child_env, manifest=manifest,
                        manifest_sha256=admission.manifest_sha256,
                        gate_identity=gate_identity,
                        lease_identity=lease.verify_current(),
                        telemetry_interval_s=telemetry_interval_s,
                        watchdog_pid=watchdog_handle.pid,
                        watchdog_nonce=watchdog_channel_nonce,
                        launch_integrity=launch_integrity,
                        controller_claim=controller_claim)
                    child_capability_contract = contract
                    capability_ack = _issue_child_capability(
                        capability_parent, contract=contract, process=active_process)
                    record["child_capability_ack"] = capability_ack
                    capability_parent.close()
                    capability_parent = None
                writer.write("launch", {
                    "controller_id": cid, "job": job.name, "child_pid": child_pid,
                    "watchdog_pid": watchdog_handle.pid, "budget": fit,
                    "child_capability_ack": capability_ack,
                    "launch_gate_receipt": launch_gate_receipt,
                })
                runtime_error = None
                while active_process.poll() is None:
                    now_mono = time.monotonic()
                    elapsed = now_mono - started_mono
                    cap_elapsed = summary["gpu_elapsed_s"] + elapsed
                    if datetime.datetime.now(datetime.timezone.utc) >= _deadline(
                            manifest["deadline_utc"]):
                        runtime_error = "strict deadline_utc reached while child was running"
                    elif cap_elapsed >= float(manifest["gpu_hours_cap"]) * 3600.0:
                        runtime_error = "cumulative gpu_hours_cap reached while child was running"
                    elif elapsed >= hard_runtime_s:
                        runtime_error = "child exceeded registered p90 wall +15% bound"
                    try:
                        lease.verify_current()
                    except Exception as exc:
                        runtime_error = f"GPU lease identity changed at runtime: {exc}"
                    try:
                        _verify_cumulative_registry(cumulative_registry)
                    except Exception as exc:
                        runtime_error = (
                            f"completed predecessor integrity changed at runtime: {exc}")
                    if runtime_error:
                        writer.write("runtime-violation", {
                            "controller_id": cid, "job": job.name,
                            "error": runtime_error, "snapshot": last_gpu,
                        })
                        _terminate_process_group(active_process)
                        break
                    if now_mono - last_telemetry >= float(telemetry_interval_s):
                        try:
                            if accountant is not None:
                                supplied = (telemetry_provider(job, child_pid)
                                            if telemetry_provider is not None else None)
                                last_gpu = accountant.snapshot(supplied_gpu=supplied)
                            else:
                                last_gpu = {
                                    "schema": "round0005_runtime_gpu_telemetry.v2",
                                    "at": _utcnow(), "job": job.name,
                                    "gpu_required": False,
                                    "lease_owner": lease.verify_current(),
                                }
                            writer.write("gpu-telemetry", {
                                "controller_id": cid, "job": job.name,
                                "snapshot": last_gpu,
                            })
                        except RuntimeGpuViolation as exc:
                            last_gpu = exc.snapshot
                            runtime_error = f"runtime GPU policy violation: {exc}"
                            writer.write("runtime-violation", {
                                "controller_id": cid, "job": job.name,
                                "error": runtime_error, "snapshot": last_gpu,
                            })
                            _terminate_process_group(active_process)
                            break
                        except Exception as exc:
                            runtime_error = (
                                f"runtime GPU telemetry failed closed: "
                                f"{type(exc).__name__}: {exc}")
                            writer.write("runtime-violation", {
                                "controller_id": cid, "job": job.name,
                                "error": runtime_error, "snapshot": last_gpu,
                            })
                            _terminate_process_group(active_process)
                            break
                        last_telemetry = now_mono
                    time.sleep(0.1)
                return_code = active_process.wait(timeout=5)
                elapsed = time.monotonic() - started_mono
                summary["gpu_elapsed_s"] += elapsed
                stopping_watchdog = watchdog_handle
                watchdog_handle = None
                watchdog_verdict = _stop_watchdog(stopping_watchdog)
                if accountant is not None and runtime_error is None:
                    # The child can disappear between wait and /proc sampling, so
                    # the last in-loop immutable sample is the exact final sample.
                    gpu_post = last_gpu
                else:
                    gpu_post = last_gpu
                log_handle.flush()
                os.fsync(log_handle.fileno())
                # Repeat the complete boundary after every child and before any
                # output signature, job manifest, or done marker is produced.
                QueueAdmission.comprehensive_integrity_boundary(
                    admission,
                    job.name, phase="post-child-integrity",
                    include_output_absence=False,
                    after_comparison_hook=((lambda _state: post_child_integrity_hook(job))
                                           if post_child_integrity_hook is not None else None),
                    runtime_expected=_cumulative_runtime_state(
                        lease, cumulative_registry),
                    runtime_probe=lambda: _cumulative_runtime_state(
                        lease, cumulative_registry))
                output_signatures = {path: _strict_output_signature(path)
                                     for path in job.outputs}
                outputs_ok = all(value is not None for value in output_signatures.values())
                success = (return_code == 0 and outputs_ok and runtime_error is None and
                           watchdog_verdict.get("status") == "clean")
                status = ("ok" if success else runtime_error or
                          (f"exit_{return_code}" if return_code else "missing_outputs"))
                record.update({
                    "status": status, "exit_code": return_code,
                    "seconds": round(elapsed, 3),
                    "output_signatures": output_signatures,
                    "gpu_post": gpu_post,
                    "watchdog_verdict": watchdog_verdict,
                })
                atomic_copy_new(temporary_log, job.log, immutable=True)
                log_handle.close()
                log_handle = None
                os.unlink(temporary_log)
                temporary_log = None
                final_job = {
                    "schema": "round0005_controller_job.v3", "controller_id": cid,
                    "fixture_only": fixture_only,
                    "job": job.name, "manifest_sha256": admission.manifest_sha256,
                    "controller_claim_sha256": controller_claim_sha256,
                    "child_capability_ack": capability_ack,
                    "child_capability": child_capability_contract,
                    "runtime_contract": job.manifest_contract(), "record": record,
                    "post_child_integrity": True,
                    "watchdog_clean": watchdog_verdict["status"] == "clean",
                    "finished": _utcnow(),
                }
                atomic_write_new_json(job.manifest, final_job, immutable=True)
                certified = True
                writer.write("completion", {
                    "controller_id": cid, "job": job.name, "status": status,
                    "gpu_elapsed_s": summary["gpu_elapsed_s"],
                })
                if not success:
                    raise RuntimeError(f"{job.name} failed: {status}")
                atomic_write_new_json(job.done_marker, {
                    "schema": "round0005_controller_done.v3", "status": "ok",
                    "fixture_only": fixture_only,
                    "job": job.name, "manifest_sha256": admission.manifest_sha256,
                    "controller_claim_sha256": controller_claim_sha256,
                    "runtime_contract_sha256": sha256_bytes(
                        canonical_json(job.manifest_contract())),
                    "output_signatures": output_signatures,
                    "post_child_integrity": True, "finished": _utcnow(),
                    "watchdog_verdict_sha256": watchdog_verdict["identity_sha256"],
                }, immutable=True)
                cumulative_registry = _extend_cumulative_registry(
                    cumulative_registry, job=job,
                    checkpoint_root=writer.root,
                    gate_receipts_root=manifest["gate_receipts_dir"])
                registry_sha = sha256_bytes(canonical_json(cumulative_registry))
                writer.write("cumulative-registry", {
                    "controller_id": cid, "completed_job": job.name,
                    "completed_jobs": [*sorted(done), job.name],
                    "entry_count": len(cumulative_registry),
                    "registry_sha256": registry_sha,
                })
                cumulative_registry = _extend_cumulative_registry(
                    cumulative_registry, job=job,
                    checkpoint_root=writer.root,
                    gate_receipts_root=manifest["gate_receipts_dir"])
                record["cumulative_registry_sha256"] = sha256_bytes(
                    canonical_json(cumulative_registry))
                done.add(job.name)
            except Exception as exc:
                child_pid = active_process.pid if launched and active_process is not None else None
                if launched and active_process is not None and active_process.poll() is None:
                    _terminate_process_group(active_process)
                if watchdog_handle is not None:
                    stopping_watchdog = watchdog_handle
                    watchdog_handle = None
                    try:
                        record["watchdog_verdict"] = _stop_watchdog(
                            stopping_watchdog, require_clean=False)
                    except Exception as watchdog_exc:
                        record["watchdog_result_error"] = (
                            f"{type(watchdog_exc).__name__}: {watchdog_exc}")
                if not launched:
                    for descriptor in (watchdog_read_fd, watchdog_write_fd):
                        if descriptor is not None:
                            try:
                                os.close(descriptor)
                            except OSError:
                                pass
                phase = ("post-child-integrity" if launched else
                         "gate-response-to-Popen")
                try:
                    receipt = QueueAdmission.record_integrity_failure(
                        admission,
                        job.name, exc, phase=phase, child_pid=child_pid)
                except Exception as receipt_exc:
                    receipt = None
                    record["integrity_receipt_error"] = repr(receipt_exc)
                record.update({
                    "status": "exception", "error": f"{type(exc).__name__}: {exc}",
                    "integrity_receipt": receipt,
                    "post_child_certified": certified,
                })
                raise
            finally:
                for channel in (capability_parent, capability_child):
                    if channel is not None:
                        try:
                            channel.close()
                        except OSError:
                            pass
                try:
                    if log_handle is not None:
                        log_handle.close()
                except Exception:
                    pass
                try:
                    if temporary_log is not None:
                        os.unlink(temporary_log)
                except FileNotFoundError:
                    pass
                active_process = None
        if len(done) != len(jobs):
            raise RuntimeError("not every required manifest node completed")
        # Complete the scientific terminal comparison while the lease is still
        # held.  Round 0015 defers its terminal publication until its exact
        # neutralization/release/proof receipt exists.
        _verify_cumulative_registry(cumulative_registry)
        terminal_gate = QueueAdmission.terminal_boundary(admission, jobs[-1].name)
        cumulative_registry = _extend_cumulative_registry(
            cumulative_registry, job=jobs[-1], checkpoint_root=writer.root,
            gate_receipts_root=manifest["gate_receipts_dir"])
        terminal_integrity = QueueAdmission.comprehensive_integrity_boundary(
            admission,
            jobs[-1].name, phase="terminal-comprehensive-integrity",
            include_output_absence=False,
            after_comparison_hook=(
                (lambda _state: terminal_integrity_hook(jobs[-1]))
                if terminal_integrity_hook is not None else None),
            runtime_expected=_cumulative_runtime_state(
                lease, cumulative_registry),
            runtime_probe=lambda: _cumulative_runtime_state(
                lease, cumulative_registry))
        terminal_checkpoint = writer.write("terminal-comprehensive", {
            "controller_id": cid, "terminal_gate_receipt": terminal_gate,
            "integrity_match": terminal_integrity["integrity_match"],
            "lease_owner": lease.verify_current(),
            "cumulative_registry_sha256": sha256_bytes(
                canonical_json(cumulative_registry)),
        })
        cumulative_registry = _extend_cumulative_registry(
            cumulative_registry, job=jobs[-1], checkpoint_root=writer.root,
            gate_receipts_root=manifest["gate_receipts_dir"])
        summary["terminal_verdict"] = "passed"
        summary["stop_reason"] = "every required manifest node succeeded"
        summary["terminal_gate_receipt"] = terminal_gate["receipt_path"]
        summary["terminal_comprehensive_checkpoint"] = terminal_checkpoint
        summary["cumulative_registry"] = cumulative_registry
        summary["cumulative_registry_sha256"] = sha256_bytes(
            canonical_json(cumulative_registry))
        summary["finished"] = _utcnow()
        summary["required_jobs"] = [job.name for job in jobs]
        summary["completed_jobs"] = [job.name for job in jobs]
        if not terminal_release_round:
            writer.write("terminal", {
                "controller_id": cid, "terminal_verdict": "passed",
                "stop_reason": summary["stop_reason"],
                "gpu_elapsed_s": summary["gpu_elapsed_s"],
                "cumulative_registry_sha256": summary[
                    "cumulative_registry_sha256"],
            })
            cumulative_registry = _extend_cumulative_registry(
                cumulative_registry, job=jobs[-1], checkpoint_root=writer.root,
                gate_receipts_root=manifest["gate_receipts_dir"])
            summary["cumulative_registry"] = cumulative_registry
            summary["cumulative_registry_sha256"] = sha256_bytes(
                canonical_json(cumulative_registry))
            _verify_cumulative_registry(cumulative_registry)
            lease.verify_current()
            atomic_write_new_json(summary_path, summary, immutable=True)
            terminal_published = True
    except Exception as exc:
        if active_process is not None and active_process.poll() is None:
            _terminate_process_group(active_process)
        if watchdog_handle is not None:
            stopping_watchdog = watchdog_handle
            watchdog_handle = None
            try:
                _stop_watchdog(stopping_watchdog, require_clean=False)
            except Exception:
                pass
        summary["terminal_verdict"] = "failed"
        summary["stop_reason"] = f"{type(exc).__name__}: {exc}"
        writer.write("exception", {
            "controller_id": cid, "error": summary["stop_reason"],
            "gpu_elapsed_s": summary["gpu_elapsed_s"],
        })
    finally:
        if terminal_release_round and lease is not None:
            summary["finished"] = summary.get("finished") or _utcnow()
            summary["required_jobs"] = [job.name for job in (jobs or [])]
            summary["completed_jobs"] = [job.name for job in (jobs or [])
                                         if job.name in done]
            terminal_identity = (_round0017_terminal_identity(summary)
                                 if round0017 else
                                 (_round0016_terminal_identity(summary)
                                  if round0016 else
                                  _round0015_terminal_identity(summary)))
            summary["pre_release_terminal_identity"] = terminal_identity
            try:
                release_receipt = _round0015_release_with_receipt(
                    lease,
                    receipt_path=manifest["lease_release_receipt"],
                    terminal_identity=terminal_identity,
                    _round0016=round0016, _round0017=round0017)
                summary["lease_release_receipt"] = expected_input_signature(
                    manifest["lease_release_receipt"])
                summary["lease_release_receipt_identity_sha256"] = \
                    release_receipt["identity_sha256"]
            except BaseException as release_exc:
                # The target helper normally releases before reporting a failed
                # proof.  This cleanup covers only validation failures that
                # happened before it reached that release step.
                if lease._fd is not None or lease._guard_fd is not None:
                    try:
                        lease.release()
                    except BaseException as cleanup_exc:
                        try:
                            release_exc.add_note(
                                f"Round {manifest.get('round_id')} fallback lease cleanup failed: "
                                f"{cleanup_exc!r}")
                        except AttributeError:
                            pass
                previous = summary.get("stop_reason")
                detail = f"{type(release_exc).__name__}: {release_exc}"
                summary["terminal_verdict"] = "failed"
                summary["stop_reason"] = (
                    f"{previous}; terminal lease release failed: {detail}"
                    if previous else f"terminal lease release failed: {detail}")
                if os.path.isfile(manifest["lease_release_receipt"]):
                    summary["lease_release_receipt"] = expected_input_signature(
                        manifest["lease_release_receipt"])
        elif lease is not None:
            lease.release()
        if not terminal_published:
            summary["finished"] = summary.get("finished") or _utcnow()
            summary["required_jobs"] = summary.get(
                "required_jobs", [job.name for job in (jobs or [])])
            summary["completed_jobs"] = summary.get(
                "completed_jobs", [job.name for job in (jobs or [])
                                   if job.name in done])
            writer.write("terminal", {
                "controller_id": cid,
                "terminal_verdict": summary["terminal_verdict"],
                "stop_reason": summary.get("stop_reason"),
                "gpu_elapsed_s": summary["gpu_elapsed_s"],
                "lease_release_receipt": summary.get("lease_release_receipt"),
            })
            atomic_write_new_json(summary_path, summary, immutable=True)
    return summary


def _run_jobs_fixture_only(jobs: list, controller_id: Optional[str] = None,
                           allowed_pids=(), summary_path: Optional[str] = None,
                           launch_edge_hook=None) -> dict:
    """Private CUDA-hidden compatibility helper for legacy CPU unit fixtures."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("fixture-only legacy controller requires CUDA_VISIBLE_DEVICES=''")
    return _run_jobs_legacy(
        jobs, controller_id=controller_id, allowed_pids=allowed_pids,
        summary_path=summary_path, launch_edge_hook=launch_edge_hook)


def _entry_boundary_failure_summary(*, admission, failure, fixture_only: bool,
                                    controller_id: Optional[str]) -> dict:
    """Publish failure evidence only for a registry-authenticated claimed object."""
    from .queue_admission import QueueAdmission
    if not QueueAdmission._is_authentic_capability(
            admission, claimed=True):
        raise failure
    jobs = QueueAdmission.runtime_jobs(admission)
    matches = [job for job in jobs if job.name == failure.job_name]
    if len(matches) != 1:
        raise failure
    original = failure.original
    receipt = QueueAdmission.record_integrity_failure(
        admission, failure.job_name, original,
        phase="controller-entry-boundary", child_pid=None)
    cid = controller_id or f"ctl-{uuid.uuid4().hex[:8]}"
    now = _utcnow()
    record = {
        "name": failure.job_name, "status": "exception", "child_pid": None,
        "error": f"{type(original).__name__}: {original}",
        "integrity_receipt": receipt, "post_child_certified": False,
    }
    summary = {
        "schema": "round0005_controller_terminal.v3", "controller_id": cid,
        "fixture_only": fixture_only, "controller_pid": os.getpid(),
        "controller_starttime_ticks": _proc_starttime_ticks(os.getpid()),
        "started": now, "finished": now,
        "queue_manifest_path": admission.manifest_path,
        "queue_manifest_sha256": admission.manifest_sha256,
        "queue_release_sha": admission.manifest["release_sha"],
        "jobs": [record], "gpu_elapsed_s": 0.0,
        "terminal_verdict": "failed",
        "stop_reason": f"{type(original).__name__}: {original}",
        "required_jobs": [job.name for job in jobs], "completed_jobs": [],
    }
    summary_path = admission.manifest["controller_terminal_summary"]
    refuse_existing(summary_path, label="controller terminal summary")
    atomic_write_new_json(summary_path, summary, immutable=True)
    return summary


def run_jobs(jobs: Optional[list] = None, controller_id: Optional[str] = None,
             allowed_pids=(), summary_path: Optional[str] = None, admission=None,
             launch_edge_hook=None, post_child_integrity_hook=None,
             telemetry_interval_s: float = 5.0, telemetry_provider=None) -> dict:
    """Run only an exact Round 0005 admission object in production."""
    from .queue_admission import (ControllerEntryBoundaryFailure,
                                  QueueAdmission)
    if admission is None or type(admission) is not QueueAdmission:
        raise RuntimeError(
            "production run_jobs requires an exact Round 0005 QueueAdmission; "
            "legacy behavior is private and CUDA-hidden")
    if getattr(admission, "fixture_only", None) is not False:
        raise RuntimeError("production run_jobs rejects fixture QueueAdmission capabilities")
    if allowed_pids or (summary_path is not None and
                        os.path.realpath(summary_path) !=
                        admission.manifest["controller_terminal_summary"]):
        raise RuntimeError("admitted runtime policy must come only from the hashed manifest")
    if (launch_edge_hook is not None or post_child_integrity_hook is not None or
            telemetry_provider is not None or telemetry_interval_s != 5.0):
        if getattr(admission, "fixture_only", False) is not True:
            raise RuntimeError("production admission exposes no adversarial/runtime test hooks")
    cid = controller_id or f"ctl-{uuid.uuid4().hex[:8]}"
    try:
        runtime_jobs, capability = QueueAdmission._claim_controller(
            admission, fixture_only=False, jobs=jobs, controller_id=cid)
    except ControllerEntryBoundaryFailure as exc:
        return _entry_boundary_failure_summary(
            admission=admission, failure=exc, fixture_only=False,
            controller_id=cid)
    return _run_admitted_jobs(
        runtime_jobs, controller_id=cid, admission=admission,
        controller_capability=capability, fixture_only=False,
        launch_edge_hook=launch_edge_hook,
        post_child_integrity_hook=post_child_integrity_hook,
        telemetry_interval_s=telemetry_interval_s,
        telemetry_provider=telemetry_provider)


def _run_admitted_queue_fixture_only(
        *, admission, jobs: Optional[list] = None,
        controller_id: Optional[str] = None, launch_edge_hook=None,
        post_child_integrity_hook=None, telemetry_interval_s: float = 0.05,
        telemetry_provider=None, terminal_integrity_hook=None) -> dict:
    """Private exact six-node CUDA-hidden controller integration lane."""
    from .queue_admission import (ControllerEntryBoundaryFailure,
                                  QueueAdmission)
    if (os.environ.get("CUDA_VISIBLE_DEVICES") != "" or
            type(admission) is not QueueAdmission or admission.fixture_only is not True):
        raise RuntimeError("fixture queue controller requires an authentic fixture admission")
    cid = controller_id or f"ctl-{uuid.uuid4().hex[:8]}"
    try:
        runtime_jobs, capability = QueueAdmission._claim_controller(
            admission, fixture_only=True, jobs=jobs, controller_id=cid)
    except ControllerEntryBoundaryFailure as exc:
        return _entry_boundary_failure_summary(
            admission=admission, failure=exc, fixture_only=True,
            controller_id=cid)
    return _run_admitted_jobs(
        runtime_jobs, controller_id=cid, admission=admission,
        controller_capability=capability, fixture_only=True,
        launch_edge_hook=launch_edge_hook,
        post_child_integrity_hook=post_child_integrity_hook,
        terminal_integrity_hook=terminal_integrity_hook,
        telemetry_interval_s=telemetry_interval_s,
        telemetry_provider=telemetry_provider)


def _admission_failure_evidence(raw: dict, manifest_path: str, exc: Exception) -> dict:
    summary = {
        "schema": "round0005_controller_terminal.v1", "controller_id": None,
        "controller_pid": os.getpid(), "started": _utcnow(), "finished": _utcnow(),
        "queue_manifest_path": manifest_path,
        "queue_manifest_sha256": (sha256_file(manifest_path)
                                    if os.path.isfile(manifest_path) else None),
        "jobs": [], "gpu_elapsed_s": 0.0, "terminal_verdict": "failed",
        "stop_reason": f"admission failure: {type(exc).__name__}: {exc}",
        "required_jobs": [job.get("id") for job in raw.get("jobs", [])
                          if isinstance(job, dict)], "completed_jobs": [],
    }
    # Never let arbitrary invalid JSON select filesystem outputs.  Valid
    # post-gate mutations receive their automatic receipt inside QueueAdmission,
    # using the trusted gate-preparation sidecar.
    return summary


def _run_internal_parent_death_exec(arguments: list[str]) -> int:
    if len(arguments) < 4 or arguments[1] != "--":
        raise RuntimeError("internal parent-death exec arguments are malformed")
    try:
        expected_parent = int(arguments[0])
    except ValueError as exc:
        raise RuntimeError("internal parent-death parent PID is invalid") from exc
    target = arguments[2:]
    if (expected_parent <= 0 or len(target) < 2 or
            not os.path.isabs(target[0]) or
            os.path.realpath(target[0]) != os.path.realpath(sys.executable)):
        raise RuntimeError("internal parent-death target is not the invoking Python")
    _child_parent_death_setup(expected_parent)
    os.execve(target[0], target, dict(os.environ))
    raise AssertionError("execve returned unexpectedly")


def _run_internal_watchdog(arguments: list[str]) -> int:
    if len(arguments) != 4:
        raise RuntimeError("internal watchdog descriptor arguments are malformed")
    try:
        config_fd, read_fd, result_fd, lease_fd = map(int, arguments)
    except ValueError as exc:
        raise RuntimeError("internal watchdog descriptor is invalid") from exc
    channel = socket.socket(fileno=config_fd)
    try:
        config = json.loads(channel.recv(8 << 20).decode("utf-8"))
    finally:
        channel.close()
    fields = {
        "schema", "result_nonce", "controller_pid", "checkpoint_root",
        "manifest", "job", "child_pid", "lease_path", "lease_token",
        "deadline_epoch", "runtime_deadline_epoch", "gpu_cap_deadline_epoch",
        "controller_claim_sha256",
    }
    if (not isinstance(config, dict) or set(config) != fields or
            config.get("schema") != "round0005_watchdog_spawn.v1" or
            not re.fullmatch(r"[0-9a-f]{64}", str(
                config.get("controller_claim_sha256", "")))):
        raise RuntimeError("internal watchdog spawn contract is incomplete")
    job_payload = config["job"]
    if (not isinstance(job_payload, dict) or set(job_payload) != {
            "name", "node_policy", "gpu_memory_cap_mb"}):
        raise RuntimeError("internal watchdog job contract is incomplete")
    job = Job(
        name=job_payload["name"], argv=[], outputs=[], done_marker="",
        node_policy=job_payload["node_policy"],
        gpu_memory_cap_mb=job_payload["gpu_memory_cap_mb"])
    lease = _InheritedLeaseView(
        fd=lease_fd, path=config["lease_path"], token=config["lease_token"])
    lease.verify_current()
    _watchdog_loop(
        read_fd=read_fd, result_fd=result_fd,
        result_nonce=config["result_nonce"],
        controller_pid=int(config["controller_pid"]),
        checkpoint_root=config["checkpoint_root"], manifest=config["manifest"],
        job=job, child_pid=int(config["child_pid"]), lease=lease,
        deadline_epoch=float(config["deadline_epoch"]),
        runtime_deadline_epoch=float(config["runtime_deadline_epoch"]),
        gpu_cap_deadline_epoch=float(config["gpu_cap_deadline_epoch"]),
        controller_claim_sha256=config["controller_claim_sha256"])
    return 0


def main(argv=None) -> int:
    import argparse
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ["--round0005-internal-parent-death-exec"]:
        return _run_internal_parent_death_exec(arguments[1:])
    if arguments[:1] == ["--round0005-internal-watchdog"]:
        return _run_internal_watchdog(arguments[1:])
    parser = argparse.ArgumentParser(
        description="Run the exact gate-hashed Round 0005 queue manifest")
    parser.add_argument("queue_manifest")
    args = parser.parse_args(arguments)
    spec_path = os.path.realpath(args.queue_manifest)
    raw = {}
    try:
        with open(spec_path, encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, dict):
            raise ValueError("runtime CLI accepts only a queue-manifest object")
        from .queue_admission import QueueAdmission
        repo_root = os.path.realpath(raw["repo_root"])
        admission = QueueAdmission(spec_path, repo_root)
        result = run_jobs(None, admission=admission)
    except Exception as exc:
        result = _admission_failure_evidence(raw, spec_path, exc)
    print(json.dumps(result, indent=1))
    return 0 if result.get("terminal_verdict") == "passed" else 3


if __name__ == "__main__":
    raise SystemExit(main())
