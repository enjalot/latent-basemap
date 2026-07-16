"""Immutable queue admission and per-job Roundwatch boundary checks."""
from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass

from .artifact_identity import (git_checkout_state, path_signature, sha256_bytes,
                                sha256_file)

FULL_SHA = re.compile(r"[0-9a-f]{40}")
HASH64 = re.compile(r"[0-9a-f]{64}")
CACHE_KEYS = ("XDG_CACHE_HOME", "TORCH_HOME", "HF_HOME", "TRITON_CACHE_DIR",
              "PYTHONPYCACHEPREFIX", "NUMBA_CACHE_DIR", "MPLCONFIGDIR")


def validate_queue_manifest(data: dict, path: str) -> None:
    required = {
        "schema_version": 1,
        "program": str,
        "round_id": str,
        "release_sha": str,
        "environment_freeze_sha": str,
        "environment_identity_sha": str,
        "gpu_hours_cap": (int, float),
        "jobs": list,
        "environment_manifest": str,
        "cache_environment": dict,
        "gate_receipts_dir": str,
        "repo_root": str,
        "allowed_pids": list,
    }
    if not isinstance(data, dict):
        raise ValueError("queue manifest must be a JSON object")
    if data.get("schema_version") != 1:
        raise ValueError("queue manifest schema_version must equal 1")
    for key, expected in required.items():
        if key not in data:
            raise ValueError(f"queue manifest missing {key}")
        if isinstance(expected, type) and not isinstance(data[key], expected):
            raise ValueError(f"queue manifest {key} must be {expected.__name__}")
        if isinstance(expected, tuple) and not isinstance(data[key], expected):
            raise ValueError(f"queue manifest {key} has wrong type")
    if not FULL_SHA.fullmatch(data["release_sha"]):
        raise ValueError("queue manifest release_sha must be a full lowercase SHA")
    for key in ("environment_freeze_sha", "environment_identity_sha"):
        if not HASH64.fullmatch(data[key]):
            raise ValueError(f"queue manifest {key} must be 64 lowercase hex characters")
    if not data["jobs"]:
        raise ValueError("queue manifest jobs must be nonempty")
    if float(data["gpu_hours_cap"]) <= 0:
        raise ValueError("queue manifest gpu_hours_cap must be positive")
    if not os.path.realpath(path).startswith("/data/"):
        raise ValueError("queue manifest must resolve under /data")
    for key in ("gate_receipts_dir", "environment_manifest"):
        if not os.path.realpath(data[key]).startswith("/data/"):
            raise ValueError(f"queue manifest {key} must resolve under /data")
    cache = data["cache_environment"]
    if str(cache.get("PYTHONDONTWRITEBYTECODE")) != "1":
        raise ValueError("queue cache_environment must disable Python bytecode")
    for key in CACHE_KEYS:
        value = cache.get(key)
        if not value or not os.path.realpath(str(value)).startswith("/data/"):
            raise ValueError(f"queue cache {key} must resolve under /data")
    seen = set()
    for position, job in enumerate(data["jobs"]):
        if not isinstance(job, dict):
            raise ValueError(f"queue job {position} must be an object")
        job_id = job.get("id") or job.get("name")
        if not isinstance(job_id, str) or not job_id:
            raise ValueError(f"queue job {position} needs a nonempty id")
        if job_id in seen:
            raise ValueError(f"queue job id is duplicated: {job_id}")
        seen.add(job_id)
        for key in ("argv", "inputs", "outputs"):
            if not isinstance(job.get(key), list) or not job[key]:
                raise ValueError(f"queue job {job_id} needs a nonempty {key} list")
        if not all(isinstance(value, str) and value for value in job["argv"] + job["inputs"]):
            raise ValueError(f"queue job {job_id} argv/inputs entries must be nonempty strings")
        if not all(isinstance(value, str) and os.path.realpath(value).startswith("/data/")
                   for value in job["outputs"]):
            raise ValueError(f"queue job {job_id} outputs must resolve under /data")
        for key in ("done_marker", "log", "manifest"):
            if not isinstance(job.get(key), str) or not os.path.realpath(job[key]).startswith("/data/"):
                raise ValueError(f"queue job {job_id} {key} must resolve under /data")
        if not isinstance(job.get("cwd"), str) or not job["cwd"]:
            raise ValueError(f"queue job {job_id} needs an explicit cwd")
        for key in ("required_free_gb", "predicted_wall_s"):
            if not isinstance(job.get(key), (int, float)) or float(job[key]) < 0:
                raise ValueError(f"queue job {job_id} needs nonnegative {key}")


@dataclass
class QueueAdmission:
    manifest_path: str
    repo_root: str
    roundwatch_bin: str = "roundwatch"

    def __post_init__(self):
        self.manifest_path = os.path.realpath(self.manifest_path)
        self.repo_root = os.path.realpath(self.repo_root)
        with open(self.manifest_path, encoding="utf-8") as handle:
            self.manifest = json.load(handle)
        validate_queue_manifest(self.manifest, self.manifest_path)
        self.manifest_sha256 = sha256_file(self.manifest_path)
        self.launch_checkout = self._verify_checkout()
        self.initial_environment = self._verify_environment_binding()
        self.resolved_venv = self.initial_environment["resolved_venv"]
        for job in self.manifest["jobs"]:
            executable = os.path.abspath(job["argv"][0])
            if not executable.startswith(self.resolved_venv + os.sep):
                raise RuntimeError(f"queue job {(job.get('id') or job.get('name'))!r} "
                                   f"does not use sealed venv {self.resolved_venv}: {executable}")
        self.initial_inputs = self._input_signatures()

    def _verify_checkout(self) -> dict:
        state = git_checkout_state(self.repo_root)
        if state["head"] != self.manifest["release_sha"]:
            raise RuntimeError("queue release SHA does not match run checkout HEAD")
        if not state["detached"]:
            raise RuntimeError("queue run checkout must be detached")
        if not state["clean"]:
            raise RuntimeError(f"queue run checkout is dirty: {state['porcelain']}")
        return state

    def _verify_environment_binding(self) -> dict:
        path = os.path.realpath(self.manifest["environment_manifest"])
        with open(path, encoding="utf-8") as handle:
            env = json.load(handle)
        if env.get("freeze_sha256") != self.manifest["environment_freeze_sha"]:
            raise RuntimeError("queue environment freeze does not match sealed manifest")
        if env.get("identity_sha256") != self.manifest["environment_identity_sha"]:
            raise RuntimeError("queue runtime/GPU identity does not match sealed manifest")
        freeze_path = os.path.realpath(env["freeze_file"])
        lines = sorted(line.strip() for line in open(freeze_path, encoding="utf-8")
                       if line.strip())
        freeze_sha = sha256_bytes(("".join(line + "\n" for line in lines)).encode("utf-8"))
        if freeze_sha != self.manifest["environment_freeze_sha"]:
            raise RuntimeError("installed-package freeze changed after environment sealing")
        resolved_venv = os.path.realpath(env["venv_path"])
        if not os.path.isdir(resolved_venv):
            raise RuntimeError(f"sealed venv is missing: {resolved_venv}")
        return {"manifest": path_signature(path), "freeze": path_signature(freeze_path),
                "freeze_sha256": freeze_sha, "resolved_venv": resolved_venv}

    def _job_entry(self, job_name: str) -> dict:
        matches = [job for job in self.manifest["jobs"]
                   if (job.get("id") or job.get("name")) == job_name]
        if len(matches) != 1:
            raise RuntimeError(f"queue manifest must contain exactly one job {job_name!r}")
        return matches[0]

    def _input_signatures(self) -> dict:
        result = {}
        for job in self.manifest["jobs"]:
            for path in job.get("inputs", []):
                resolved = path if os.path.isabs(path) else os.path.join(self.repo_root, path)
                result[path] = path_signature(resolved)
        return result

    def boundary(self, job_name: str) -> dict:
        """Fail closed immediately before a job and persist the gate receipt."""
        self._job_entry(job_name)
        if sha256_file(self.manifest_path) != self.manifest_sha256:
            raise RuntimeError("queue manifest changed after admission")
        checkout = self._verify_checkout()
        if checkout["dirty_tree_digest"] != self.launch_checkout["dirty_tree_digest"]:
            raise RuntimeError("run checkout dirty-tree digest changed after admission")
        environment = self._verify_environment_binding()
        if environment != self.initial_environment:
            raise RuntimeError("sealed environment manifest, freeze, or venv changed after admission")
        current_inputs = self._input_signatures()
        if current_inputs != self.initial_inputs:
            raise RuntimeError("declared queue input changed after admission")
        command = [self.roundwatch_bin, "gate-check", self.manifest["round_id"],
                   "--program", self.manifest["program"],
                   "--release-sha", self.manifest["release_sha"]]
        proc = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        try:
            service = json.loads(proc.stdout)
        except json.JSONDecodeError:
            service = {"approved": False, "raw_output": proc.stdout}
        receipt = {
            "schema": "roundwatch_job_boundary_receipt.v1",
            "job": job_name,
            "manifest_path": self.manifest_path,
            "manifest_sha256": self.manifest_sha256,
            "checkout": checkout,
            "environment": environment,
            "resolved_venv": self.resolved_venv,
            "input_signatures": current_inputs,
            "roundwatch_command": command,
            "roundwatch_exit_code": proc.returncode,
            "roundwatch": service,
        }
        out_dir = os.path.realpath(self.manifest["gate_receipts_dir"])
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, f"{len(os.listdir(out_dir)):04d}-{job_name}.json")
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(receipt, handle, indent=2)
            handle.flush(); os.fsync(handle.fileno())
        if proc.returncode != 0 or service.get("approved") is not True:
            raise RuntimeError(f"Roundwatch gate-check rejected job {job_name}: {service}")
        receipt["receipt_path"] = out
        return receipt
