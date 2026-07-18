"""Round 0018: the unchanged 30M seed-42 treatment under bf16 autocast.

Identical to Round 0017 in every registered respect (inputs, graph, seed,
500k positive-LR updates, h2048, device_uniform pipeline, thresholds) except
the autocast dtype: fp16 + GradScaler diverged at step 408,317 when the
low-distance gradient singularity exceeded fp16's exponent range (299 amp
overflows, then 100 consecutive at one step — see round-0017 addendum 3).
bfloat16 keeps fp16-class throughput with fp32-class range and needs no loss
scaler, so any non-finite gradient is genuine and stays first-strike fatal.
"""
from __future__ import annotations

import copy

from .artifact_identity import canonical_json, sha256_bytes
from .round0017_program import (NODES, Round0017MaterializedArray,
                                TRAIN_CONFIG as _ROUND0017_TRAIN_CONFIG)

ROUND_ID = "0018"
Round0018MaterializedArray = Round0017MaterializedArray

TRAIN_CONFIG = copy.deepcopy(_ROUND0017_TRAIN_CONFIG)
TRAIN_CONFIG["optimizer"]["use_amp"] = "bf16"
TRAIN_CONFIG_SHA256 = sha256_bytes(canonical_json(TRAIN_CONFIG))
