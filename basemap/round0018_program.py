"""Round 0018: the unchanged 30M seed-42 treatment under bf16 autocast.

Identical to Round 0017 in every registered respect (inputs, graph, seed,
500k positive-LR updates, h2048, device_uniform pipeline, thresholds) except
the autocast dtype. The fp16 + GradScaler path stopped after 408,317 successful
updates on a terminal run of consecutive non-finite gradients; its accounting
cannot distinguish scaler overflow from a genuine fp16 model-gradient event.
Round 0018 establishes that bf16 completes the treatment, not why fp16 failed.
bfloat16 uses no loss scaler, so its non-finite-gradient guard stays
first-strike fatal.
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
