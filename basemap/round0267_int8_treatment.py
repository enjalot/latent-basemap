"""R0267's treatment closure: the promoted fneg recipe at N=50M, dose ×2, host-int8.

R0267 is the **50M staging rung** of step-5 (owner ruling: flagship dose ×2, staged
through 50M before the 100M commit). It trains the R0265-promoted fneg recipe on the
sealed R0237 50M substrate + exact k15 graph, on the **host-int8 X path** R0266 proved
at the map level (2M), at **dose ×2** and **N=50M**, across **three seeds (42, 43, 44)**.

**The recipe, pinned.** Off R0265's promoted seed-recipe, R0267 moves exactly two axes:

* the **residency + routing** delta R0266 registered: `x_residency` device_fp16 ->
  `host_int8`, `required_pipeline` device -> `host_int8`, and the two
  `expected_pipeline_stamp` mirrors ({x_residency, pipeline}) device -> host_int8.
* the **dose**: ×2 instead of R0265's ×4, and the **scale**: N=50M on the sealed R0237
  substrate + k15 graph instead of R0216's 2M. Everything else -- the umap kernel
  (a=1.9328/b=0.7905, min_dist 0), fneg_weight=1.0 band [0.1,0.4], UNIFORM positive-edge
  sampling -- is R0265's recipe.

**Why a new module, not a delegation to R0266's config.** R0217's `train_config`
hard-locks rows=2M and edges=R0216's sealed count, and R0266's `int8_train_config`
delegates to R0265's ×4 config through it. No parameterised path to a 50M training config
exists in the program (R0237 built the 50M substrate + graph but trained no map). So the
50M ×2 config is built HERE by taking R0265's 2M ×4 template and RETARGETING the
scale-bearing fields (rows, substrate/graph identity, edge count) to the sealed R0237 50M
artifacts and the dose to ×2, then applying R0266's int8 routing delta.

**The guard -- `assert_registered_50m_int8_recipe`.** The whole R0265 recipe proof is
reused on a PROBE copy: the actual config carries the ×2 dose and the host-int8 routing,
so a probe is built with the dose restored to ×4 and the routing restored to device (the
values R0265's monolithic `assert_registered_recipe` pins) and R0265's proof is run on
it -- proving kernel/curve/fneg/uniform-sampling are byte-for-byte R0265's. The ACTUAL
config is then checked for the ×2 dose (derived from the sealed edge count, mirroring
R0265's dose-derivation discipline, so a ×4 dose is caught by the RULE not a literal) and
the int8 residency + routing delta (mirroring R0266). The refusal controls plant a ×4
(R0265) dose, an fp32 residency, a weighted sampler, and fneg-off, and prove the SHIPPED
guard refuses each.

**Import-closure seal.** Reuses R0265/R0266's closure primitives; R0267's closure is
R0265's PLUS the R0266 int8 module PLUS this module, so the seal proves the fneg-merged
core, the int8 routing bytes, and the 50M ×2 recipe bytes all ran.
"""
from __future__ import annotations

import copy
import hashlib
import os
from collections.abc import Mapping, Sequence
from typing import Any

from .artifact_identity import canonical_json, sha256_bytes
from . import round0265_fneg_treatment as R0265
from . import round0266_int8_treatment as R0266
from .round0221_minilm_2m_seed_extension import (
    SEALED_GRAPH_MANIFEST_SIGNATURE,
    SEALED_GRAPH_SIGNATURE,
    SEALED_SUBSTRATE_SIGNATURE,
)
from .round0217_minilm_2m_seed_family import (
    ROWS as ROWS_2M,
    SEALED_DIRECTED_EDGES as SEALED_DIRECTED_EDGES_2M,
    TARGET_POSITIVE_DRAWS_PER_EDGE,
    POSITIVE_ROWS_PER_UPDATE,
    achieved_draws_per_edge,
    dose_quantum,
    performance_windows,
    seed_invariant_sha256,
    successful_updates_for_edges,
)


ROUND_ID = "0267"

INT8_TRAIN_CONFIG_SCHEMA = "round0267-minilm-mixed-50000k-fneg-x2-md000-hostint8-config-v1"
INT8_RECIPE_SCHEMA = "round0267-fneg-x2-md000-hostint8-50m-recipe-v1"
CLOSURE_SCHEMA = "round0267-hostint8-x2-50m-treatment-import-closure-v1"

#: The three seeds of the 50M rung: 1 train + 2 replicates (plan criterion 5). The
#: √n shrinkage on the collapse seed-mean gate is load-bearing, so n=3 is registered.
SEEDS: tuple[int, ...] = (42, 43, 44)
CANONICAL_SEED = 42

#: The 50M rung: dose ×2 (the flagship dose), not R0265's ×4.
DOSE_MULTIPLIER = 2

#: The sealed R0237 50M scale. Rows and the exact k15 fuzzy graph's directed edge count
#: are the scale-bearing constants the 2M -> 50M retarget writes; the dose is DERIVED
#: from the edge count at build time (never the literal), and the literals below exist
#: only so a reviewer sees the intended values and the guard refuses if a derivation
#: lands elsewhere.
ROWS = 50_000_000
SEALED_DIRECTED_EDGES = 1_255_091_326

#: The sealed R0237 substrate + graph capabilities (identity written into the config's
#: graph/input provenance so the retargeted config declares its 50M lineage honestly).
R0237_ROUND_ID = "0237"
R0237_SUBSTRATE_CAPABILITY = "minilm-mixed-50000k-nested-substrate-and-reserves-v1"
R0237_GRAPH_CAPABILITY = "minilm-mixed-50000k-cluster-spill-k15-fuzzy-graph-v1"
R0237_SUBSTRATE_ORDERED_SHA256 = (
    "e7ccf848e0f42e7ac4efa84636b243c9fa75d8e8e0644c21973179cdde1002a9"
)

#: The int8 delta, reused verbatim from R0266.
X_RESIDENCY = R0266.X_RESIDENCY  # "host_int8"
FP32_X_RESIDENCY = R0266.FP32_X_RESIDENCY  # "device_fp16"
INT8_REQUIRED_PIPELINE = R0266.INT8_REQUIRED_PIPELINE  # "host_int8"
FP32_REQUIRED_PIPELINE = R0266.FP32_REQUIRED_PIPELINE  # "device"

CAPABILITY_TEMPLATE = "minilm-mixed-50000k-fneg-x2-md000-hostint8-seed{seed}-r0267-v1"

#: The ×2 horizon and draws-per-edge at the sealed 50M graph, DERIVED at build time; the
#: literals are the published cross-check anchors (2,081,114 base -> 4,162,228 at ×2).
BASE_HORIZON = successful_updates_for_edges(SEALED_DIRECTED_EDGES)  # 2_081_114
HORIZON = DOSE_MULTIPLIER * BASE_HORIZON  # 4_162_228
TARGET_DRAWS_PER_EDGE = DOSE_MULTIPLIER * TARGET_POSITIVE_DRAWS_PER_EDGE

#: R0267's training import closure: R0266's closure (R0265's + the int8 module) PLUS this
#: module. The added member seals the 50M ×2 recipe bytes beside the fneg-merged core and
#: the int8 routing bytes R0266 already sealed.
TRAIN_CLOSURE_MODULES: tuple[str, ...] = (
    *R0266.TRAIN_CLOSURE_MODULES,
    "basemap.round0267_int8_treatment",
)

#: The scale + routing paths R0267 rewrites on top of R0265's ×4 2M template.
INT8_RESIDENCY_PATHS: tuple[tuple[str, ...], ...] = R0266.INT8_RESIDENCY_PATHS

#: The MiniLM feature dimension (the sealed 50M/100M substrates are 384-dim).
DIMENSION = 384

# --------------------------------------------------------------------------- #
# The pre-sealed int8 substrate (the delegate-approved fix for the 50M host-int8
# setup failures): LOAD R0262's sealed 100M host-int8 substrate's first-50M-row
# nested prefix instead of encoding fp32->int8 on the fly at train time. The
# multi-minute on-the-fly encode blocked the node liveness watchdog; the per-row
# int8 encoder makes the first 50M rows a byte-exact nested prefix (offset 0) of
# the 100M substrate, so the LOAD is MAP-IDENTICAL to the encode (offline
# receipt: the port's quantize_int8_rows over the sealed 50M fp32 prefix equals
# R0262's minilm-mixed-100m-int8-v1 first-50M rows bit-for-bit, 0 mismatches).
# --------------------------------------------------------------------------- #

#: R0262's sealed 100M host-int8 substrate — the nested-prefix PARENT of the 50M rung.
R0262_ROUND_ID = "0262"
R0262_INT8_CAPABILITY = "minilm-mixed-100m-int8-v1"
R0262_INT8_ROOT = (
    "/data/latent-basemap/runs/round-0262/artifacts/minilm-mixed-100m-int8-v1"
)
R0262_PARENT_I8_PATH = os.path.join(R0262_INT8_ROOT, "substrate.i8")
R0262_PARENT_SCALES_PATH = os.path.join(R0262_INT8_ROOT, "substrate-scales.f16")
R0262_PARENT_ROWS = 100_000_000
R0262_PARENT_I8_BYTES = R0262_PARENT_ROWS * DIMENSION   # 38_400_000_000 (raw int8)
R0262_PARENT_SCALES_BYTES = R0262_PARENT_ROWS * 2       #    200_000_000 (raw fp16)

#: The 50M nested prefix (offset 0) actually trained on, pinned by CONTENT digest.
SLICE_OFFSET = 0
PREFIX_I8_SHA256 = (
    "49e41c519487f732ef562af0c508a2c2d93cb2427ac3e1cfe9f6054304780d85"
)
PREFIX_SCALES_SHA256 = (
    "4162242e6105ec5083588c2ab3f80c4de406a7ffebfe17300474c270d6c0e96a"
)
PREFIX_I8_BYTES = ROWS * DIMENSION   # 50M * 384 = 19_200_000_000
PREFIX_SCALES_BYTES = ROWS * 2       #             100_000_000

INT8_SLICE_SUBSTRATE_CAPABILITY = "minilm-mixed-50000k-hostint8-nested-prefix-of-100m-v1"
INT8_SLICE_SUBSTRATE_SCHEMA = "round0267-hostint8-50m-nested-prefix-substrate-v1"

#: Streaming chunk for the prefix hash (never materialises the 19.2 GB payload).
_PREFIX_HASH_CHUNK = 1 << 26  # 64 MiB


def sha256_file_prefix(path: str, nbytes: int, *, chunk: int = _PREFIX_HASH_CHUNK) -> str:
    """SHA-256 over EXACTLY the first ``nbytes`` bytes of ``path``, streamed.

    Reads in bounded chunks so the 19.2 GB int8 prefix (or the 100 MB scales
    prefix) is never materialised as one buffer. Raises if the file is shorter
    than ``nbytes`` (the parent must contain the full 50M prefix).
    """
    nbytes = int(nbytes)
    if nbytes < 0:
        raise ValueError("nbytes must be >= 0")
    digest = hashlib.sha256()
    remaining = nbytes
    with open(path, "rb") as handle:
        while remaining > 0:
            block = handle.read(min(int(chunk), remaining))
            if not block:
                raise Round0267RecipeError(
                    f"R0267 int8 prefix hash: {path} is shorter than {nbytes} bytes "
                    f"({nbytes - remaining} read)"
                )
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def int8_slice_prefix_digests(
    i8_path: str, scales_path: str, *, rows: int, dimension: int = DIMENSION
) -> dict[str, Any]:
    """The content digests of the first ``rows`` int8 rows and ``rows`` fp16 scales.

    The int8 file is raw C-contiguous ``rows x dimension`` int8, so the first-``rows``
    prefix is exactly the first ``rows * dimension`` bytes; the scales file is raw
    fp16, so its prefix is the first ``rows * 2`` bytes.
    """
    rows = int(rows)
    dimension = int(dimension)
    return {
        "rows": rows,
        "dimension": dimension,
        "prefix_i8_bytes": rows * dimension,
        "prefix_scales_bytes": rows * 2,
        "prefix_i8_sha256": sha256_file_prefix(i8_path, rows * dimension),
        "prefix_scales_sha256": sha256_file_prefix(scales_path, rows * 2),
    }


def slice_law_block(
    *,
    i8_path: str = R0262_PARENT_I8_PATH,
    scales_path: str = R0262_PARENT_SCALES_PATH,
) -> dict[str, Any]:
    """The SLICE LAW recorded in R0267's int8 substrate manifest.

    Pins the parent (R0262's 100M int8 substrate) by path + size and — the
    load-bearing content binding — pins the 50M bytes actually trained on by the
    prefix digests. The loader re-hashes the prefix and refuses on any mismatch.
    """
    return {
        "law": "nested-prefix-of-100m-int8-substrate-offset-0",
        "parent_artifact": R0262_INT8_CAPABILITY,
        "parent_round": R0262_ROUND_ID,
        "parent_i8_path": str(i8_path),
        "parent_scales_path": str(scales_path),
        "parent_rows": R0262_PARENT_ROWS,
        "parent_i8_bytes": R0262_PARENT_I8_BYTES,
        "parent_scales_bytes": R0262_PARENT_SCALES_BYTES,
        "rows": ROWS,
        "offset": SLICE_OFFSET,
        "dimension": DIMENSION,
        "prefix_i8_bytes": PREFIX_I8_BYTES,
        "prefix_scales_bytes": PREFIX_SCALES_BYTES,
        "prefix_i8_sha256": PREFIX_I8_SHA256,
        "prefix_scales_sha256": PREFIX_SCALES_SHA256,
        "byte_consistency_receipt": (
            "the port's quantize_int8_rows over the sealed R0237 50M fp32 prefix equals "
            "R0262's minilm-mixed-100m-int8-v1 first-50M rows bit-for-bit (0 mismatches, "
            "offline receipt); the per-row int8 encoder makes the first 50M rows a "
            "byte-exact nested prefix (offset 0), so this LOAD is MAP-IDENTICAL to the "
            "on-the-fly fp32->int8 encode it replaces"
        ),
    }


def int8_slice_substrate_manifest_body(*, release_sha: str) -> dict[str, Any]:
    """The R0267 int8 substrate manifest body (sealed by the prepare builder).

    Records the SLICE LAW block (parent digest + the pinned 50M prefix digests)
    so the train node can LOAD R0262's int8 nested prefix file-backed and verify
    the exact bytes it trains on against a sealed pin.
    """
    return {
        "schema": INT8_SLICE_SUBSTRATE_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": str(release_sha),
        "capability": INT8_SLICE_SUBSTRATE_CAPABILITY,
        "rows": ROWS,
        "dimension": DIMENSION,
        "x_residency": X_RESIDENCY,
        "slice_law": slice_law_block(),
        "how_to_read_this": (
            "R0267 trains on the FIRST 50M rows of R0262's sealed 100M host-int8 substrate "
            "instead of re-encoding fp32->int8 on the fly at train time (the multi-minute "
            "on-the-fly encode blocked the liveness watchdog). Because the int8 encoder is "
            "per-row, the first 50M rows are a byte-exact nested prefix (offset 0) of the "
            "100M substrate, so this LOAD is map-identical to the encode. The 50M bytes "
            "trained on are pinned by prefix_i8_sha256 / prefix_scales_sha256; the loader "
            "re-hashes the prefix and refuses on any mismatch (Round0267NodeError)."
        ),
    }


class Round0267RecipeError(RuntimeError):
    """The config is not the registered R0267 50M ×2 host-int8 fneg recipe."""


# --------------------------------------------------------------------------- #
# path helpers (local, so the module is self-contained)
# --------------------------------------------------------------------------- #


def _get_path(value: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    cursor: Any = value
    for key in path:
        if not isinstance(cursor, Mapping) or key not in cursor:
            raise Round0267RecipeError(f"R0267 config is missing {'.'.join(path)}")
        cursor = cursor[key]
    return cursor


def _set_path(value: dict[str, Any], path: tuple[str, ...], replacement: Any) -> None:
    cursor: Any = value
    for key in path[:-1]:
        if not isinstance(cursor, dict) or key not in cursor:
            raise Round0267RecipeError(f"R0267 config is missing {'.'.join(path)}")
        cursor = cursor[key]
    if not isinstance(cursor, dict):
        raise Round0267RecipeError(f"R0267 config is missing {'.'.join(path)}")
    cursor[path[-1]] = replacement


def _set_path_new(value: dict[str, Any], path: tuple[str, ...], replacement: Any) -> None:
    cursor: Any = value
    for key in path[:-1]:
        if not isinstance(cursor, dict) or key not in cursor:
            raise Round0267RecipeError(f"R0267 config is missing parent of {'.'.join(path)}")
        cursor = cursor[key]
    if not isinstance(cursor, dict):
        raise Round0267RecipeError(f"R0267 config parent of {'.'.join(path)} is not a map")
    cursor[path[-1]] = replacement


def capability_for_seed(seed: int) -> str:
    if int(seed) not in SEEDS:
        raise Round0267RecipeError(f"R0267 seed {seed!r} is not a registered 50M rung cell")
    return CAPABILITY_TEMPLATE.format(seed=int(seed))


CAPABILITIES: tuple[str, ...] = tuple(capability_for_seed(seed) for seed in SEEDS)


def exact_cell_id(seed: int) -> str:
    return f"fneg-x2-md000-hostint8-50m-seed{int(seed)}"


# --------------------------------------------------------------------------- #
# the ×2 dose receipt (mirrors R0265.validate_fneg_dose with DOSE_MULTIPLIER=2)
# --------------------------------------------------------------------------- #


def validate_dose_x2(*, updates: int, edge_count: int) -> dict[str, Any]:
    """Pin the ×2 horizon exactly and bound the achieved dose by ×2 quantisation steps.

    DERIVED from the config's own sealed edge count, exactly as R0265 derives the ×4
    horizon: `updates == 2 * successful_updates_for_edges(edge_count)`. A ×4 (or ×1)
    dose is caught by this rule, not by a literal.
    """
    base = successful_updates_for_edges(edge_count)
    exact = DOSE_MULTIPLIER * base
    if int(updates) != exact:
        raise Round0267RecipeError(
            f"R0267 ×2 update horizon {int(updates)} is not "
            f"{DOSE_MULTIPLIER} * successful_updates_for_edges({int(edge_count)}) "
            f"= {DOSE_MULTIPLIER} * {base} = {exact}"
        )
    achieved = achieved_draws_per_edge(updates=exact, edge_count=edge_count)
    quantum = dose_quantum(edge_count)
    target = DOSE_MULTIPLIER * TARGET_POSITIVE_DRAWS_PER_EDGE
    tolerance = DOSE_MULTIPLIER * quantum
    deviation = abs(achieved - target)
    if deviation > tolerance:
        raise Round0267RecipeError(
            f"R0267 achieved ×2 dose {achieved!r} is {deviation:.3e} from the "
            f"registered {target!r}, beyond the {tolerance:.3e} "
            f"({DOSE_MULTIPLIER} x one-update) quantisation bound"
        )
    return {
        "source_round": "0184-x2",
        "dose_multiplier": DOSE_MULTIPLIER,
        "base_successful_updates": base,
        "positive_rows_per_update": POSITIVE_ROWS_PER_UPDATE,
        "active_graph_edges": int(edge_count),
        "successful_updates": exact,
        "dose_rule": (
            f"{DOSE_MULTIPLIER} * ceil(R0184_successful_updates * active_edges / "
            "R0184_directed_edges)"
        ),
        "target_positive_draws_per_edge": target,
        "achieved_positive_draws_per_edge": achieved,
        "dose_quantum_draws_per_edge": quantum,
        "achieved_minus_target": achieved - target,
        "tolerance_basis": (
            f"{DOSE_MULTIPLIER} successful-update quantisation steps; the ×2 horizon "
            "inherits the base ceil rule's rounding at each of the two multiples"
        ),
    }


def _restore_x4_dose(config: dict[str, Any], edges: int) -> None:
    """On a probe copy, restore R0265's ×4 dose so R0265's monolithic proof accepts it."""
    x4 = R0265.validate_fneg_dose(
        updates=R0265.DOSE_MULTIPLIER * successful_updates_for_edges(edges),
        edge_count=edges,
    )
    _set_path(config, ("optimizer", "successful_positive_lr_updates"), x4["successful_updates"])
    _set_path(
        config, ("execution", "target_positive_draws_per_edge"),
        x4["target_positive_draws_per_edge"],
    )
    _set_path(
        config, ("execution", "achieved_positive_draws_per_edge"),
        x4["achieved_positive_draws_per_edge"],
    )
    config["dose_registration"] = x4


# --------------------------------------------------------------------------- #
# building the recipe config -- R0265's 2M ×4 template, retargeted to 50M ×2
# --------------------------------------------------------------------------- #


def int8_50m_train_config(
    *,
    graph_signature: Mapping[str, Any],
    graph_manifest_signature: Mapping[str, Any],
    substrate_signature: Mapping[str, Any],
    graph_edges: int,
    rows: int,
    seed: int = CANONICAL_SEED,
) -> tuple[dict[str, Any], str]:
    """R0265's promoted fneg config, retargeted to the sealed R0237 50M scale + dose ×2 +
    the R0266 host-int8 routing delta, for `seed`.

    Built from R0265's canonical 2M ×4 template (proving the kernel/curve/fneg/uniform
    fields exist), then every scale-bearing field is retargeted to the passed sealed 50M
    signatures and edge count, the dose is recomputed at ×2, and the int8 residency +
    routing delta is applied. `assert_registered_50m_int8_recipe` proves the result.
    """
    if int(seed) not in SEEDS:
        raise Round0267RecipeError(
            f"R0267 is a three-seed rung (42/43/44); got seed {seed!r}"
        )
    if int(rows) != ROWS:
        raise Round0267RecipeError(
            f"R0267 is the 50M rung; got rows {rows!r}, expected {ROWS}"
        )
    if int(graph_edges) != SEALED_DIRECTED_EDGES:
        raise Round0267RecipeError(
            f"R0267 sealed 50M graph has {SEALED_DIRECTED_EDGES} directed edges; "
            f"got {int(graph_edges)}"
        )

    # 1. R0265's canonical 2M ×4 template (kernel/curve/fneg/uniform + device routing).
    template, _sha = R0265.fneg_train_config(
        seed=CANONICAL_SEED,
        graph_signature=SEALED_GRAPH_SIGNATURE,
        graph_manifest_signature=SEALED_GRAPH_MANIFEST_SIGNATURE,
        substrate_signature=SEALED_SUBSTRATE_SIGNATURE,
        graph_edges=SEALED_DIRECTED_EDGES_2M,
        rows=ROWS_2M,
    )
    config = copy.deepcopy(template)

    # 2. Round identity: this is R0265's recipe at 50M ×2 on the int8 path.
    config["round_id"] = ROUND_ID
    config["schema"] = INT8_TRAIN_CONFIG_SCHEMA
    treatment = dict(config.get("treatment") or {})
    treatment["name"] = "fneg-x2-md000-hostint8-50m"
    treatment["recipe_schema"] = INT8_RECIPE_SCHEMA
    treatment["base_recipe_schema"] = R0265.FNEG_RECIPE_SCHEMA
    treatment["int8_base_recipe_schema"] = R0266.INT8_RECIPE_SCHEMA
    treatment["x_residency"] = X_RESIDENCY
    treatment["dose_multiplier"] = DOSE_MULTIPLIER
    treatment["rows"] = ROWS
    config["treatment"] = treatment

    # 3. Scale retarget: rows + the sealed R0237 substrate/graph identity.
    _set_path(config, ("input", "rows"), ROWS)
    _set_path(config, ("input", "substrate_path"), str(substrate_signature["canonical_path"]))
    _set_path(config, ("input", "substrate_sha256"), str(substrate_signature["sha256"]))
    _set_path(config, ("family_invariant", "rows"), ROWS)
    _set_path(config, ("graph", "capability"), R0237_GRAPH_CAPABILITY)
    _set_path(config, ("graph", "source_round"), R0237_ROUND_ID)
    _set_path(config, ("graph", "path"), str(graph_signature["canonical_path"]))
    _set_path(config, ("graph", "sha256"), str(graph_signature["sha256"]))
    _set_path(config, ("graph", "manifest_path"), str(graph_manifest_signature["canonical_path"]))
    _set_path(config, ("graph", "manifest_sha256"), str(graph_manifest_signature["sha256"]))
    _set_path(config, ("graph", "directed_edges"), int(graph_edges))
    stamp = config["execution"]["expected_pipeline_stamp"]
    stamp["negative_sampling"] = f"uniform-{ROWS}-substrate-rows-nonself"
    stamp["valid_canonical_edge_count"] = int(graph_edges)
    stamp["compact_retained_rows"] = ROWS

    # 4. The ×2 dose, DERIVED from the sealed 50M edge count.
    dose = validate_dose_x2(updates=DOSE_MULTIPLIER * successful_updates_for_edges(
        graph_edges), edge_count=graph_edges)
    _set_path(config, ("optimizer", "successful_positive_lr_updates"), dose["successful_updates"])
    _set_path(config, ("execution", "target_positive_draws_per_edge"),
              dose["target_positive_draws_per_edge"])
    _set_path(config, ("execution", "achieved_positive_draws_per_edge"),
              dose["achieved_positive_draws_per_edge"])
    _set_path(config, ("execution", "performance_windows"),
              performance_windows(dose["successful_updates"]))
    config["dose_registration"] = dose

    # 5. The int8 residency + routing delta (R0266's, verbatim). execution.x_residency is
    # a new key R0265's config never carried; the stamp keys exist and are overwritten.
    _set_path_new(config, ("execution", "x_residency"), X_RESIDENCY)
    _set_path(config, ("execution", "expected_pipeline_stamp", "x_residency"), X_RESIDENCY)
    _set_path(config, ("execution", "required_pipeline"), INT8_REQUIRED_PIPELINE)
    _set_path(config, ("execution", "expected_pipeline_stamp", "pipeline"), INT8_REQUIRED_PIPELINE)

    # 6. The 50M three-seed family metadata + this cell's seed-bearing paths.
    _set_path(config, ("seed_family", "seeds"), list(SEEDS))
    _set_path(config, ("seed_family", "cells"), len(SEEDS))
    _set_path(config, ("seed_family", "canonical_seed"), CANONICAL_SEED)
    for path, replacement in _seed_bearing_values(seed).items():
        _set_path(config, path, replacement)

    return config, sha256_bytes(canonical_json(config))


def _seed_bearing_values(seed: int) -> dict[tuple[str, ...], Any]:
    """The nine seed-bearing paths for an R0267 cell (R0265's set, R0267 capability)."""
    seed = int(seed)
    from .round0113_prompt_contrast import NEGATIVE_RNG_SEED_OFFSET

    capability = capability_for_seed(seed)
    return {
        ("seed",): seed,
        ("capability",): capability,
        ("optimizer", "seed"): seed,
        ("optimizer", "positive_rng_seed"): seed,
        ("optimizer", "negative_rng_seed"): seed + NEGATIVE_RNG_SEED_OFFSET,
        ("execution", "expected_pipeline_stamp", "positive_rng_seed"): seed,
        ("execution", "expected_pipeline_stamp", "negative_rng_seed"): (
            seed + NEGATIVE_RNG_SEED_OFFSET
        ),
        ("seed_family", "this_seed"): seed,
        ("seed_family", "this_capability"): capability,
    }


def fneg_seed_invariant_sha256(config: Mapping[str, Any]) -> str:
    """The masked digest of an R0267 config -- R0217's masker over the same nine paths."""
    return seed_invariant_sha256(config)


# --------------------------------------------------------------------------- #
# the recipe invariant -- R0265's proof on a probe, then the ×2 + int8 delta
# --------------------------------------------------------------------------- #


def assert_registered_50m_int8_recipe(config: Mapping[str, Any]) -> dict[str, Any]:
    """Refuse unless `config` is EXACTLY R0265's recipe + dose ×2 + host-int8, at 50M.

    R0265's whole recipe proof (kernel/curve/fneg/uniform-sampling) is reused on a PROBE
    copy with the dose restored to ×4 and the routing restored to device -- the values
    R0265's monolithic guard pins. Then the ACTUAL config is checked for the ×2 dose
    (derived from its own sealed edge count) and the int8 residency + routing delta.
    """
    edges = int(_get_path(config, ("dose_registration", "active_graph_edges")))

    probe = copy.deepcopy(dict(config))
    _restore_x4_dose(probe, edges)
    _set_path(probe, ("execution", "required_pipeline"), FP32_REQUIRED_PIPELINE)
    _set_path(probe, ("execution", "expected_pipeline_stamp", "pipeline"), FP32_REQUIRED_PIPELINE)
    base = R0265.assert_registered_recipe(probe)  # raises Round0265RecipeError on drift

    problems: list[str] = []
    # The ×2 dose, on the ACTUAL config, by the RULE (not a literal).
    horizon = int(_get_path(config, ("optimizer", "successful_positive_lr_updates")))
    expected_horizon = DOSE_MULTIPLIER * successful_updates_for_edges(edges)
    if horizon != expected_horizon:
        problems.append(
            f"optimizer.successful_positive_lr_updates {horizon} != "
            f"{DOSE_MULTIPLIER} * base_horizon = {expected_horizon} (wrong dose; a ×4 "
            "R0265 dose or a ×1 base dose is refused here)"
        )
    expected_draws = DOSE_MULTIPLIER * TARGET_POSITIVE_DRAWS_PER_EDGE
    draws = float(_get_path(config, ("execution", "target_positive_draws_per_edge")))
    if draws != expected_draws:
        problems.append(
            f"execution.target_positive_draws_per_edge {draws!r} != {expected_draws!r} "
            "(the ×2 target)"
        )
    # The int8 residency + routing delta (R0266's, on the ACTUAL config).
    if str(_get_path(config, ("execution", "x_residency"))) != X_RESIDENCY:
        problems.append(
            f"execution.x_residency != {X_RESIDENCY!r} (a device_fp16/auto value silently "
            "runs the fp32 device path, hardware-infeasible at 50M and unproven)"
        )
    if str(_get_path(config, ("execution", "required_pipeline"))) != INT8_REQUIRED_PIPELINE:
        problems.append(
            f"execution.required_pipeline != {INT8_REQUIRED_PIPELINE!r} (R0265's 'device' "
            "pin makes core.fit refuse the host_int8 stamp)"
        )
    stamp = _get_path(config, ("execution", "expected_pipeline_stamp"))
    if not isinstance(stamp, Mapping) or stamp.get("x_residency") != X_RESIDENCY:
        problems.append(
            f"execution.expected_pipeline_stamp.x_residency != {X_RESIDENCY!r}"
        )
    if not isinstance(stamp, Mapping) or stamp.get("pipeline") != INT8_REQUIRED_PIPELINE:
        problems.append(
            f"execution.expected_pipeline_stamp.pipeline != {INT8_REQUIRED_PIPELINE!r}"
        )
    # The scale, on the ACTUAL config.
    if int(_get_path(config, ("input", "rows"))) != ROWS:
        problems.append(f"input.rows != {ROWS} (R0267 is the 50M rung)")
    if edges != SEALED_DIRECTED_EDGES:
        problems.append(
            f"dose_registration.active_graph_edges {edges} != {SEALED_DIRECTED_EDGES} "
            "(the sealed R0237 50M k15 graph edge count)"
        )
    if problems:
        raise Round0267RecipeError(
            "R0267 config is not the registered 50M ×2 host-int8 fneg recipe: "
            + "; ".join(problems)
        )

    dose = validate_dose_x2(updates=horizon, edge_count=edges)
    recipe = dict(base)
    recipe["recipe_schema"] = INT8_RECIPE_SCHEMA
    recipe["base_recipe_schema"] = R0265.FNEG_RECIPE_SCHEMA
    recipe["int8_base_recipe_schema"] = R0266.INT8_RECIPE_SCHEMA
    recipe["dose_multiplier"] = DOSE_MULTIPLIER
    recipe["rows"] = ROWS
    recipe["successful_positive_lr_updates"] = horizon
    recipe["base_horizon"] = successful_updates_for_edges(edges)
    recipe["target_positive_draws_per_edge"] = expected_draws
    recipe["achieved_positive_draws_per_edge"] = dose["achieved_positive_draws_per_edge"]
    recipe["x_residency"] = X_RESIDENCY
    recipe["expected_pipeline_stamp_x_residency"] = X_RESIDENCY
    recipe["directed_edges"] = edges
    recipe["seed_invariant_sha256"] = fneg_seed_invariant_sha256(config)
    recipe["delta_over_r0265"] = (
        "dose ×4->×2, scale 2M->50M (sealed R0237 substrate+graph), and R0266's "
        "residency+routing: x_residency device_fp16->host_int8, required_pipeline "
        "device->host_int8, expected_pipeline_stamp {x_residency, pipeline} device->host_int8"
    )
    return recipe


def _honest_50m_config(seed: int = CANONICAL_SEED) -> dict[str, Any]:
    """A canonical honest 50M ×2 config on synthetic sealed 50M signatures (no artifact read)."""
    config, _sha = int8_50m_train_config(
        graph_signature={"canonical_path": "/sealed/50m/edges-k15-fuzzy.npz", "sha256": "b" * 64},
        graph_manifest_signature={"canonical_path": "/sealed/50m/qualified-graph.json", "sha256": "c" * 64},
        substrate_signature={"canonical_path": "/sealed/50m/substrate.f32.npy", "sha256": "a" * 64},
        graph_edges=SEALED_DIRECTED_EDGES,
        rows=ROWS,
        seed=seed,
    )
    return config


def recipe_refusal_controls() -> dict[str, Any]:
    """Plant the ×2/int8 delta defects AND R0265 recipe defects against the SHIPPED guard.

    Every control calls the shipped `assert_registered_50m_int8_recipe`; none reimplements
    it. The dose plant (a ×4 R0265 dose) proves the ×2 rule is enforced; the residency and
    weighted plants prove the int8 delta and the delegated R0265 uniform proof still fire.
    """
    honest = _honest_50m_config()
    controls: list[dict[str, Any]] = []

    def _plant(name: str, description: str, mutate) -> None:
        planted = copy.deepcopy(honest)
        mutate(planted)
        refused = False
        error = None
        try:
            assert_registered_50m_int8_recipe(planted)
        except (Round0267RecipeError, R0265.Round0265RecipeError) as raised:
            refused = True
            error = f"{type(raised).__name__}: {raised}"
        controls.append({
            "control": name,
            "plants": description,
            "shipped_predicate_refused": refused,
            "shipped_predicate_error": error,
        })

    def _wrong_dose_x4(cfg: dict[str, Any]) -> None:
        # R0265's ×4 dose instead of the ×2 rung dose -- refused by the ×2 rule.
        x4 = R0265.validate_fneg_dose(
            updates=R0265.DOSE_MULTIPLIER * successful_updates_for_edges(SEALED_DIRECTED_EDGES),
            edge_count=SEALED_DIRECTED_EDGES,
        )
        _set_path(cfg, ("optimizer", "successful_positive_lr_updates"), x4["successful_updates"])
        _set_path(cfg, ("execution", "target_positive_draws_per_edge"),
                  x4["target_positive_draws_per_edge"])
        cfg["dose_registration"] = x4

    _plant("wrong_dose_x4",
           "R0265's ×4 dose instead of the ×2 rung dose -- refused by the ×2 rule",
           _wrong_dose_x4)
    _plant("x_residency_device_fp16",
           "execution.x_residency left at the fp32 default device_fp16 -- silent fp32 path",
           lambda c: _set_path(c, ("execution", "x_residency"), FP32_X_RESIDENCY))
    _plant("required_pipeline_device",
           "execution.required_pipeline left at R0265's 'device' pin -- core.fit refuses the "
           "host_int8 stamp",
           lambda c: _set_path(c, ("execution", "required_pipeline"), FP32_REQUIRED_PIPELINE))
    _plant("stamp_x_residency_device_fp16",
           "declared expected_pipeline_stamp.x_residency still device_fp16 -- dishonest stamp",
           lambda c: _set_path(c, ("execution", "expected_pipeline_stamp", "x_residency"),
                               FP32_X_RESIDENCY))
    _plant("base_recipe_weighted_sampling_on",
           "weighted_edge_sampling=True -- R0217's fuzzy sampler, refused through the delegated proof",
           lambda c: _set_path(c, ("optimizer", "weighted_edge_sampling"), True))
    _plant("base_recipe_fneg_off",
           "fneg_weight=0.0 -- an R0265 recipe defect, refused through the delegated proof",
           lambda c: _set_path(c, ("optimizer", "fneg_weight"), 0.0))

    honest_refused = False
    honest_error = None
    try:
        assert_registered_50m_int8_recipe(honest)
    except (Round0267RecipeError, R0265.Round0265RecipeError) as raised:
        honest_refused = True
        honest_error = f"{type(raised).__name__}: {raised}"
    return {
        "controls": controls,
        "planted": len(controls),
        "every_planted_defect_was_refused": all(
            item["shipped_predicate_refused"] for item in controls
        ),
        "the_honest_recipe_still_passes": not honest_refused,
        "honest_error": honest_error,
        "note": (
            "every control calls the SHIPPED assert_registered_50m_int8_recipe; none "
            "reimplements it. The ×4 dose plant exercises the ×2 rule; the residency and "
            "weighted/fneg plants exercise the int8 delta and the delegated R0265 proof."
        ),
    }


# --------------------------------------------------------------------------- #
# the import-closure seal (reused R0265/R0266 primitives, R0267's module list)
# --------------------------------------------------------------------------- #

file_sha256 = R0265.file_sha256


def runtime_closure_hashes(
    modules: Sequence[str] = TRAIN_CLOSURE_MODULES,
) -> dict[str, dict[str, Any]]:
    """Hash the files behind R0267's closure (R0266's + this module)."""
    return R0265.runtime_closure_hashes(modules)


def assert_runtime_closure_matches_seal(
    *,
    sealed: Mapping[str, Any],
    observed: Mapping[str, Mapping[str, Any]],
    modules: Sequence[str] = TRAIN_CLOSURE_MODULES,
) -> dict[str, Any]:
    """Refuse unless every module in R0267's registered closure ran the sealed bytes."""
    return R0265.assert_runtime_closure_matches_seal(
        sealed=sealed, observed=observed, modules=modules
    )


def treatment_closure_controls(
    *, sealed: Mapping[str, Any], observed: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Plant closure defects against the SHIPPED R0267 predicate. Every control calls it."""
    honest_observed = {name: dict(value) for name, value in observed.items()}
    controls: list[dict[str, Any]] = []

    def _run(name: str, description: str, **kwargs: Any) -> None:
        refused = False
        error = None
        try:
            assert_runtime_closure_matches_seal(**kwargs)
        except R0265.Round0265TreatmentError as raised:
            refused = True
            error = f"{type(raised).__name__}: {raised}"
        controls.append({
            "control": name,
            "plants": description,
            "shipped_predicate_refused": refused,
            "shipped_predicate_error": error,
        })

    victim = "basemap.round0267_int8_treatment"  # the 50M ×2 recipe bytes
    mutated = {name: dict(value) for name, value in honest_observed.items()}
    mutated[victim]["sha256"] = "0" * 64
    _run("content_drift", "the R0267 50M ×2 recipe ran bytes that are not the sealed ones",
         sealed=sealed, observed=mutated)

    dropped = {name: dict(value) for name, value in honest_observed.items() if name != victim}
    _run("missing_module", "the R0267 50M ×2 recipe module is absent from the runtime map",
         sealed=sealed, observed=dropped)

    extra = {name: dict(value) for name, value in honest_observed.items()}
    extra["basemap.round0267_missing"] = {"path": __file__, "bytes": 0, "sha256": "1" * 64}
    _run("extra_module", "the runtime map carries a module the seal does not",
         sealed=sealed, observed=extra)

    malformed = {name: dict(value) for name, value in honest_observed.items()}
    malformed[victim]["sha256"] = "not-a-digest"
    _run("malformed_digest", "a digest that is not a SHA-256 hex string",
         sealed=sealed, observed=malformed)

    _run("empty_closure", "an EMPTY registered closure -- the vacuous accept",
         sealed={"files": {}}, observed={}, modules=())

    honest_refused = False
    honest_error = None
    try:
        assert_runtime_closure_matches_seal(sealed=sealed, observed=honest_observed)
    except R0265.Round0265TreatmentError as raised:
        honest_refused = True
        honest_error = f"{type(raised).__name__}: {raised}"
    return {
        "controls": controls,
        "planted": len(controls),
        "every_planted_defect_was_refused": all(
            item["shipped_predicate_refused"] for item in controls
        ),
        "the_honest_closure_still_passes": not honest_refused,
        "honest_error": honest_error,
    }


__all__ = [
    "BASE_HORIZON",
    "CANONICAL_SEED",
    "CAPABILITIES",
    "CAPABILITY_TEMPLATE",
    "CLOSURE_SCHEMA",
    "DIMENSION",
    "DOSE_MULTIPLIER",
    "HORIZON",
    "INT8_SLICE_SUBSTRATE_CAPABILITY",
    "INT8_SLICE_SUBSTRATE_SCHEMA",
    "PREFIX_I8_SHA256",
    "PREFIX_SCALES_SHA256",
    "R0262_INT8_CAPABILITY",
    "R0262_PARENT_I8_PATH",
    "R0262_PARENT_SCALES_PATH",
    "R0262_ROUND_ID",
    "SLICE_OFFSET",
    "int8_slice_prefix_digests",
    "int8_slice_substrate_manifest_body",
    "sha256_file_prefix",
    "slice_law_block",
    "INT8_RECIPE_SCHEMA",
    "INT8_RESIDENCY_PATHS",
    "INT8_TRAIN_CONFIG_SCHEMA",
    "R0237_GRAPH_CAPABILITY",
    "R0237_SUBSTRATE_CAPABILITY",
    "R0237_SUBSTRATE_ORDERED_SHA256",
    "ROUND_ID",
    "ROWS",
    "SEALED_DIRECTED_EDGES",
    "SEEDS",
    "TARGET_DRAWS_PER_EDGE",
    "TRAIN_CLOSURE_MODULES",
    "X_RESIDENCY",
    "Round0267RecipeError",
    "assert_registered_50m_int8_recipe",
    "assert_runtime_closure_matches_seal",
    "capability_for_seed",
    "exact_cell_id",
    "file_sha256",
    "fneg_seed_invariant_sha256",
    "int8_50m_train_config",
    "recipe_refusal_controls",
    "runtime_closure_hashes",
    "treatment_closure_controls",
    "validate_dose_x2",
]
