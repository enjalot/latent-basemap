"""R0268's treatment closure: the promoted fneg recipe at N=100M, dose ×2, host-int8.

R0268 is the **100M ×2 host-int8 FLAGSHIP** of step-5 (pre-registration
`latent-labs/basemap-100m/plan-100m-flagship-2026-08-17.md`, delegate-signed n=3). It
trains the R0265-promoted fneg recipe on the sealed R0238 100M substrate + exact R0243 k15
graph, on the **host-int8 X path** R0266 proved at the map level and R0267 proved at 50M,
at **dose ×2** and **N=100M**, across **three FRESH seeds (42, 43, 44)**.

**A FRESH-TRAIN round.** Unlike R0267 (whose 50M correction saga salvaged/bound cells),
R0268 trains all three seeds from scratch — no salvage, no bind. Everything binds from the
established 50M/R0262 precedent; nothing new is invented.

**The recipe, pinned.** Off R0265's promoted seed-recipe, R0268 moves exactly two axes
relative to R0265:

* the **residency + routing** delta R0266 registered: `x_residency` device_fp16 ->
  `host_int8`, `required_pipeline` device -> `host_int8`, and the two
  `expected_pipeline_stamp` mirrors ({x_residency, pipeline}) device -> host_int8.
* the **dose**: ×2 instead of R0265's ×4, and the **scale**: N=100M on the sealed R0238
  substrate + R0243 k15 graph instead of R0216's 2M. Everything else -- the umap kernel
  (a=1.9328/b=0.7905, min_dist 0), fneg_weight=1.0 band [0.1,0.4], UNIFORM positive-edge
  sampling -- is R0265's recipe.

**The guard -- `assert_registered_100m_int8_recipe`.** The whole R0265 recipe proof is
reused on a PROBE copy: the actual config carries the ×2 dose and the host-int8 routing,
so a probe is built with the dose restored to ×4 and the routing restored to device (the
values R0265's monolithic `assert_registered_recipe` pins) and R0265's proof is run on
it -- proving kernel/curve/fneg/uniform-sampling are byte-for-byte R0265's. The ACTUAL
config is then checked for the ×2 dose (DERIVED from the sealed R0243 edge count, mirroring
R0265/R0267's dose-derivation discipline, so a ×4 dose is caught by the RULE not a literal)
and the int8 residency + routing delta (mirroring R0266).

**The int8 X: a PRE-SEALED FULL-FILE load.** R0268 LOADS R0262's sealed 100M host-int8
substrate -- the WHOLE file this time (R0267 sliced a 50M prefix of the same 100M int8; at
100M the file IS the substrate, so the slice-law is dropped and only the full-file digest
bind remains). `substrate.i8` (38,400,000,000 B = 100M×384 int8) + `substrate-scales.f16`
(200,000,000 B = 100M×2). Digest-bound to the R0262 identity manifest.

**Import-closure seal.** Reuses R0265/R0266's closure primitives; R0268's closure is
R0266's PLUS this module, so the seal proves the fneg-merged core, the int8 routing bytes,
and the 100M ×2 recipe bytes all ran.
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


ROUND_ID = "0268"

INT8_TRAIN_CONFIG_SCHEMA = "round0268-minilm-mixed-100000k-fneg-x2-md000-hostint8-config-v1"
INT8_RECIPE_SCHEMA = "round0268-fneg-x2-md000-hostint8-100m-recipe-v1"
CLOSURE_SCHEMA = "round0268-hostint8-x2-100m-treatment-import-closure-v1"

#: The three seeds of the 100M flagship: 1 train + 2 replicates (plan criterion 5). The
#: √n shrinkage on the collapse seed-mean gate is load-bearing, so n=3 is registered.
SEEDS: tuple[int, ...] = (42, 43, 44)
CANONICAL_SEED = 42

#: The flagship dose ×2 (not R0265's ×4).
DOSE_MULTIPLIER = 2

#: The sealed R0238 100M scale. Rows and the exact R0243 k15 fuzzy graph's directed edge
#: count are the scale-bearing constants the 2M -> 100M retarget writes; the dose is DERIVED
#: from the edge count at build time (never the literal), and the literals below exist only
#: so a reviewer sees the intended values and the guard refuses if a derivation lands
#: elsewhere.
ROWS = 100_000_000
SEALED_DIRECTED_EDGES = 2_511_103_254

#: The MiniLM feature dimension (the sealed 100M substrates are 384-dim).
DIMENSION = 384

#: The sealed R0238 substrate capabilities (identity written into the config's input
#: provenance so the retargeted config declares its 100M lineage honestly).
R0238_ROUND_ID = "0238"
R0238_SUBSTRATE_CAPABILITY = "minilm-mixed-100000k-nested-substrate-and-reserves-v1"
R0238_SUBSTRATE_ORDERED_SHA256 = (
    "f3f1b4b75d2612683a275c0b483fac948527d58e9f9836c165572fbe147ab645"
)

#: The sealed R0243 100M k15 fuzzy graph capabilities. R0243 seals under a
#: `fuzzy-graph.json` manifest that ships the graph as four streamed .npy/.npz members
#: (edges src/dst/wts + a scalar header), NOT a single .npz with an inner `graph`
#: signature; the node's graph binding is retargeted accordingly.
R0243_ROUND_ID = "0243"
R0243_GRAPH_CAPABILITY = "minilm-mixed-100000k-cluster-spill-k15-fuzzy-graph-v1"

#: The int8 delta, reused verbatim from R0266.
X_RESIDENCY = R0266.X_RESIDENCY  # "host_int8"
FP32_X_RESIDENCY = R0266.FP32_X_RESIDENCY  # "device_fp16"
INT8_REQUIRED_PIPELINE = R0266.INT8_REQUIRED_PIPELINE  # "host_int8"
FP32_REQUIRED_PIPELINE = R0266.FP32_REQUIRED_PIPELINE  # "device"

CAPABILITY_TEMPLATE = "minilm-mixed-100000k-fneg-x2-md000-hostint8-seed{seed}-r0268-v1"

#: The ×2 horizon and draws-per-edge at the sealed R0243 100M graph, DERIVED at build time;
#: the literals are the published cross-check anchors (base 4,163,754 -> 8,327,508 at ×2).
#: NOTE the 100M edge count 2,511,103,254 is NOT exactly 2× the 50M count 1,255,091,326
#: (2× = 2,510,182,652; ~920K / 0.037% apart), and the base's ceil crossed a boundary, so
#: the exact horizon is 8,327,508 (+3,052 over a naive 2× of 50M's 4,162,228). The
#: prepare-time computation from the sealed edge count GOVERNS.
BASE_HORIZON = successful_updates_for_edges(SEALED_DIRECTED_EDGES)  # 4_163_754
HORIZON = DOSE_MULTIPLIER * BASE_HORIZON  # 8_327_508
TARGET_DRAWS_PER_EDGE = DOSE_MULTIPLIER * TARGET_POSITIVE_DRAWS_PER_EDGE

#: R0268's training import closure: R0266's closure (R0265's + the int8 module) PLUS this
#: module. The added member seals the 100M ×2 recipe bytes beside the fneg-merged core and
#: the int8 routing bytes R0266 already sealed.
TRAIN_CLOSURE_MODULES: tuple[str, ...] = (
    *R0266.TRAIN_CLOSURE_MODULES,
    "basemap.round0268_int8_treatment",
)

#: The scale + routing paths R0268 rewrites on top of R0265's ×4 2M template.
INT8_RESIDENCY_PATHS: tuple[tuple[str, ...], ...] = R0266.INT8_RESIDENCY_PATHS

# --------------------------------------------------------------------------- #
# The pre-sealed int8 substrate (the delegate-approved host-int8 fix, proven at 50M
# in R0267): LOAD R0262's sealed 100M host-int8 substrate instead of encoding
# fp32->int8 on the fly at train time (the multi-minute on-the-fly encode blocked
# the node liveness watchdog). At 100M the file IS the whole substrate — no prefix
# slice — so R0267's slice-law is dropped and only the FULL-FILE digest bind remains.
# --------------------------------------------------------------------------- #

#: R0262's sealed 100M host-int8 substrate — loaded WHOLE (offset 0, full 100M rows).
R0262_ROUND_ID = "0262"
R0262_INT8_CAPABILITY = "minilm-mixed-100m-int8-v1"
R0262_INT8_ROOT = (
    "/data/latent-basemap/runs/round-0262/artifacts/minilm-mixed-100m-int8-v1"
)
R0262_I8_PATH = os.path.join(R0262_INT8_ROOT, "substrate.i8")
R0262_SCALES_PATH = os.path.join(R0262_INT8_ROOT, "substrate-scales.f16")
R0262_ROWS = 100_000_000
R0262_I8_BYTES = R0262_ROWS * DIMENSION   # 38_400_000_000 (raw int8)
R0262_SCALES_BYTES = R0262_ROWS * 2       #    200_000_000 (raw fp16)

#: The R0262 identity manifest the full-file substrate is digest-bound to.
R0262_IDENTITY_MANIFEST = (
    "/data/latent-basemap/runs/round-0262/queue/artifacts/"
    "round0262-hundred-m-int8-substrate-and-fidelity-v1/quantise0262-int8-substrate.json"
)
R0262_IDENTITY_SCHEMA = "round0262-hundred-m-int8-substrate-and-fidelity-v1"

#: The FULL-FILE content digests of R0262's 100M int8 substrate + fp16 scales, pinned so
#: the loader re-hashes the whole file and refuses on any mismatch (the load-bearing
#: content bind — size alone would not prove content). Computed offline over the sealed
#: R0262 artifact bytes (streamed sha256).
FULL_I8_SHA256 = (
    "b26150c644a907193e34e3cd83dd57f5aeef20a1f94474e47b4088db40230b52"
)
FULL_SCALES_SHA256 = (
    "afcc9991b0def879172f0b7e9ecad38664284da96887eee1f56bdc275d4646f5"
)

INT8_SUBSTRATE_CAPABILITY = "minilm-mixed-100m-hostint8-full-file-v1"
INT8_SUBSTRATE_SCHEMA = "round0268-hostint8-100m-full-file-substrate-v1"

#: Streaming chunk for the full-file hash (never materialises the 38.4 GB payload).
_HASH_CHUNK = 1 << 26  # 64 MiB


def sha256_file(path: str, *, chunk: int = _HASH_CHUNK) -> str:
    """SHA-256 over the ENTIRE file at ``path``, streamed in bounded chunks.

    Reads in bounded chunks so the 38.4 GB int8 file (or the 200 MB scales file) is
    never materialised as one buffer.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(int(chunk)), b""):
            digest.update(block)
    return digest.hexdigest()


def int8_full_digests(
    i8_path: str, scales_path: str, *, rows: int, dimension: int = DIMENSION
) -> dict[str, Any]:
    """The full-file content digests + byte sizes of the int8 substrate and fp16 scales.

    The int8 file is raw C-contiguous ``rows x dimension`` int8, so its size is
    ``rows * dimension`` bytes; the scales file is raw fp16, ``rows * 2`` bytes.
    """
    rows = int(rows)
    dimension = int(dimension)
    return {
        "rows": rows,
        "dimension": dimension,
        "i8_bytes": rows * dimension,
        "scales_bytes": rows * 2,
        "i8_sha256": sha256_file(i8_path),
        "scales_sha256": sha256_file(scales_path),
    }


def full_file_law_block(
    *,
    i8_path: str = R0262_I8_PATH,
    scales_path: str = R0262_SCALES_PATH,
) -> dict[str, Any]:
    """The FULL-FILE LAW recorded in R0268's int8 substrate manifest.

    Pins R0262's 100M int8 substrate by path + size and — the load-bearing content
    binding — by the whole-file digests. The loader re-hashes the whole file and refuses
    on any mismatch. Unlike R0267's slice-law there is no offset/prefix: R0268 trains on
    the whole file (all 100M rows), so this is a full-file identity, not a nested prefix.
    """
    return {
        "law": "full-file-of-100m-int8-substrate",
        "parent_artifact": R0262_INT8_CAPABILITY,
        "parent_round": R0262_ROUND_ID,
        "parent_identity_manifest": R0262_IDENTITY_MANIFEST,
        "i8_path": str(i8_path),
        "scales_path": str(scales_path),
        "rows": ROWS,
        "offset": 0,
        "dimension": DIMENSION,
        "i8_bytes": R0262_I8_BYTES,
        "scales_bytes": R0262_SCALES_BYTES,
        "i8_sha256": FULL_I8_SHA256,
        "scales_sha256": FULL_SCALES_SHA256,
        "note": (
            "R0268 trains on the WHOLE R0262 100M host-int8 substrate (all 100M rows) "
            "instead of re-encoding fp32->int8 on the fly at train time (the multi-minute "
            "on-the-fly encode blocked the liveness watchdog). This is the design-fix path "
            "R0267 proved at 50M, here at full 100M without the prefix slice. The 38.4 GB "
            "int8 + 200 MB scales are pinned by i8_sha256 / scales_sha256; the loader "
            "re-hashes the whole file and refuses on any mismatch (Round0268NodeError)."
        ),
    }


def int8_full_substrate_manifest_body(*, release_sha: str) -> dict[str, Any]:
    """The R0268 int8 substrate manifest body (sealed by the prepare builder).

    Records the FULL-FILE LAW block (parent identity + the pinned whole-file digests) so
    the train node can LOAD R0262's int8 substrate file-backed and verify the exact bytes
    it trains on against a sealed pin.
    """
    return {
        "schema": INT8_SUBSTRATE_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": str(release_sha),
        "capability": INT8_SUBSTRATE_CAPABILITY,
        "rows": ROWS,
        "dimension": DIMENSION,
        "x_residency": X_RESIDENCY,
        "full_file_law": full_file_law_block(),
        "how_to_read_this": (
            "R0268 trains on the WHOLE R0262 sealed 100M host-int8 substrate (38.4 GB int8 "
            "+ 200 MB fp16 scales) instead of re-encoding fp32->int8 on the fly at train "
            "time. The bytes trained on are pinned by i8_sha256 / scales_sha256; the loader "
            "re-hashes the whole file and refuses on any mismatch (Round0268NodeError). No "
            "prefix slice: at 100M the file IS the substrate (R0267 sliced a 50M prefix of "
            "this same file — here it is loaded whole)."
        ),
    }


class Round0268RecipeError(RuntimeError):
    """The config is not the registered R0268 100M ×2 host-int8 fneg recipe."""


# --------------------------------------------------------------------------- #
# path helpers (local, so the module is self-contained)
# --------------------------------------------------------------------------- #


def _get_path(value: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    cursor: Any = value
    for key in path:
        if not isinstance(cursor, Mapping) or key not in cursor:
            raise Round0268RecipeError(f"R0268 config is missing {'.'.join(path)}")
        cursor = cursor[key]
    return cursor


def _set_path(value: dict[str, Any], path: tuple[str, ...], replacement: Any) -> None:
    cursor: Any = value
    for key in path[:-1]:
        if not isinstance(cursor, dict) or key not in cursor:
            raise Round0268RecipeError(f"R0268 config is missing {'.'.join(path)}")
        cursor = cursor[key]
    if not isinstance(cursor, dict):
        raise Round0268RecipeError(f"R0268 config is missing {'.'.join(path)}")
    cursor[path[-1]] = replacement


def _set_path_new(value: dict[str, Any], path: tuple[str, ...], replacement: Any) -> None:
    cursor: Any = value
    for key in path[:-1]:
        if not isinstance(cursor, dict) or key not in cursor:
            raise Round0268RecipeError(f"R0268 config is missing parent of {'.'.join(path)}")
        cursor = cursor[key]
    if not isinstance(cursor, dict):
        raise Round0268RecipeError(f"R0268 config parent of {'.'.join(path)} is not a map")
    cursor[path[-1]] = replacement


def capability_for_seed(seed: int) -> str:
    if int(seed) not in SEEDS:
        raise Round0268RecipeError(f"R0268 seed {seed!r} is not a registered 100M flagship cell")
    return CAPABILITY_TEMPLATE.format(seed=int(seed))


CAPABILITIES: tuple[str, ...] = tuple(capability_for_seed(seed) for seed in SEEDS)


def exact_cell_id(seed: int) -> str:
    return f"fneg-x2-md000-hostint8-100m-seed{int(seed)}"


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
        raise Round0268RecipeError(
            f"R0268 ×2 update horizon {int(updates)} is not "
            f"{DOSE_MULTIPLIER} * successful_updates_for_edges({int(edge_count)}) "
            f"= {DOSE_MULTIPLIER} * {base} = {exact}"
        )
    achieved = achieved_draws_per_edge(updates=exact, edge_count=edge_count)
    quantum = dose_quantum(edge_count)
    target = DOSE_MULTIPLIER * TARGET_POSITIVE_DRAWS_PER_EDGE
    tolerance = DOSE_MULTIPLIER * quantum
    deviation = abs(achieved - target)
    if deviation > tolerance:
        raise Round0268RecipeError(
            f"R0268 achieved ×2 dose {achieved!r} is {deviation:.3e} from the "
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
# building the recipe config -- R0265's 2M ×4 template, retargeted to 100M ×2
# --------------------------------------------------------------------------- #


def int8_100m_train_config(
    *,
    graph_signature: Mapping[str, Any],
    graph_manifest_signature: Mapping[str, Any],
    substrate_signature: Mapping[str, Any],
    graph_edges: int,
    rows: int,
    seed: int = CANONICAL_SEED,
) -> tuple[dict[str, Any], str]:
    """R0265's promoted fneg config, retargeted to the sealed R0238 100M scale + dose ×2 +
    the R0266 host-int8 routing delta, for `seed`.

    Built from R0265's canonical 2M ×4 template (proving the kernel/curve/fneg/uniform
    fields exist), then every scale-bearing field is retargeted to the passed sealed 100M
    signatures and edge count, the dose is recomputed at ×2, and the int8 residency +
    routing delta is applied. `assert_registered_100m_int8_recipe` proves the result.
    """
    if int(seed) not in SEEDS:
        raise Round0268RecipeError(
            f"R0268 is a three-seed flagship (42/43/44); got seed {seed!r}"
        )
    if int(rows) != ROWS:
        raise Round0268RecipeError(
            f"R0268 is the 100M flagship; got rows {rows!r}, expected {ROWS}"
        )
    if int(graph_edges) != SEALED_DIRECTED_EDGES:
        raise Round0268RecipeError(
            f"R0268 sealed R0243 100M graph has {SEALED_DIRECTED_EDGES} directed edges; "
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

    # 2. Round identity: this is R0265's recipe at 100M ×2 on the int8 path.
    config["round_id"] = ROUND_ID
    config["schema"] = INT8_TRAIN_CONFIG_SCHEMA
    treatment = dict(config.get("treatment") or {})
    treatment["name"] = "fneg-x2-md000-hostint8-100m"
    treatment["recipe_schema"] = INT8_RECIPE_SCHEMA
    treatment["base_recipe_schema"] = R0265.FNEG_RECIPE_SCHEMA
    treatment["int8_base_recipe_schema"] = R0266.INT8_RECIPE_SCHEMA
    treatment["x_residency"] = X_RESIDENCY
    treatment["dose_multiplier"] = DOSE_MULTIPLIER
    treatment["rows"] = ROWS
    config["treatment"] = treatment

    # 3. Scale retarget: rows + the sealed R0238 substrate / R0243 graph identity.
    _set_path(config, ("input", "rows"), ROWS)
    _set_path(config, ("input", "substrate_path"), str(substrate_signature["canonical_path"]))
    _set_path(config, ("input", "substrate_sha256"), str(substrate_signature["sha256"]))
    _set_path(config, ("family_invariant", "rows"), ROWS)
    _set_path(config, ("graph", "capability"), R0243_GRAPH_CAPABILITY)
    _set_path(config, ("graph", "source_round"), R0243_ROUND_ID)
    _set_path(config, ("graph", "path"), str(graph_signature["canonical_path"]))
    _set_path(config, ("graph", "sha256"), str(graph_signature["sha256"]))
    _set_path(config, ("graph", "manifest_path"), str(graph_manifest_signature["canonical_path"]))
    _set_path(config, ("graph", "manifest_sha256"), str(graph_manifest_signature["sha256"]))
    _set_path(config, ("graph", "directed_edges"), int(graph_edges))
    stamp = config["execution"]["expected_pipeline_stamp"]
    stamp["negative_sampling"] = f"uniform-{ROWS}-substrate-rows-nonself"
    stamp["valid_canonical_edge_count"] = int(graph_edges)
    stamp["compact_retained_rows"] = ROWS

    # 4. The ×2 dose, DERIVED from the sealed R0243 100M edge count.
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

    # 6. The 100M three-seed family metadata + this cell's seed-bearing paths.
    _set_path(config, ("seed_family", "seeds"), list(SEEDS))
    _set_path(config, ("seed_family", "cells"), len(SEEDS))
    _set_path(config, ("seed_family", "canonical_seed"), CANONICAL_SEED)
    for path, replacement in _seed_bearing_values(seed).items():
        _set_path(config, path, replacement)

    return config, sha256_bytes(canonical_json(config))


def _seed_bearing_values(seed: int) -> dict[tuple[str, ...], Any]:
    """The nine seed-bearing paths for an R0268 cell (R0265's set, R0268 capability)."""
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
    """The masked digest of an R0268 config -- R0217's masker over the same nine paths."""
    return seed_invariant_sha256(config)


# --------------------------------------------------------------------------- #
# the recipe invariant -- R0265's proof on a probe, then the ×2 + int8 delta
# --------------------------------------------------------------------------- #


def assert_registered_100m_int8_recipe(config: Mapping[str, Any]) -> dict[str, Any]:
    """Refuse unless `config` is EXACTLY R0265's recipe + dose ×2 + host-int8, at 100M.

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
            "runs the fp32 device path, hardware-infeasible at 100M and unproven)"
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
        problems.append(f"input.rows != {ROWS} (R0268 is the 100M flagship)")
    if edges != SEALED_DIRECTED_EDGES:
        problems.append(
            f"dose_registration.active_graph_edges {edges} != {SEALED_DIRECTED_EDGES} "
            "(the sealed R0243 100M k15 graph edge count)"
        )
    if problems:
        raise Round0268RecipeError(
            "R0268 config is not the registered 100M ×2 host-int8 fneg recipe: "
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
        "dose ×4->×2, scale 2M->100M (sealed R0238 substrate + R0243 graph), and R0266's "
        "residency+routing: x_residency device_fp16->host_int8, required_pipeline "
        "device->host_int8, expected_pipeline_stamp {x_residency, pipeline} device->host_int8"
    )
    return recipe


def _honest_100m_config(seed: int = CANONICAL_SEED) -> dict[str, Any]:
    """A canonical honest 100M ×2 config on synthetic sealed 100M signatures (no artifact read)."""
    config, _sha = int8_100m_train_config(
        graph_signature={"canonical_path": "/sealed/100m/edges-k15-fuzzy", "sha256": "b" * 64},
        graph_manifest_signature={"canonical_path": "/sealed/100m/fuzzy-graph.json", "sha256": "c" * 64},
        substrate_signature={"canonical_path": "/sealed/100m/substrate.f32.npy", "sha256": "a" * 64},
        graph_edges=SEALED_DIRECTED_EDGES,
        rows=ROWS,
        seed=seed,
    )
    return config


def recipe_refusal_controls() -> dict[str, Any]:
    """Plant the ×2/int8 delta defects AND R0265 recipe defects against the SHIPPED guard.

    Every control calls the shipped `assert_registered_100m_int8_recipe`; none reimplements
    it. The dose plant (a ×4 R0265 dose) proves the ×2 rule is enforced; the residency and
    weighted plants prove the int8 delta and the delegated R0265 uniform proof still fire.
    """
    honest = _honest_100m_config()
    controls: list[dict[str, Any]] = []

    def _plant(name: str, description: str, mutate) -> None:
        planted = copy.deepcopy(honest)
        mutate(planted)
        refused = False
        error = None
        try:
            assert_registered_100m_int8_recipe(planted)
        except (Round0268RecipeError, R0265.Round0265RecipeError) as raised:
            refused = True
            error = f"{type(raised).__name__}: {raised}"
        controls.append({
            "control": name,
            "plants": description,
            "shipped_predicate_refused": refused,
            "shipped_predicate_error": error,
        })

    def _wrong_dose_x4(cfg: dict[str, Any]) -> None:
        # R0265's ×4 dose instead of the ×2 flagship dose -- refused by the ×2 rule.
        x4 = R0265.validate_fneg_dose(
            updates=R0265.DOSE_MULTIPLIER * successful_updates_for_edges(SEALED_DIRECTED_EDGES),
            edge_count=SEALED_DIRECTED_EDGES,
        )
        _set_path(cfg, ("optimizer", "successful_positive_lr_updates"), x4["successful_updates"])
        _set_path(cfg, ("execution", "target_positive_draws_per_edge"),
                  x4["target_positive_draws_per_edge"])
        cfg["dose_registration"] = x4

    _plant("wrong_dose_x4",
           "R0265's ×4 dose instead of the ×2 flagship dose -- refused by the ×2 rule",
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
        assert_registered_100m_int8_recipe(honest)
    except (Round0268RecipeError, R0265.Round0265RecipeError) as raised:
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
            "every control calls the SHIPPED assert_registered_100m_int8_recipe; none "
            "reimplements it. The ×4 dose plant exercises the ×2 rule; the residency and "
            "weighted/fneg plants exercise the int8 delta and the delegated R0265 proof."
        ),
    }


# --------------------------------------------------------------------------- #
# the import-closure seal (reused R0265/R0266 primitives, R0268's module list)
# --------------------------------------------------------------------------- #

file_sha256 = R0265.file_sha256


def runtime_closure_hashes(
    modules: Sequence[str] = TRAIN_CLOSURE_MODULES,
) -> dict[str, dict[str, Any]]:
    """Hash the files behind R0268's closure (R0266's + this module)."""
    return R0265.runtime_closure_hashes(modules)


def assert_runtime_closure_matches_seal(
    *,
    sealed: Mapping[str, Any],
    observed: Mapping[str, Mapping[str, Any]],
    modules: Sequence[str] = TRAIN_CLOSURE_MODULES,
) -> dict[str, Any]:
    """Refuse unless every module in R0268's registered closure ran the sealed bytes."""
    return R0265.assert_runtime_closure_matches_seal(
        sealed=sealed, observed=observed, modules=modules
    )


def treatment_closure_controls(
    *, sealed: Mapping[str, Any], observed: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Plant closure defects against the SHIPPED R0268 predicate. Every control calls it."""
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

    victim = "basemap.round0268_int8_treatment"  # the 100M ×2 recipe bytes
    mutated = {name: dict(value) for name, value in honest_observed.items()}
    mutated[victim]["sha256"] = "0" * 64
    _run("content_drift", "the R0268 100M ×2 recipe ran bytes that are not the sealed ones",
         sealed=sealed, observed=mutated)

    dropped = {name: dict(value) for name, value in honest_observed.items() if name != victim}
    _run("missing_module", "the R0268 100M ×2 recipe module is absent from the runtime map",
         sealed=sealed, observed=dropped)

    extra = {name: dict(value) for name, value in honest_observed.items()}
    extra["basemap.round0268_missing"] = {"path": __file__, "bytes": 0, "sha256": "1" * 64}
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
    "FULL_I8_SHA256",
    "FULL_SCALES_SHA256",
    "HORIZON",
    "INT8_SUBSTRATE_CAPABILITY",
    "INT8_SUBSTRATE_SCHEMA",
    "R0238_ROUND_ID",
    "R0238_SUBSTRATE_CAPABILITY",
    "R0238_SUBSTRATE_ORDERED_SHA256",
    "R0243_GRAPH_CAPABILITY",
    "R0243_ROUND_ID",
    "R0262_I8_PATH",
    "R0262_IDENTITY_MANIFEST",
    "R0262_INT8_CAPABILITY",
    "R0262_ROUND_ID",
    "R0262_SCALES_PATH",
    "INT8_RECIPE_SCHEMA",
    "INT8_RESIDENCY_PATHS",
    "INT8_TRAIN_CONFIG_SCHEMA",
    "ROUND_ID",
    "ROWS",
    "SEALED_DIRECTED_EDGES",
    "SEEDS",
    "TARGET_DRAWS_PER_EDGE",
    "TRAIN_CLOSURE_MODULES",
    "X_RESIDENCY",
    "Round0268RecipeError",
    "assert_registered_100m_int8_recipe",
    "assert_runtime_closure_matches_seal",
    "capability_for_seed",
    "exact_cell_id",
    "file_sha256",
    "fneg_seed_invariant_sha256",
    "full_file_law_block",
    "int8_100m_train_config",
    "int8_full_digests",
    "int8_full_substrate_manifest_body",
    "recipe_refusal_controls",
    "runtime_closure_hashes",
    "sha256_file",
    "treatment_closure_controls",
    "validate_dose_x2",
]
