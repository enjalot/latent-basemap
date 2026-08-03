"""Exact prompted representation of the frozen R0132 U12 population."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from basemap.artifact_identity import canonical_json, sha256_bytes


ROUND_ID = "0168"
CAPABILITY = "jina-document-diverse-r0132-u12-host-fp16-v1"
MANIFEST_SCHEMA = "round0168-prompted-diverse-u12-staging-v1"
SELECTION_SCHEMA = "round0168-prompted-r0087-layout-v1"
TOTAL_ROWS = 25_000_000
U12_ROWS = 12_474_331
DIMENSION = 768
DTYPE = "<f2"
POLISH = "pol_Latn"
EXPECTED_SCHEMAS = {
    "0116": "jina-document-english-fineweb-rpj-5p727m-v1",
    "0120": "jina-document-pile-english-3p399m-v1",
    "0126": "jina-document-multilingual-arb-ces-cmn-2p506m-v1",
    "0127": "jina-document-multilingual-deu-ell-fra-2p506m-v1",
    "0139": "jina-document-multilingual-hin-ind-ita-2p506m-v1",
    "0141": "jina-document-multilingual-jpn-kor-nld-2p506m-v1",
    "0143": "jina-document-multilingual-pol-por-rus-2p506m-v1",
    "0144": "jina-document-multilingual-spa-swe-tha-2p506m-v1",
    "0145": "jina-document-multilingual-tur-vie-1p671m-v1",
}


class Round0168Error(RuntimeError):
    """The reviewed prompted tranche or frozen U12 lineage changed."""


def _signature(value: Any, *, rows: int) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Round0168Error("prompted chunk output signature is missing")
    output = dict(value)
    if (
        output.get("kind") != "file"
        or not isinstance(output.get("canonical_path"), str)
        or int(output.get("bytes", -1)) != rows * DIMENSION * 2 + 128
        or not isinstance(output.get("sha256"), str)
        or len(str(output["sha256"])) != 64
    ):
        raise Round0168Error("prompted chunk output signature is malformed")
    output["rows"] = rows
    return output


def _iter_sources(manifests: Mapping[str, Mapping[str, Any]]):
    """Yield ``(round, language, dataset description)`` in manifest order."""
    expected_rounds = ("0116", "0120", "0126", "0127", "0139", "0141", "0143", "0144", "0145")
    if tuple(manifests) != expected_rounds:
        raise Round0168Error("prompted tranche manifest order changed")
    for round_id, manifest in manifests.items():
        convention = manifest.get("convention") or {}
        model = manifest.get("model") or {}
        if (
            str(manifest.get("round_id") or "") != round_id
            or manifest.get("schema") != EXPECTED_SCHEMAS[round_id]
            or manifest.get("training_performed") is not False
            or int(manifest.get("dimension", -1)) != DIMENSION
            or manifest.get("dtype") != DTYPE
            or convention.get("prompt_prefix") != "Document: "
            or model.get("id") != "jinaai/jina-embeddings-v5-text-nano-retrieval"
            or model.get("revision") != "ac5d898c8d382b17167c33e5c8af644a3519b47d"
        ):
            raise Round0168Error(f"prompted R{round_id} tranche contract changed")
        if round_id == "0116":
            sources = manifest.get("datasets")
            order = manifest.get("source_order")
            if not isinstance(sources, Mapping) or not isinstance(order, list):
                raise Round0168Error("R0116 prompted sources are missing")
            for dataset in order:
                yield round_id, None, str(dataset), sources.get(dataset)
        elif round_id == "0120":
            source = manifest.get("dataset")
            order = manifest.get("source_order")
            if not isinstance(source, Mapping) or not isinstance(order, list) or len(order) != 1:
                raise Round0168Error("R0120 prompted source is missing")
            yield round_id, None, str(order[0]), source
        else:
            sources = manifest.get("languages")
            order = manifest.get("source_order")
            if not isinstance(sources, Mapping) or not isinstance(order, list):
                raise Round0168Error(f"R{round_id} prompted languages are missing")
            for language in order:
                source = sources.get(language)
                if not isinstance(source, Mapping):
                    raise Round0168Error(f"R{round_id} prompted language is missing")
                chunks = source.get("chunks") or []
                dataset = str(chunks[0].get("dataset") or "") if chunks else ""
                yield round_id, str(language), dataset, source


def prompted_selection(
    raw_inventory: Mapping[str, Any],
    manifests: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Normalize reviewed prompted chunks onto the exact R0087 global layout.

    Polish is a probe-only tranche and therefore has no R0087 row range.  It is
    the only accepted omission.  Every included chunk must encode the same
    dataset-row to global-row affine mapping as the accepted raw inventory.
    """
    raw = raw_inventory.get("selection")
    if not isinstance(raw, Mapping):
        raise Round0168Error("accepted R0087 selection is missing")
    source_order = [str(value) for value in raw.get("source_order") or []]
    raw_ranges = list(raw.get("ranges") or [])
    if int(raw.get("selected_rows", -1)) != TOTAL_ROWS or not raw_ranges:
        raise Round0168Error("accepted R0087 selection geometry changed")

    expected: dict[str, tuple[int, int, int]] = {}
    for item in raw_ranges:
        dataset = str(item.get("dataset") or "")
        ds_start = int(item.get("dataset_row_start", -1))
        ds_stop = int(item.get("dataset_row_stop", -1))
        gl_start = int(item.get("global_row_start", -1))
        gl_stop = int(item.get("global_row_stop", -1))
        if ds_stop - ds_start != gl_stop - gl_start:
            raise Round0168Error("accepted R0087 dataset/global mapping changed")
        previous = expected.get(dataset)
        if previous is None:
            expected[dataset] = (ds_start, ds_stop, gl_start - ds_start)
        else:
            first, stop, offset = previous
            if ds_start != stop or gl_start - ds_start != offset:
                raise Round0168Error("accepted R0087 dataset ranges are not affine")
            expected[dataset] = (first, ds_stop, offset)

    ranges: list[dict[str, Any]] = []
    omitted: list[dict[str, Any]] = []
    source_signatures: list[dict[str, Any]] = []
    for round_id, language, dataset, source in _iter_sources(manifests):
        if not isinstance(source, Mapping):
            raise Round0168Error(f"R{round_id} prompted source description is missing")
        chunks = source.get("chunks")
        if (
            not dataset
            or not isinstance(chunks, list)
            or not chunks
            or int(source.get("row_count", -1)) <= 0
            or int(source.get("chunk_count", -1)) != len(chunks)
        ):
            raise Round0168Error(f"R{round_id} prompted source geometry changed")
        dataset_cursor = 0
        for chunk in chunks:
            dataset_range = list(chunk.get("dataset_row_range") or [])
            shape = list(chunk.get("output_shape") or [])
            if (
                len(dataset_range) != 2
                or dataset_range[0] != dataset_cursor
                or dataset_range[1] <= dataset_range[0]
                or shape != [dataset_range[1] - dataset_range[0], DIMENSION]
                or chunk.get("output_dtype") != DTYPE
                or str(chunk.get("dataset") or "") != dataset
            ):
                raise Round0168Error(f"R{round_id} prompted chunk order changed")
            rows = dataset_range[1] - dataset_range[0]
            output = _signature(chunk.get("output"), rows=rows)
            global_range = chunk.get("r0087_global_row_range")
            if global_range is None and round_id == "0116":
                global_range = chunk.get("corpus_global_row_range")
            if global_range is None:
                if language != POLISH or round_id != "0143":
                    raise Round0168Error("only held-out Polish may lack an R0087 range")
                omitted.append({
                    "round_id": round_id,
                    "language": language,
                    "dataset": dataset,
                    "dataset_row_range": dataset_range,
                    "source_output": output,
                })
            else:
                source_signatures.append({
                    key: output[key]
                    for key in ("canonical_path", "kind", "bytes", "sha256")
                })
                global_range = list(global_range)
                if len(global_range) != 2 or global_range[1] - global_range[0] != rows:
                    raise Round0168Error("prompted R0087 global range changed")
                ranges.append({
                    "dataset": dataset,
                    "language": language,
                    "source_round": round_id,
                    "dataset_row_start": dataset_range[0],
                    "dataset_row_stop": dataset_range[1],
                    "global_row_start": int(global_range[0]),
                    "global_row_stop": int(global_range[1]),
                    "shard": output,
                    "shard_row_start": 0,
                    "shard_row_stop": rows,
                })
            dataset_cursor = dataset_range[1]
        if dataset_cursor != int(source["row_count"]):
            raise Round0168Error(f"R{round_id} prompted source rows do not close")

    ranges.sort(key=lambda item: int(item["global_row_start"]))
    cursor = 0
    observed_order: list[str] = []
    dataset_cursors: dict[str, int] = {}
    for item in ranges:
        dataset = str(item["dataset"])
        ds_start = int(item["dataset_row_start"])
        ds_stop = int(item["dataset_row_stop"])
        gl_start = int(item["global_row_start"])
        gl_stop = int(item["global_row_stop"])
        contract = expected.get(dataset)
        if (
            contract is None
            or gl_start != cursor
            or ds_start != dataset_cursors.get(dataset, contract[0])
            or gl_start - ds_start != contract[2]
            or ds_stop - ds_start != gl_stop - gl_start
        ):
            raise Round0168Error("prompted chunks do not preserve exact R0087 row identity")
        dataset_cursors[dataset] = ds_stop
        cursor = gl_stop
        if not observed_order or observed_order[-1] != dataset:
            observed_order.append(dataset)
    if (
        cursor != TOTAL_ROWS
        or observed_order != source_order
        or set(dataset_cursors) != set(expected)
        or any(dataset_cursors[name] != expected[name][1] for name in expected)
        or len(omitted) != 34
        or {item["language"] for item in omitted} != {POLISH}
    ):
        raise Round0168Error("prompted 25M coverage or Polish exclusion changed")

    body = {
        "schema": SELECTION_SCHEMA,
        "selected_rows": TOTAL_ROWS,
        "dimension": DIMENSION,
        "dtype": DTYPE,
        "embedding_convention": "Document: ",
        "source_order": source_order,
        "ranges": ranges,
        "heldout_polish": {
            "language": POLISH,
            "chunks": len(omitted),
            "rows": sum(int(item["dataset_row_range"][1]) - int(item["dataset_row_range"][0]) for item in omitted),
            "excluded_from_training": True,
        },
        "coverage": {
            "gap_free": True,
            "overlap_free": True,
            "dataset_row_identity_matches_r0087": True,
        },
    }
    return {
        **body,
        "ordered_selection_sha256": sha256_bytes(canonical_json(body)),
        "source_signatures": source_signatures,
    }
