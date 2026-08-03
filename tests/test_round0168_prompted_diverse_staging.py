"""Pure population-lineage tests for R0168."""
from __future__ import annotations

import copy

import pytest

from basemap.round0168_prompted_diverse_staging import (
    DIMENSION,
    DTYPE,
    EXPECTED_SCHEMAS,
    TOTAL_ROWS,
    Round0168Error,
    prompted_selection,
)


GROUPS = {
    "0116": [None, None],
    "0120": [None],
    "0126": ["arb_Arab", "ces_Latn", "cmn_Hani"],
    "0127": ["deu_Latn", "ell_Grek", "fra_Latn"],
    "0139": ["hin_Deva", "ind_Latn", "ita_Latn"],
    "0141": ["jpn_Jpan", "kor_Hang", "nld_Latn"],
    "0143": ["pol_Latn", "por_Latn", "rus_Cyrl"],
    "0144": ["spa_Latn", "swe_Latn", "tha_Thai"],
    "0145": ["tur_Latn", "vie_Latn"],
}
ENGLISH = ["fineweb", "redpajama", "pile"]


def _fixtures():
    languages = [language for values in GROUPS.values() for language in values if language]
    included_languages = [language for language in languages if language != "pol_Latn"]
    datasets = ENGLISH + [f"fineweb2-{language}" for language in included_languages]
    widths = [1_000_000] * (len(datasets) - 1) + [TOTAL_ROWS - 1_000_000 * (len(datasets) - 1)]
    raw_ranges = []
    geometry = {}
    cursor = 0
    for dataset, width in zip(datasets, widths, strict=True):
        geometry[dataset] = (cursor, width)
        raw_ranges.append({
            "dataset": dataset,
            "dataset_row_start": 0,
            "dataset_row_stop": width,
            "global_row_start": cursor,
            "global_row_stop": cursor + width,
        })
        cursor += width
    raw = {"selection": {"selected_rows": TOTAL_ROWS, "source_order": datasets, "ranges": raw_ranges}}

    def chunk(dataset, rows, global_start, *, round_id, language=None, part=0, parts=1):
        start = rows * part // parts
        stop = rows * (part + 1) // parts
        value = {
            "dataset": dataset,
            "dataset_row_range": [start, stop],
            "output": {
                "canonical_path": f"/data/{round_id}-{dataset}-{part}.npy",
                "kind": "file",
                "bytes": (stop - start) * DIMENSION * 2 + 128,
                "sha256": f"{(part + len(dataset)) % 16:x}" * 64,
            },
            "output_dtype": DTYPE,
            "output_shape": [stop - start, DIMENSION],
        }
        if language is not None:
            value["language"] = language
        if language != "pol_Latn":
            key = "corpus_global_row_range" if round_id == "0116" else "r0087_global_row_range"
            value[key] = [global_start + start, global_start + stop]
        return value

    manifests = {}
    english_cursor = 0
    for round_id, group in GROUPS.items():
        base = {
            "round_id": round_id,
            "schema": EXPECTED_SCHEMAS[round_id],
            "training_performed": False,
            "dimension": DIMENSION,
            "dtype": DTYPE,
            "convention": {"prompt_prefix": "Document: "},
            "model": {
                "id": "jinaai/jina-embeddings-v5-text-nano-retrieval",
                "revision": "ac5d898c8d382b17167c33e5c8af644a3519b47d",
            },
        }
        if round_id == "0116":
            order = ENGLISH[:2]
            sources = {}
            for dataset in order:
                start, rows = geometry[dataset]
                sources[dataset] = {"row_count": rows, "chunk_count": 1, "chunks": [chunk(dataset, rows, start, round_id=round_id)]}
            base.update({"source_order": order, "datasets": sources})
        elif round_id == "0120":
            dataset = ENGLISH[2]
            start, rows = geometry[dataset]
            base.update({"source_order": [dataset], "dataset": {"row_count": rows, "chunk_count": 1, "chunks": [chunk(dataset, rows, start, round_id=round_id)]}})
        else:
            sources = {}
            for language in group:
                dataset = f"fineweb2-{language}"
                if language == "pol_Latn":
                    rows, start, parts = 34, 0, 34
                else:
                    start, rows = geometry[dataset]
                    parts = 1
                chunks = [chunk(dataset, rows, start, round_id=round_id, language=language, part=part, parts=parts) for part in range(parts)]
                sources[language] = {"row_count": rows, "chunk_count": len(chunks), "chunks": chunks}
            base.update({"source_order": group, "languages": sources})
        manifests[round_id] = base
    return raw, manifests


def test_prompted_selection_matches_exact_raw_global_population() -> None:
    raw, manifests = _fixtures()
    selection = prompted_selection(raw, manifests)
    assert selection["selected_rows"] == TOTAL_ROWS
    assert len(selection["ranges"]) == 22
    assert len(selection["source_signatures"]) == 22
    assert selection["ranges"][0]["global_row_start"] == 0
    assert selection["ranges"][-1]["global_row_stop"] == TOTAL_ROWS
    assert selection["heldout_polish"] == {
        "language": "pol_Latn",
        "chunks": 34,
        "rows": 34,
        "excluded_from_training": True,
    }
    assert selection["coverage"]["dataset_row_identity_matches_r0087"] is True
    assert len(selection["ordered_selection_sha256"]) == 64


def test_prompted_selection_rejects_one_row_population_shift() -> None:
    raw, manifests = _fixtures()
    broken = copy.deepcopy(manifests)
    chunk = broken["0145"]["languages"]["vie_Latn"]["chunks"][0]
    chunk["r0087_global_row_range"][0] -= 1
    chunk["r0087_global_row_range"][1] -= 1
    with pytest.raises(Round0168Error, match="exact R0087 row identity"):
        prompted_selection(raw, broken)


def test_prompted_selection_rejects_unmapped_non_polish_chunk() -> None:
    raw, manifests = _fixtures()
    broken = copy.deepcopy(manifests)
    del broken["0126"]["languages"]["arb_Arab"]["chunks"][0]["r0087_global_row_range"]
    with pytest.raises(Round0168Error, match="only held-out Polish"):
        prompted_selection(raw, broken)
