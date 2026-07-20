import json

import pytest

from experiments import round0011_jina_inventory as r0011


def test_cpu_only_guard_rejects_visible_cuda(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    with pytest.raises(r0011.ContractError) as excinfo:
        r0011.assert_cpu_only_process()

    assert excinfo.value.code == "cuda_not_hidden"


def test_range_and_document_identity_rejections():
    ranges = [
        {
            "relative_path": "a.parquet",
            "corpus_row_start": 0,
            "corpus_row_end": 2,
            "row_count": 2,
            "filesystem": {"device": 1, "inode": 1},
        },
        {
            "relative_path": "b.parquet",
            "corpus_row_start": 2,
            "corpus_row_end": 4,
            "row_count": 2,
            "filesystem": {"device": 1, "inode": 2},
        },
    ]
    r0011.validate_range_cover(ranges, 4)

    gap = json.loads(json.dumps(ranges))
    gap[1]["corpus_row_start"] = 3
    gap[1]["corpus_row_end"] = 5
    with pytest.raises(r0011.ContractError) as excinfo:
        r0011.validate_range_cover(gap, 5)
    assert excinfo.value.code == "range_gap"

    overlap = json.loads(json.dumps(ranges))
    overlap[1]["corpus_row_start"] = 1
    overlap[1]["row_count"] = 3
    with pytest.raises(r0011.ContractError) as excinfo:
        r0011.validate_range_cover(overlap, 4)
    assert excinfo.value.code == "range_overlap"

    with pytest.raises(r0011.ContractError) as excinfo:
        r0011.validate_unique_document_ids(["doc-a", "doc-a"])
    assert excinfo.value.code == "duplicate_document_id"


def test_resume_index_rejects_partial_and_prompt_drift(tmp_path):
    unit = r0011._fixture_unit()
    r0011._write_fixture_chunk(tmp_path, unit, 0)
    assert r0011.resume_index(unit, tmp_path) == 1

    partial = tmp_path / "partial"
    partial.mkdir()
    r0011.atomic_write_new_json(partial / "chunk-0000.receipt.json", {"status": "complete"})
    with pytest.raises(r0011.ContractError) as excinfo:
        r0011.resume_index(unit, partial)
    assert excinfo.value.code == "partial_receipt"

    prompt = tmp_path / "prompt"
    prompt.mkdir()
    r0011._write_fixture_chunk(prompt, unit, 0)
    receipt_path = prompt / "chunk-0000.receipt.json"
    receipt = r0011.read_json(receipt_path)
    receipt["prompt_utf8_hex"] = "00"
    receipt_path.unlink()
    r0011.atomic_write_new_json(receipt_path, receipt)
    with pytest.raises(r0011.ContractError) as excinfo:
        r0011.resume_index(unit, prompt)
    assert excinfo.value.code == "prompt_drift"


def test_fixture_matrix_receipt_is_self_validating(tmp_path):
    receipt = r0011.run_fixture_matrix(scratch_parent=tmp_path)

    r0011.validate_fixture_artifact(receipt)
    assert receipt["all_passed"] is True
    assert {case["name"] for case in receipt["cases"]} == {
        "resume-after-interrupted-chunk",
        "partial-receipt",
        "output-collision",
        "prompt-drift",
        "tokenizer-drift",
        "source-drift",
        "source-row-reorder",
        "range-overlap",
        "range-gap",
        "duplicate-document-id",
    }
