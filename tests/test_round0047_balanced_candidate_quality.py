from __future__ import annotations

import inspect

from experiments import prepare_round0047_queue as queue_prep
from experiments import round0044_nodes as node


def test_r0047_selects_first_member_of_each_corpus_block() -> None:
    assert node._embedding_member_indices("0047") == (0, 10, 20)
    assert node._embedding_member_indices("0045") == (0, 1, 2)


def test_r0047_handler_is_explicitly_allowlisted() -> None:
    assert node.CORRECTION_ROUND_ID == "0047"
    source = inspect.getsource(node.run_job)
    assert "CORRECTION_ROUND_ID" in source


def test_r0047_queue_is_bounded_no_training_correction() -> None:
    source = inspect.getsource(queue_prep.prepare_round0047)
    assert "gpu_hours_cap=0.5" in source
    assert '"training_performed"] = False' in source
    assert '"supersedes"] = ["0045"]' in source
    assert "CORRECTED_MEMBER_INDICES" in source
