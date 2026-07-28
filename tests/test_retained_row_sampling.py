import numpy as np

from experiments.round0049_nodes import _sample_retained_rows


def test_retained_row_sample_is_not_low_id_truncated() -> None:
    excluded = np.asarray([1, 4, 101, 202, 299], dtype=np.int64)
    rows = _sample_retained_rows(
        excluded,
        count=90,
        seed=86,
        row_count=300,
    )
    assert rows.shape == (90,)
    assert len(np.unique(rows)) == 90
    assert not np.isin(rows, excluded).any()
    assert ((rows >= 0) & (rows < 100)).any()
    assert ((rows >= 100) & (rows < 200)).any()
    assert ((rows >= 200) & (rows < 300)).any()


def test_retained_row_sample_is_deterministic() -> None:
    excluded = np.asarray([0, 7, 19, 88], dtype=np.int64)
    first = _sample_retained_rows(
        excluded,
        count=32,
        seed=95,
        row_count=100,
    )
    second = _sample_retained_rows(
        excluded,
        count=32,
        seed=95,
        row_count=100,
    )
    np.testing.assert_array_equal(first, second)
