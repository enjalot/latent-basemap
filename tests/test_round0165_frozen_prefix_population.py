"""Tests for the prompted 8M frozen-prefix extension law."""
from __future__ import annotations

import numpy as np

from basemap.round0165_frozen_prefix_population import frozen_prefix_extension


def test_frozen_prefix_drops_only_prompted_base_delta_and_keeps_extension() -> None:
    # Patch the module-scale constants through valid sparse mappings: the pure
    # law's boundary behavior is the subject, not materializing 8M test rows.
    prefix = np.arange(1_993_761, dtype=np.int64)
    prompted = np.concatenate((prefix, np.asarray([1_999_999, 2_000_000, 2_000_002])))
    prior = np.concatenate((prefix, np.asarray([2_000_000])))
    mapping, _excluded, dropped, added, positions, report = frozen_prefix_extension(
        accepted_prefix=prefix,
        prompted_only_mapping=prompted,
        prior_three_relation_mapping=prior,
    )
    assert dropped.tolist() == [1_999_999]
    assert added.tolist() == [2_000_002]
    assert mapping[-2:].tolist() == [2_000_000, 2_000_002]
    assert np.array_equal(prompted[positions], mapping)
    assert report["dropped_prompted_only_prefix_rows"] == 1

