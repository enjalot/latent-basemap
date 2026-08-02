"""Campaign-authorized 12.5M historical-prefix/drop-only census contract."""
from __future__ import annotations

from basemap.round0151_scale_census import (  # noqa: F401
    EXPECTED_DROPPED_ROWS,
    EXPECTED_GROUP_IDS_ORDERED_SHA256,
    EXPECTED_MAPPING_ORDERED_SHA256,
    EXPECTED_RETAINED_ROWS,
    EXPECTED_U12_OVERLAP,
    FULL_RAW_ROWS,
    RAW_PREFIX_TARGET,
    build_prefix_drop_mapping,
    compare_to_u12,
    inventory_group_ranges,
    largest_remainder_prefix_quotas,
)


ROUND_ID = "0155"
CAPABILITY = "jina-diverse-12p5m-historical-prefix-census-v1"


class Round0155Error(RuntimeError):
    """Raised when the campaign-authorized R0155 census changes."""
