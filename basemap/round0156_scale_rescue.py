"""Scale-native k15 treatment for the R0153-activated 12.5M candidate."""
from __future__ import annotations


ROUND_ID = "0156"
CAPABILITY = "jina-diverse-12p5m-historical-prefix-map-v1"
PARENT_ROUND_ID = "0155"
PARENT_CAPABILITY = "jina-diverse-12p5m-historical-prefix-census-v1"

RETAINED_ROWS = 12_485_206
GRAPH_K = 15
N_NEIGHBORS = 16
SEED = 42

SUBSET_SCHEMA = "round0156-historical-prefix-subset-v1"
INDEX_SCHEMA = "round0156-historical-prefix-search-index-v1"
QUALIFICATION_SCHEMA = "round0156-historical-prefix-search-qualification-v1"
GRAPH_SHARD_SCHEMA = "round0156-historical-prefix-graph-shard-v1"
GRAPH_PART_SCHEMA = "round0156-historical-prefix-graph-part-v1"
GRAPH_SCHEMA = "round0156-historical-prefix-fuzzy-graph-v1"
TRAIN_CONFIG_SCHEMA = "round0156-historical-prefix-train-config-v1"
PRODUCTION_CONFIG_SCHEMA = "round0156-historical-prefix-production-config-v1"
TRAIN_RECEIPT_SCHEMA = "round0156-historical-prefix-train-receipt-v1"
NATIVE_SCHEMA = "round0156-historical-prefix-matched-native-panel-v1"
OOD_SCHEMA = "round0156-historical-prefix-matched-ood-panel-v1"
FUNCTIONAL_SCHEMA = "round0156-historical-prefix-functional-density-panel-v1"
DECISION_SCHEMA = "round0156-historical-prefix-decision-v1"

PIPELINE = "host_weighted_jina_diverse_12p5m_historical_prefix"
PIPELINE_SCHEMA = "round0156-host-weighted-jina-diverse-historical-prefix-v1"
POSITIVE_DESTINATION_POLICY = (
    "R0156-global-historical-prefix-retained-fuzzy-tconorm-graph"
)
UPDATE_RULE = "ceil(actual-R0156-directed-fuzzy-edges/409)"
GRAPH_DEGREE = "variable-symmetric-fuzzy-k15-topology"

DENSITY_FLOOR = 0.17589389755990817
OOD_RETENTION = 0.97
OUTCOME_PASS = "historical-prefix-12p5m-shippable-v1-passes"
OUTCOME_FAIL = "historical-prefix-12p5m-quality-gate-fails"
OUTCOME_INVALID = "invalid-execution"


class Round0156Error(RuntimeError):
    """Raised when the registered R0156 scale treatment changes."""
