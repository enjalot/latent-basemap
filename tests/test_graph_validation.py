"""P0.8: graph/data pair validation + node manifests."""
import sys, os, numpy as np, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from basemap.graph_validation import (validate_edge_bounds, validate_graph_data_pair,
                                       graph_manifest, edge_endpoint_cosine_check)


def test_rejects_negative_sentinel():
    s = np.array([0, 1, -1, 3]); t = np.array([1, 2, 3, 0])
    with pytest.raises(ValueError, match="negative id"):
        validate_edge_bounds(s, t, n_nodes=4)


def test_rejects_out_of_range():
    s = np.array([0, 1, 5]); t = np.array([1, 2, 3])
    with pytest.raises(ValueError, match="out of range|>= n_nodes"):
        validate_edge_bounds(s, t, n_nodes=4)


def test_aligned_pair_ok_no_filter():
    s = np.array([0, 1, 2]); t = np.array([1, 2, 0])
    assert validate_graph_data_pair(s, t, n_nodes=3, n_train=3) is None


def test_prefix_filter_rejected_by_default():
    # larger graph than X -> must NOT silently prefix-filter
    s = np.array([0, 1, 5, 60]); t = np.array([1, 2, 3, 4])
    with pytest.raises(ValueError, match="prefix-filter|balanced"):
        validate_graph_data_pair(s, t, n_nodes=100, n_train=50)


def test_prefix_filter_allowed_masks_and_rejects_sentinel_in_range():
    s = np.array([0, 1, 60, 5]); t = np.array([1, 2, 3, 70])
    mask = validate_graph_data_pair(s, t, n_nodes=100, n_train=50, allow_prefix_filter=True)
    # edges within [0,50): (0,1),(1,2) kept; (60,3),(5,70) dropped
    assert mask.tolist() == [True, True, False, False]


def test_graph_smaller_than_data_errors():
    s = np.array([0, 1]); t = np.array([1, 0])
    with pytest.raises(ValueError, match="< training rows"):
        validate_graph_data_pair(s, t, n_nodes=2, n_train=10)


def test_manifest_fields():
    s = np.array([0, 1, 2]); t = np.array([1, 2, 0]); X = np.random.randn(3, 4).astype('float32')
    m = graph_manifest(s, t, 3, X=X)
    assert m['n_nodes'] == 3 and m['n_edges'] == 3 and m['source_max'] == 2
    assert 'data_fingerprint' in m and m['data_len'] == 3


def test_endpoint_cosine_catches_wrong_pairing():
    # random edges over random X -> low margin -> must raise
    X = np.random.RandomState(0).randn(2000, 8).astype('float32')
    s = np.random.randint(0, 2000, 5000); t = np.random.randint(0, 2000, 5000)
    with pytest.raises(ValueError, match="endpoint-cosine check FAILED"):
        edge_endpoint_cosine_check(s, t, X, n_probe=2000)




def test_manifest_catches_shuffle_same_length():
    from basemap.graph_validation import graph_manifest, validate_against_manifest
    X = np.random.RandomState(0).randn(500, 8).astype('float32')
    s = np.array([0,1,2]); t = np.array([1,2,0])
    man = graph_manifest(s, t, 500, X=X)
    validate_against_manifest(X, man)                      # identical → ok
    Xsh = X[np.random.RandomState(1).permutation(500)]     # same length, shuffled rows
    with pytest.raises(ValueError, match="reordered|fingerprint"):
        validate_against_manifest(Xsh, man)


def test_manifest_catches_changed_shard():
    from basemap.graph_validation import graph_manifest, validate_against_manifest
    X = np.random.RandomState(0).randn(500, 8).astype('float32')
    man = graph_manifest(np.array([0,1]), np.array([1,0]), 500, X=X)
    Xc = X.copy(); Xc[123] += 5.0                          # one row changed
    with pytest.raises(ValueError, match="reordered|fingerprint"):
        validate_against_manifest(Xc, man)


def test_manifest_length_mismatch_raises():
    from basemap.graph_validation import graph_manifest, validate_against_manifest
    X = np.random.RandomState(0).randn(500, 8).astype('float32')
    man = graph_manifest(np.array([0,1]), np.array([1,0]), 500, X=X)
    with pytest.raises(ValueError, match="does not match manifest"):
        validate_against_manifest(X[:400], man)            # shorter, no prefix allowance


def test_prefix_refused_without_stored_fingerprint():
    # P0-2: allow_prefix must NOT be an unconditional pass — a prefix with no
    # stored fingerprint to verify against is refused.
    from basemap.graph_validation import graph_manifest, validate_against_manifest, data_fingerprint
    X = np.random.RandomState(0).randn(500, 8).astype('float32')
    man = graph_manifest(np.array([0,1]), np.array([1,0]), 500, X=X)
    with pytest.raises(ValueError, match="no stored prefix fingerprint|refuse the prefix"):
        validate_against_manifest(X[:400], man, allow_prefix=True)
    # with a stored prefix fingerprint, the verified prefix passes; a mismatched one fails
    man["prefix_fingerprints"] = {"400": data_fingerprint(X[:400])[1]}
    validate_against_manifest(X[:400], man, allow_prefix=True)
    Xbad = X.copy(); Xbad[123] += 9.0
    with pytest.raises(ValueError, match="prefix fingerprint"):
        validate_against_manifest(Xbad[:400], man, allow_prefix=True)


if __name__ == '__main__':
    import traceback
    fns = [test_rejects_negative_sentinel, test_rejects_out_of_range, test_aligned_pair_ok_no_filter,
           test_prefix_filter_rejected_by_default, test_prefix_filter_allowed_masks_and_rejects_sentinel_in_range,
           test_graph_smaller_than_data_errors, test_manifest_fields, test_endpoint_cosine_catches_wrong_pairing]
    for fn in fns:
        pass
    print("ALL P0.8 TESTS PASSED")
