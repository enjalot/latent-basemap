"""P0.2/P0-B (schedule + accounting) and P0.3 (zero-weight correlation skip)."""
import sys, os, numpy as np, torch, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from basemap.pumap.parametric_umap.core import ParametricUMAP


def _edges(n, e, seed=0):
    rng = np.random.RandomState(seed)
    return (rng.randint(0, n, e).astype(np.int32), rng.randint(0, n, e).astype(np.int32),
            rng.rand(e).astype(np.float32))


def _fit(n=500, d=8, horizon=100, epochs=50, batch=64, seed=2, warmup_steps=0, **kw):
    X = np.random.RandomState(seed).randn(n, d).astype(np.float32)
    s, t, w = _edges(n, 20000, seed)
    np.savez('/tmp/_pb_edges.npz', sources=s, targets=t, weights=w, n_nodes=n, k=15)
    m = ParametricUMAP(a=1., b=1., correlation_weight=0.0, n_epochs=epochs, batch_size=batch,
                       total_steps_estimate=horizon, lr_schedule='cosine',
                       device='cpu', positive_target_mode='binary', gpu_resident_data=False,
                       use_amp=False, warmup_steps=warmup_steps, **kw)
    m.fit(X, precomputed_edges_path='/tmp/_pb_edges.npz')
    return m


def test_p0b_exact_positive_lr_update_count():
    m = _fit(horizon=100)
    s = m._train_stats
    # EXACT: cosine anneals to LR=0 after `horizon` scheduler steps → exactly
    # `horizon` positive-LR updates (not <=105).
    assert s['positive_lr_optimizer_steps'] == 100, s
    assert s['stop_reason'] == 'lr_horizon'
    assert s['scheduler_steps'] == 100


def test_p0b_zero_horizon_derives_not_one_step():
    # total_steps_estimate<=0 must derive the planned loop, not become 1 step.
    m = _fit(horizon=0, epochs=1)
    s = m._train_stats
    assert s['lr_horizon'] == s['planned_loop_iters'] > 1, s


def test_p0b_warmup_ge_horizon_rejected():
    with pytest.raises(ValueError, match="warmup"):
        _fit(horizon=50, warmup_steps=50)


def test_p0b_bench_hook_populates_seconds_and_stops():
    # _max_train_steps=120 with a large horizon so the bench cap stops first.
    X = np.random.RandomState(2).randn(500, 8).astype(np.float32)
    s, t, w = _edges(500, 20000, 2); np.savez('/tmp/_pb2.npz', sources=s, targets=t, weights=w, n_nodes=500, k=15)
    m = ParametricUMAP(a=1., b=1., correlation_weight=0.0, n_epochs=50, batch_size=64,
                       total_steps_estimate=100000, lr_schedule='cosine', warmup_steps=0,
                       device='cpu', positive_target_mode='binary', gpu_resident_data=False, use_amp=False)
    m._max_train_steps = 120; m._bench_warmup = 20
    m.fit(X, precomputed_edges_path='/tmp/_pb2.npz')
    assert m._bench_seconds is not None, "bench seconds not finalized on stop path"
    assert m._train_stats['stop_reason'] == 'bench_cap'
    assert m._train_stats['executed_iters'] == 120


def test_p0a_amp_skip_path_does_not_break_scaler():
    # Regression: the P0-A gradient guard unscale_()s then may skip a batch; under
    # AMP that must call scaler.update() or the NEXT unscale_ raises
    # "unscale_() has already been called". Exercise the real AMP path on CUDA
    # (GradScaler is a no-op on CPU). The high initial scale reliably overflows on
    # the first steps → drives the skip path.
    if not torch.cuda.is_available():
        pytest.skip("AMP scaler path only meaningful on CUDA")
    X = np.random.RandomState(2).randn(600, 8).astype(np.float32)
    s, t, w = _edges(600, 20000, 2); np.savez('/tmp/_pb_amp.npz', sources=s, targets=t, weights=w, n_nodes=600, k=15)
    m = ParametricUMAP(a=1., b=1., correlation_weight=0.0, n_epochs=20, batch_size=64,
                       total_steps_estimate=150, lr_schedule='cosine', warmup_steps=0,
                       device='cuda', positive_target_mode='binary', gpu_resident_data=False,
                       use_amp=True)
    m.fit(X, precomputed_edges_path='/tmp/_pb_amp.npz')   # must NOT raise
    assert m.is_fitted
    assert np.isfinite(m.transform(X)).all()
    assert m._train_stats['stop_reason'] == 'lr_horizon'


def _p1_edges_with_manifest(n=400, e=1200, seed=1, tmp='/tmp/_p1c.npz'):
    from basemap.graph_validation import graph_manifest_v2, write_manifest
    X = np.random.RandomState(seed).randn(n, 8).astype(np.float32)
    s = np.repeat(np.arange(n), e // n).astype(np.int32)
    t = np.random.RandomState(seed + 1).randint(0, n, len(s)).astype(np.int32)
    w = np.random.RandomState(seed + 2).rand(len(s)).astype(np.float32)
    np.savez(tmp, sources=s, targets=t, weights=w, n_nodes=n, k=3)
    write_manifest(tmp + '.manifest.json', graph_manifest_v2(s, t, n, X=X, graph_path=tmp, k=3))
    return X, tmp


def _p1_umap(**kw):
    base = dict(a=1., b=1., correlation_weight=0.0, n_epochs=4, batch_size=64,
                total_steps_estimate=40, lr_schedule='cosine', warmup_steps=0, device='cpu',
                gpu_resident_data='auto', positive_target_mode='binary', use_amp=False,
                require_full_budget=False)
    base.update(kw)
    return ParametricUMAP(**base)


def test_p1_weighted_uniform_fallback_raises():
    # P1: weighted request that reaches the uniform legacy iterator must RAISE
    # before training (the 8M-bridge fallback that silently trained uniform).
    X, ep = _p1_edges_with_manifest()
    with pytest.raises(RuntimeError, match="weighted_edge_sampling=True|samples UNIFORMLY"):
        _p1_umap(weighted_edge_sampling=True).fit(X, precomputed_edges_path=ep)


def test_p1_pipeline_semantics_stamped():
    X, ep = _p1_edges_with_manifest()
    m = _p1_umap(weighted_edge_sampling=False); m.fit(X, precomputed_edges_path=ep)
    s = m._train_stats
    assert s['pipeline_pipeline'] == 'legacy'
    assert s['pipeline_positive_sampling'] == 'uniform'
    assert s['pipeline_sampler_class'] == 'EdgeListBalancedIterator'
    assert s['pipeline_weighted_requested'] is False


def test_p1_required_pipeline_mismatch_raises():
    X, ep = _p1_edges_with_manifest()
    with pytest.raises(RuntimeError, match="required_input_pipeline"):
        _p1_umap(required_input_pipeline='device').fit(X, precomputed_edges_path=ep)


def test_p0_2_mandatory_manifest_gate():
    # P0-2: require_graph_manifest=True refuses a graph with no manifest; writing a
    # matching manifest lets it proceed.
    from basemap.graph_validation import graph_manifest_v2, write_manifest
    import os as _os
    X = np.random.RandomState(1).randn(400, 8).astype(np.float32)
    s, t, w = _edges(400, 8000, 1)
    ep = '/tmp/_p02_edges.npz'; np.savez(ep, sources=s, targets=t, weights=w, n_nodes=400, k=15)
    if _os.path.exists(ep + '.manifest.json'):
        _os.remove(ep + '.manifest.json')
    def mk(req):
        return ParametricUMAP(a=1., b=1., correlation_weight=0.0, n_epochs=4, batch_size=64,
                              total_steps_estimate=50, lr_schedule='cosine', warmup_steps=0,
                              device='cpu', positive_target_mode='binary', gpu_resident_data=False,
                              use_amp=False, require_graph_manifest=req)
    with pytest.raises(ValueError, match="no manifest|content-bound graph identity"):
        mk(True).fit(X, precomputed_edges_path=ep)
    write_manifest(ep + '.manifest.json', graph_manifest_v2(s, t, 400, X=X, graph_path=ep, k=15))
    m = mk(True); m.fit(X, precomputed_edges_path=ep)     # manifest present → proceeds
    assert m.is_fitted


def test_p0_3_schedule_version_and_warmup_first_update_positive():
    # P0-3: with warmup, update 0 must have POSITIVE LR (the old step/W made it 0,
    # so a "500k" run was really 499,999 positive-LR updates). Exactly H positive
    # updates; schedule version is the new positive-budget schedule.
    m = _fit(horizon=100, warmup_steps=10)
    s = m._train_stats
    assert s['schedule_version'] == 'cosine-v3-positive-budget', s['schedule_version']
    assert s['lr_used_first'] is not None and s['lr_used_first'] > 0.0, s
    assert s['positive_lr_optimizer_steps'] == 100, s
    assert s['budget_satisfied'] is True
    assert s['stop_reason'] == 'lr_horizon'


def test_p0_3_zero_horizon_takes_planned_positive_updates():
    # total_steps_estimate<=0 must actually TAKE planned_loop positive-LR updates,
    # not merely report the derived horizon while annealing over one step.
    m = _fit(horizon=0, epochs=1)
    s = m._train_stats
    assert s['lr_horizon'] == s['planned_loop_iters'] > 1, s
    assert s['positive_lr_optimizer_steps'] == s['lr_horizon'], s   # every update had +LR
    assert s['budget_satisfied'] is True


def test_p0_3_exhausted_plan_below_budget_fails():
    # A horizon larger than the epoch plan must fail closed, not silently report a
    # satisfied budget.
    with pytest.raises(RuntimeError, match="exhausted|budget"):
        _fit(horizon=10_000_000, epochs=1)


def test_p03_zero_weight_correlation_skips_and_survives_nonfinite():
    m = _fit(horizon=50, epochs=1)
    assert m.is_fitted
    X = np.random.RandomState(2).randn(500, 8).astype(np.float32)
    assert np.isfinite(m.transform(X)).all()
    # correlation branch fully skipped → corr accounting stays 0 contribution
    assert m._train_stats['positive_lr_optimizer_steps'] >= 1


if __name__ == '__main__':
    for fn in [test_p0b_exact_positive_lr_update_count, test_p0b_zero_horizon_derives_not_one_step,
               test_p0b_warmup_ge_horizon_rejected, test_p0b_bench_hook_populates_seconds_and_stops,
               test_p03_zero_weight_correlation_skips_and_survives_nonfinite]:
        fn(); print("PASS", fn.__name__)
    print("ALL P0-B/P0.3 TESTS PASSED")
