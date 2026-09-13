from card085_common import N,BATCH,SEED,LR
POS_RATIO=.1
def validate_resume_payload(ck, step, n_nodes=None):
    """Validate the actual PERM/device checkpoint schema observed in real Card019 step20K.
    CPU RNG is load-tested; CUDA Philox states are checked against this runtime's observed16-byte schema.
    This does not substitute for the mandatory real GPU resume twin.
    """
    import torch, numpy as np
    n_nodes = N if n_nodes is None else n_nodes
    assert ck.get("schema") == "pumap-ckpt-2026-08-30", "checkpoint schema"
    assert isinstance(ck.get("epoch"), int) and ck["epoch"] >= 0, "missing epoch"
    ts = ck.get("train_stats", {})
    assert ts.get("executed_iters") == ts.get("positive_lr_optimizer_steps") == ts.get("optimizer_steps_succeeded") == step, "checkpoint successful counters"
    cfg = ck["config"]
    ident = ck.get("card012_identity", {})
    assert ident.get("gpu_resident_vram_budget_gb") == (10.0 if ident["arm"]=="fresh" else 14.0), "resident budget identity drift"
    assert ident.get("fneg_weight")==1.0 and ident.get("neg_tanh_gamma")==4.0 and ident.get("positive_target_mode")=="binary", "negative/positive recipe identity drift"
    assert cfg.get("batch_size") == BATCH and cfg.get("random_state") == SEED and cfg.get("architecture") == "residual_bottleneck", "checkpoint sampler/model config"
    assert cfg.get("replay_enabled") == bool(ident["replay_weight"]) and cfg.get("deriv_enabled") is False, "unexpected preservation treatment"
    opt = ck["optimizer"]; groups = opt.get("param_groups", []); states = opt.get("state", {})
    assert groups and states, "empty optimizer state/groups"
    pids = [i for g in groups for i in g.get("params", [])]
    assert len(pids) == len(set(pids)) == len(ck["model"]) and set(pids) == set(states), "optimizer parameter identity"
    for g in groups:
        assert all(k in g for k in ["lr", "betas", "eps", "weight_decay", "amsgrad", "params"]), "incomplete Adam group"
        assert g["lr"] == LR[ident['arm']] and g["weight_decay"] == .01 and g["eps"] > 0 and len(g["betas"]) == 2, "Adam group configuration"
    for pid, parameter in zip(pids, ck["model"].values()):
        state = states[pid]
        assert all(k in state for k in ["step", "exp_avg", "exp_avg_sq"]), "missing Adam moments"
        assert torch.is_tensor(state["step"]) and state["step"].numel() == 1 and float(state["step"]) == step, "Adam successful dose"
        for key in ["exp_avg", "exp_avg_sq"]:
            assert state[key].shape == parameter.shape and bool(torch.isfinite(state[key]).all()), "Adam moment shape/nonfinite"
    sched = ck["scheduler"]
    assert all(k in sched for k in ["base_lrs", "last_epoch", "_step_count", "_last_lr", "lr_lambdas"]), "incomplete constant scheduler"
    assert len(sched["base_lrs"]) == len(groups) == len(sched["_last_lr"]) and all(x == LR[ident['arm']] for x in sched["base_lrs"] + sched["_last_lr"]), "scheduler LR mismatch"
    assert isinstance(sched["last_epoch"], int) and isinstance(sched["_step_count"], int) and sched["_step_count"] >= 1, "scheduler counters"
    scaler=ck.get("scaler")
    assert isinstance(scaler,dict) and all(k in scaler for k in ["scale","growth_factor","backoff_factor","growth_interval","_growth_tracker"]), "missing AMP scaler continuation state"
    assert all(np.isfinite(scaler[k]) for k in scaler) and scaler["scale"]>0, "invalid AMP scaler"
    cpu = ck["torch_rng"]
    assert torch.is_tensor(cpu) and cpu.dtype == torch.uint8 and cpu.ndim == 1, "CPU RNG dtype/shape"
    torch.Generator(device="cpu").set_state(cpu.cpu())
    def cuda_state(v):
        return torch.is_tensor(v) and v.dtype == torch.uint8 and v.ndim == 1 and v.numel() == 16
    assert isinstance(ck["cuda_rng"], (list, tuple)) and len(ck["cuda_rng"]) >= 1 and all(cuda_state(v) for v in ck["cuda_rng"]), "CUDA RNG schema"
    assert cuda_state(ck["loader_gen"]), "device loader RNG schema"
    if cfg["replay_enabled"]:assert cuda_state(ck["replay_gen"]), "replay RNG schema"
    else:assert ck["replay_gen"] is None, "disabled replay has generator"
    for k in ["loader_perm", "loader_pos_idx", "loader_batch_no"]:
        assert k in ck and ck[k] is not None, "missing PERM/rank continuation field: " + k
    perm = ck["loader_perm"]; n_edges = 15 * n_nodes
    assert torch.is_tensor(perm) and perm.dtype == torch.int64 and perm.shape == (n_edges,), "PERM shape/dtype"
    assert int(perm.min()) >= 0 and int(perm.max()) < n_edges, "PERM bounds"
    assert isinstance(ck["loader_pos_idx"], int) and 0 <= ck["loader_pos_idx"] < n_edges + int(BATCH * POS_RATIO), "PERM cursor"
    assert isinstance(ck["loader_batch_no"], int) and ck["loader_batch_no"] >= 0, "loader batch cursor"
    assert ident['rankneg_window']==0, 'uniform policy identity'
    assert ck['loader_rank_of_node'] is None and ck['loader_node_at_rank'] is None and ck['rankneg_scale'] is None, 'uniform policy unexpectedly retains ranking'
