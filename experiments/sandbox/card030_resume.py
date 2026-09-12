N=300000;BATCH=16384;SEED=42;LR=.001;POS_RATIO=.1
PARENT_RECIPE = {"architecture":"residual_bottleneck","neck_fraction":0.75,"input_dim":1536,"n_components":2,
 "hidden_dim":2048,"n_layers":3,"n_neighbors":15,"a":1.9328,"b":0.7905,"low_dim_kernel":"umap",
 "kernel_alpha":1.0,"correlation_weight":0.0,"use_batchnorm":False,"use_dropout":False,
 "clip_grad_norm":1.0,"clip_grad_value":None,"pos_ratio":0.1,"positive_target_mode":"binary",
 "density_weight":0.0,"midnear_enabled":False,"fneg_weight":1.0,"fneg_lo":0.1,"fneg_hi":0.4,
 "rankneg_window":75000,"rankneg_exclude_neighbors":False,"neg_tanh_gamma":4.0,"kernel_anneal_frac":0.0}
def validate_recipe(p):
    for k,v in PARENT_RECIPE.items():
        assert getattr(p,k,None)==v, f"inherited recipe mismatch {k}: {getattr(p,k,None)!r} != {v!r}"
    assert getattr(p,"_kernel_radii",None) is None, "unexpected local-scale radii"
    return dict(PARENT_RECIPE)


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
    assert cfg.get("final_activation") in ["relu","leaky_relu_slope_0p01"], "missing activation identity"
    assert cfg.get("batch_size") == BATCH and cfg.get("random_state") == SEED and cfg.get("architecture") == "residual_bottleneck", "checkpoint sampler/model config"
    assert cfg.get("learning_rate") == LR and cfg.get("lr_schedule") == "constant", "checkpoint LR identity"
    assert cfg.get("replay_enabled") is False and cfg.get("deriv_enabled") is False, "unexpected preservation treatment"
    opt = ck["optimizer"]; groups = opt.get("param_groups", []); states = opt.get("state", {})
    assert groups and states, "empty optimizer state/groups"
    pids = [i for g in groups for i in g.get("params", [])]
    assert len(pids) == len(set(pids)) == len(ck["model"]) and set(pids) == set(states), "optimizer parameter identity"
    for g in groups:
        assert all(k in g for k in ["lr", "betas", "eps", "weight_decay", "amsgrad", "params"]), "incomplete Adam group"
        assert g["lr"] == LR and g["eps"] > 0 and len(g["betas"]) == 2, "Adam group configuration"
    for pid, parameter in zip(pids, ck["model"].values()):
        state = states[pid]
        assert all(k in state for k in ["step", "exp_avg", "exp_avg_sq"]), "missing Adam moments"
        assert torch.is_tensor(state["step"]) and state["step"].numel() == 1 and float(state["step"]) == step, "Adam successful dose"
        for key in ["exp_avg", "exp_avg_sq"]:
            assert state[key].shape == parameter.shape and bool(torch.isfinite(state[key]).all()), "Adam moment shape/nonfinite"
    sched = ck["scheduler"]
    assert all(k in sched for k in ["base_lrs", "last_epoch", "_step_count", "_last_lr", "lr_lambdas"]), "incomplete constant scheduler"
    assert len(sched["base_lrs"]) == len(groups) == len(sched["_last_lr"]) and all(x == LR for x in sched["base_lrs"] + sched["_last_lr"]), "scheduler LR mismatch"
    assert isinstance(sched["last_epoch"], int) and isinstance(sched["_step_count"], int) and sched["_step_count"] >= 1, "scheduler counters"
    cpu = ck["torch_rng"]
    assert torch.is_tensor(cpu) and cpu.dtype == torch.uint8 and cpu.ndim == 1, "CPU RNG dtype/shape"
    torch.Generator(device="cpu").set_state(cpu.cpu())
    def cuda_state(v):
        return torch.is_tensor(v) and v.dtype == torch.uint8 and v.ndim == 1 and v.numel() == 16
    assert isinstance(ck["cuda_rng"], (list, tuple)) and len(ck["cuda_rng"]) >= 1 and all(cuda_state(v) for v in ck["cuda_rng"]), "CUDA RNG schema"
    assert cuda_state(ck["loader_gen"]), "device loader RNG schema"
    for k in ["loader_perm", "loader_pos_idx", "loader_batch_no", "loader_rank_of_node", "loader_node_at_rank", "rankneg_scale"]:
        assert k in ck and ck[k] is not None, "missing PERM/rank continuation field: " + k
    perm = ck["loader_perm"]; n_edges = 15 * n_nodes
    assert torch.is_tensor(perm) and perm.dtype == torch.int64 and perm.shape == (n_edges,), "PERM shape/dtype"
    assert int(perm.min()) >= 0 and int(perm.max()) < n_edges, "PERM bounds"
    assert isinstance(ck["loader_pos_idx"], int) and 0 <= ck["loader_pos_idx"] < n_edges + int(BATCH * POS_RATIO), "PERM cursor"
    assert isinstance(ck["loader_batch_no"], int) and ck["loader_batch_no"] >= 0, "loader batch cursor"
    rank, node = ck["loader_rank_of_node"], ck["loader_node_at_rank"]
    assert all(torch.is_tensor(v) and v.dtype == torch.int64 and v.shape == (n_nodes,) for v in [rank, node]), "rank-order shape/dtype"
    assert int(rank.min()) >= 0 and int(rank.max()) < n_nodes and int(node.min()) >= 0 and int(node.max()) < n_nodes, "rank-order bounds"
    assert torch.equal(rank[node], torch.arange(n_nodes)), "rank-order inverse mismatch"
    assert np.isfinite(ck["rankneg_scale"]) and ck["rankneg_scale"] > 0, "rank scale invalid"


    scaler=ck.get("scaler")
    assert isinstance(scaler,dict) and all(k in scaler for k in ["scale","growth_factor","backoff_factor","growth_interval","_growth_tracker"]), "AMP scaler state absent"
    assert all(np.isfinite(v) for v in scaler.values()) and scaler["scale"]>0, "invalid scaler state"
    assert len(groups)==1 and groups[0]["weight_decay"]==.01, "model-only parent AdamW configuration"
