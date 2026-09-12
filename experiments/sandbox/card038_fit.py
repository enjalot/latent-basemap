"""Card038 shared production fit configuration. The trainer, device canary and preflight ALL configure their
ParametricUMAP through configure_pumap() so they exercise identical production behavior (width, half-radii
Hook C, LR/wd/clip/rankneg/fneg/tanh, device_fp16, checkpoint targets, admission identity). No experimental
engine — this is the proven core fit (card033 @ 8b7ecab)."""
import card038_validate as V


def configure_pumap(pumap, arm, steps, radii, identity, ckpt_targets):
    """Set the exact production recipe onto a freshly-loaded champion (pumap.model already reset to None).
    Width via hidden_dim; original arch/neck/n_layers kept. Returns the configured pumap."""
    pumap.model = None; pumap.n_components = V.NC; pumap.hidden_dim = V.WIDTH[arm]
    pumap.learning_rate = V.LR; pumap.lr_schedule = "constant"; pumap.batch_size = V.BATCH; pumap.warmup_steps = 0
    pumap.n_epochs = 100000; pumap._max_train_steps = steps; pumap.rankneg_window = V.RANKNEG
    pumap.fneg_weight = V.FNEG; pumap.neg_tanh_gamma = V.TANH
    pumap.x_residency = "auto"; pumap.required_input_pipeline = "device"
    assert abs(float(getattr(pumap, "pos_ratio", -1)) - V.POS_RATIO) < 1e-9, "pos_ratio must be .1"
    for a, v in (("anchor_ids_path", ""), ("anchor_hold_weight", 0.0), ("replay_bank_path", ""),
                 ("replay_weight", 0.0), ("deriv_bank_path", ""), ("deriv_weight", 0.0),
                 ("midnear_enabled", False), ("density_weight", 0.0), ("correlation_weight", 0.0)):
        if hasattr(pumap, a): setattr(pumap, a, v)
    pumap.gpu_resident_vram_budget_gb=14.0
    pumap.positive_target_mode="binary"
    pumap._card038_observe=True
    pumap._card013_radii = radii                       # Hook C half-strength local scale
    pumap._checkpoint_step_targets = set(ckpt_targets); pumap._card012_identity = identity
    return pumap
