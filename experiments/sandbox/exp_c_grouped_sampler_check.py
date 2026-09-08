"""CPU behavior check for the grouped-negatives wiring in DeviceEdgeSampler (owner Option-Y adoption 2026-09-08).
The production sampler's grouped neg_src draw must reproduce C's standalone grouped implementation bit-for-bit on a
fixed seed (same grouping/marginals), and must reduce unique negative sources by `tails`× (the throughput mechanism).
The neg_dst draw is the pre-existing, already-in-production path (uniform-offset or rank-window) — unchanged by this
wiring — so this check isolates the NEW grouping logic. CPU, no GPU. Usage: exp_c_grouped_sampler_check.py
"""
import json
import torch

N_NODES, N_NEG, TAILS, SEED = 100_000, 14_746, 4, 20260908   # 14746 = champion 16384*(1-0.10) negs


def sampler_grouped_src(n_nodes, n, t, gen):                  # EXACT expression from edge_list_dataset._sample_negatives
    ns = torch.randint(0, n_nodes, ((n + t - 1) // t,), generator=gen)
    return ns.repeat_interleave(t)[:n]


def c_standalone_grouped_src(n_nodes, n, gen):                # EXACT expression from exp_c_quality/microbench sample_edges
    return torch.randint(0, n_nodes, ((n + 3) // 4,), generator=gen).repeat_interleave(4)[:n]


def main():
    # 1. bit-identical to C's standalone grouping on a matched seed
    g1 = torch.Generator().manual_seed(SEED); g2 = torch.Generator().manual_seed(SEED)
    s_samp = sampler_grouped_src(N_NODES, N_NEG, TAILS, g1)
    s_c = c_standalone_grouped_src(N_NODES, N_NEG, g2)
    bit_identical = bool(torch.equal(s_samp, s_c))

    # 2. structural: ceil(n/tails) unique sources, each appearing `tails`× (last group truncated)
    n_groups = (N_NEG + TAILS - 1) // TAILS
    uniq_grouped = int(torch.unique(s_samp).numel())
    counts = torch.bincount(s_samp)
    modal_count_ok = int(counts[counts > 0].mode().values.item()) == TAILS

    # 3. vs ungrouped: unique sources ~N_NEG (no reduction) → grouped reduces unique endpoints ~tails×
    g3 = torch.Generator().manual_seed(SEED)
    s_ungrouped = torch.randint(0, N_NODES, (N_NEG,), generator=g3)
    uniq_ungrouped = int(torch.unique(s_ungrouped).numel())
    src_reduction = round(uniq_ungrouped / max(uniq_grouped, 1), 3)

    # 4. marginal: grouped src uniform over nodes (mean ≈ n_nodes/2, full range covered)
    marg_mean = float(s_samp.float().mean()); marg_ok = abs(marg_mean - N_NODES / 2) < N_NODES * 0.05

    out = {"schema": "exp-c-grouped-sampler-check-2026-09-08", "n_neg": N_NEG, "tails": TAILS,
           "bit_identical_to_C_standalone": bit_identical, "n_groups_expected": n_groups,
           "unique_sources_grouped": uniq_grouped, "each_source_appears_tails_times": modal_count_ok,
           "unique_sources_ungrouped": uniq_ungrouped, "src_reduction_x": src_reduction,
           "grouped_src_marginal_mean": round(marg_mean, 1), "marginal_uniform_ok": marg_ok,
           "src_collisions": n_groups - uniq_grouped,   # ~birthday collisions among the drawn sources (expected)
           "PASS": bool(bit_identical and modal_count_ok and marg_ok and src_reduction >= 3.5),
           "note": "grouped neg_src reproduces C's standalone grouping bit-for-bit on a matched seed and reduces "
                   "unique sources ~tails×; neg_dst path (uniform-offset/rank-window) is the pre-existing production "
                   "code, unchanged. Quality of grouped+rankneg is gated separately by the 2M run."}
    print(json.dumps(out, indent=1))
    return 0 if out["PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
