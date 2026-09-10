"""Card005 preregistered arrival gate on balanced, paired query cohorts."""
import numpy as np


def arrival_gate(arm, frozen, sources, arriving, seed=0):
    arm, frozen, sources = map(np.asarray, (arm, frozen, sources))
    if arm.ndim != 1 or arm.shape != frozen.shape or arm.shape != sources.shape:
        raise ValueError("Expected matching one-dimensional query arrays")
    cohorts = [(arm - frozen)[sources == c] for c in arriving]
    counts = [len(d) for d in cohorts]
    # Pooling equals the preregistered equal-source mean only on this balanced seal.
    if not counts or min(counts) == 0 or len(set(counts)) != 1:
        raise ValueError("Card005 requires nonempty, equally sized arriving cohorts")
    delta = np.concatenate(cohorts)
    if not np.isfinite(delta).all():
        raise ValueError("Non-finite paired arrival metrics")
    rng = np.random.default_rng(seed)
    bootstrap = np.array([
        delta[rng.integers(0, delta.size, delta.size)].mean() for _ in range(2000)
    ])
    gain = float(delta.mean())
    ci = np.percentile(bootstrap, [2.5, 97.5]).tolist()
    # Apply thresholds before display rounding and use this arm's own interval.
    return {"arriving_agg_gain_B250": gain, "arriving_agg_ci95_B250": ci,
            "gate_new_source": bool(gain >= 0.03 and ci[0] > 0)}
