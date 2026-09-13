"""Card084 prospective stateless positive-edge objective; never calibrates itself."""
import math
import torch


def add_attraction(base, src, dst, positive, pair_scale, *, coefficient=0.,
                   family='pseudo_huber', delta=None):
    # Exact off path: no validation, allocation, graph node, data read or RNG draw.
    if coefficient == 0.:
        return base
    if not math.isfinite(coefficient) or coefficient < 0:
        raise ValueError('coefficient must be finite and nonnegative')
    if family not in ('quadratic', 'pseudo_huber'):
        raise ValueError('unknown attraction family')
    if family == 'pseudo_huber' and (delta is None or not math.isfinite(delta) or delta <= 0):
        raise ValueError('delta must be frozen, finite and positive')
    if pair_scale is None:
        raise ValueError('radius products required')
    mask = positive.detach().bool()
    if not bool(mask.any()):
        raise ValueError('positive graph edges required')
    # Preserve double for CPU gradcheck; promote AMP outputs before subtraction.
    dtype = torch.float64 if src.dtype == torch.float64 else torch.float32
    scale = pair_scale.detach()[mask].to(dtype)
    if not bool(torch.isfinite(scale).all() and (scale > 0).all()):
        raise ValueError('invalid radius products')
    diff = src[mask].to(dtype) - dst[mask].to(dtype)
    u = diff.square().sum(-1) / scale
    if family == 'quadratic':
        values = u / 2
    else:
        # Rationalized pseudo-Huber: no sqrt(u), no cancellation near zero.
        values = u / (torch.sqrt(1 + u / (delta * delta)) + 1)
    return base + coefficient * values.mean()
