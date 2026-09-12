"""Standalone compact projector and an APPA-inspired, fixed-grid density penalty."""
import torch
from torch import nn


class CompactProjector(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1536, 1024), nn.SiLU(),
                                 nn.Linear(1024, 512), nn.SiLU(),
                                 nn.Linear(512, 256), nn.SiLU(),
                                 nn.Linear(256, 2), nn.Sigmoid())

    def forward(self, x):
        hidden = self.net[:-2](x)
        # Keep coordinates finer than bf16 bins; hidden matmuls may remain autocast.
        with torch.autocast(device_type=x.device.type, enabled=False):
            return self.net[-1](self.net[-2](hidden.float()))


def interpolate_density(grid, xy):
    """Bilinear interpolation at grid nodes spanning [0,1], without grid gradients."""
    n = grid.shape[0]
    assert grid.shape == (n, n) and not grid.requires_grad
    uv = xy * (n - 1)
    lo = uv.floor().long().clamp(0, n - 2)
    delta = uv - lo.to(uv.dtype)
    ix, iy = lo.unbind(1); dx, dy = delta.unbind(1)
    return ((1 - dx) * (1 - dy) * grid[ix, iy] + dx * (1 - dy) * grid[ix + 1, iy]
            + (1 - dx) * dy * grid[ix, iy + 1] + dx * dy * grid[ix + 1, iy + 1])


def density_penalty(grid, xy):
    logp = interpolate_density(grid, xy.float()).clamp_min(1e-12).log()
    return (-2 - logp).clamp_min(0).mean()


def loss_for(arm, pred, target, grid):
    pred = pred.float(); target = target.float()
    if arm == 'compact_mse':
        base = (pred - target).square().mean()
    else:
        base = (pred - target).abs().mean()
    prior = density_penalty(grid, pred) if arm == 'compact_l1_prior' else pred.new_zeros(())
    return base + .002 * prior, base, prior
