"""Optional base negative multiplier; band addition and normalization remain in core."""
import math
import torch
def override_negative_base(weights, negative_mask, value):
 if not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
  raise ValueError('base negative weight must be finite and positive')
 return torch.where(negative_mask, torch.full_like(weights,float(value)), weights)
