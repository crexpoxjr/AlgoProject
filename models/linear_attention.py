"""Non-causal, normalized multi-head kernel attention in linear time in T.

For phi(x) = ELU(x) + 1, each head computes
  y_i = phi(q_i) (sum_j phi(k_j)^T v_j) / (phi(q_i) sum_j phi(k_j)).
The key/value summaries have no T-by-T dimension. This is a different kernel
from softmax attention, not an exact reordering of softmax attention.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearSelfAttention(nn.Module):
    def __init__(self, d_model=64, n_heads=4, eps=1e-6):
        super().__init__()
        if n_heads <= 0 or d_model <= 0 or d_model % n_heads:
            raise ValueError('d_model must be positive and divisible by positive n_heads')
        if eps <= 0:
            raise ValueError('eps must be positive')
        self.n_heads, self.d_head, self.eps = n_heads, d_model // n_heads, eps
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        b, t, d = x.shape
        if t == 0:
            raise ValueError('sequence length must be positive')
        qkv = self.qkv(x).reshape(b, t, 3, self.n_heads, self.d_head)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q, k = F.elu(q) + 1, F.elu(k) + 1
        # Summaries are (B,H,D,D) and (B,H,D), independent of T.
        kv = k.transpose(-2, -1) @ v
        numerator = q @ kv
        denominator = (q * k.sum(dim=-2, keepdim=True)).sum(dim=-1, keepdim=True)
        out = numerator / denominator.clamp_min(self.eps)
        return self.out(out.transpose(1, 2).reshape(b, t, d))
