"""
Reference implementation of a Llama-style transformer decoder block.

This serves as a known-correct solution for testing the judge.  The LLM agent
in the RL environment is expected to produce an equivalent implementation.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE)."""

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len

    def forward(
        self, x: torch.Tensor, seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)          # (seq_len, dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)         # (seq_len, dim)
        return emb.cos(), emb.sin()


def apply_rotary_emb(
    x: torch.Tensor,       # (batch, num_heads, seq_len, head_dim)
    cos: torch.Tensor,     # (seq_len, head_dim)
    sin: torch.Tensor,     # (seq_len, head_dim)
) -> torch.Tensor:
    """Apply rotary embeddings to *x* using the rotate-half approach."""
    half = x.shape[-1] // 2
    x_rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return x * cos + x_rotated * sin


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads *n_rep* times to match the number of query heads."""
    if n_rep == 1:
        return x
    batch, num_kv_heads, seq_len, head_dim = x.shape
    return (
        x[:, :, None, :, :]
        .expand(batch, num_kv_heads, n_rep, seq_len, head_dim)
        .reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)
    )


class GroupedQueryAttention(nn.Module):
    """Grouped-Query Attention (GQA)."""

    def __init__(self, dim: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        assert num_heads % num_kv_heads == 0
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.n_rep = num_heads // num_kv_heads

        self.w_q = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.w_k = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.w_v = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.w_o = nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch, seq_len, _ = x.shape

        q = self.w_q(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        new_kv_cache = (k, v)

        k = _repeat_kv(k, self.n_rep)
        v = _repeat_kv(v, self.n_rep)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores + mask
        elif kv_cache is None:
            causal = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=x.device, dtype=scores.dtype),
                diagonal=1,
            )
            scores = scores + causal

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.w_o(out), new_kv_cache


class SwiGLU(nn.Module):
    """SwiGLU Feed-Forward Network."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        effective = int(2 * hidden_dim / 3)
        # Round up to the nearest multiple of 256
        effective = ((effective + 255) // 256) * 256
        self.w_gate = nn.Linear(dim, effective, bias=False)
        self.w_up = nn.Linear(dim, effective, bias=False)
        self.w_down = nn.Linear(effective, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class LlamaBlock(nn.Module):
    """Complete Llama-style transformer decoder block."""

    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, hidden_dim: int):
        super().__init__()
        self.attention_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)
        self.attention = GroupedQueryAttention(dim, num_heads, num_kv_heads)
        self.ffn = SwiGLU(dim, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        h, kv_cache = self.attention(self.attention_norm(x), cos, sin, mask, kv_cache)
        h = x + h
        out = h + self.ffn(self.ffn_norm(h))
        return out, kv_cache
