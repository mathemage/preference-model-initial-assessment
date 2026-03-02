# Task: Implement a Llama-Style Transformer Decoder Block from Scratch

Your task is to implement a complete Llama-style transformer decoder block from
scratch in PyTorch, including all modern architectural components.

You must create a file `/solution/transformer_block.py` that contains the
following classes, all implemented from scratch using **only** `torch`,
`torch.nn`, `torch.nn.functional`, `math`, and the Python standard library.
You may **not** use `transformers`, `xformers`, `flash_attn`, or any other
third-party ML library.

---

## 1. `RMSNorm(nn.Module)` — Root Mean Square Layer Normalization

```python
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        ...
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...
```

- Computes: `x * weight / sqrt(mean(x^2) + eps)`
- `weight` is a learnable parameter of shape `(dim,)` initialized to ones.

---

## 2. `RotaryPositionEmbedding(nn.Module)` — Rotary Position Embeddings (RoPE)

```python
class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        ...
    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ...
```

- Precomputes frequency tensors for rotary embeddings.
- `forward` returns `(cos, sin)` tensors of shape `(seq_len, dim)`.
- Frequencies are computed as: `theta_i = 1 / (base ^ (2i / dim))` for `i = 0, 1, ..., dim/2 - 1`.
- The positions `t = 0, 1, ..., seq_len - 1` are combined with frequencies via outer product.
- `cos` and `sin` are then computed from these combined values, repeated to fill the full `dim`.

You must also provide a standalone function:

```python
def apply_rotary_emb(
    x: torch.Tensor,       # (batch, num_heads, seq_len, head_dim)
    cos: torch.Tensor,     # (seq_len, head_dim)
    sin: torch.Tensor,     # (seq_len, head_dim)
) -> torch.Tensor:
    ...
```

This applies the rotation by:
1. Splitting `x` into two halves: `(x_first, x_second) = (x[..., :dim//2], x[..., dim//2:])`
2. Forming a rotated version: `x_rotated = cat([-x_second, x_first], dim=-1)`
3. Computing: `x * cos + x_rotated * sin`

---

## 3. `GroupedQueryAttention(nn.Module)` — Grouped-Query Attention (GQA)

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int):
        ...
    def forward(
        self,
        x: torch.Tensor,                                           # (batch, seq_len, dim)
        cos: torch.Tensor,                                         # (seq_len, head_dim)
        sin: torch.Tensor,                                         # (seq_len, head_dim)
        mask: Optional[torch.Tensor] = None,                       # (seq_len, seq_len)
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        ...
```

- `num_heads` query heads, `num_kv_heads` key/value heads.
- `num_heads` must be divisible by `num_kv_heads`.
- `head_dim = dim // num_heads`.
- Linear projections: `W_q` of shape `(dim, num_heads * head_dim)`, `W_k` of shape `(dim, num_kv_heads * head_dim)`, `W_v` of shape `(dim, num_kv_heads * head_dim)`, `W_o` of shape `(dim, dim)`.
- Apply RoPE to queries and keys.
- Repeat key/value heads to match query heads: each KV head is shared by `num_heads // num_kv_heads` query heads.
- Compute scaled dot-product attention: `softmax(Q @ K^T / sqrt(head_dim)) @ V`.
- When `mask is None` and `kv_cache is None`, apply a **causal mask** (upper-triangular with -inf).
- When `kv_cache` is provided, concatenate new K, V with cached K, V and return the updated cache.
- Returns `(output, kv_cache)` where `kv_cache` is `(K, V)` or `None`.

---

## 4. `SwiGLU(nn.Module)` — SwiGLU Feed-Forward Network

```python
class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        ...
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...
```

- The effective hidden dimension is: `int(2 * hidden_dim / 3)` rounded up to the nearest multiple of 256.
- Three linear projections (no bias): `W_gate (dim → effective_hidden)`, `W_up (dim → effective_hidden)`, `W_down (effective_hidden → dim)`.
- Computation: `W_down(SiLU(W_gate(x)) * W_up(x))`

---

## 5. `LlamaBlock(nn.Module)` — The Complete Decoder Block

```python
class LlamaBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, hidden_dim: int):
        ...
    def forward(
        self,
        x: torch.Tensor,                                           # (batch, seq_len, dim)
        cos: torch.Tensor,                                         # (seq_len, head_dim)
        sin: torch.Tensor,                                         # (seq_len, head_dim)
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        ...
```

Architecture (pre-norm transformer):
1. `h = x + GQA(RMSNorm(x), cos, sin, mask, kv_cache)`
2. `out = h + SwiGLU(RMSNorm(h))`
3. Returns `(out, kv_cache)`

---

## Summary of Requirements

| Requirement | Detail |
|---|---|
| File location | `/solution/transformer_block.py` |
| Allowed imports | `torch`, `torch.nn`, `torch.nn.functional`, `math`, `typing` |
| Disallowed imports | `transformers`, `xformers`, `flash_attn`, any other ML library |
| Dtypes | Must work with `torch.float32` and `torch.float16` |
| Differentiability | All modules must support backward pass |
| KV-cache | Must correctly concatenate and return updated caches |
| Causal masking | Must apply causal mask when no explicit mask and no kv_cache |
