# Preference Model Initial Assessment — Answers

## 1. Environment Description

**Environment: Implement a Llama-Style Transformer Decoder Block from Scratch**

The LLM agent must implement five core components of modern transformer
architectures — RMSNorm, Rotary Position Embeddings (RoPE), Grouped-Query
Attention (GQA), SwiGLU feed-forward network, and a complete LlamaBlock
decoder layer — entirely from scratch in PyTorch, without using any
third-party ML libraries.

### What makes this environment interesting?

1. **Depth of understanding**: The task requires grasping multiple interacting
   sub-systems.  Getting any single component wrong causes cascading failures.
2. **Numerical subtlety**: RoPE involves complex-number-like rotations, GQA
   requires correct head-group broadcasting, and the KV-cache must be managed
   precisely during autoregressive generation.
3. **Challengingly composable**: Each component is non-trivial on its own
   (e.g., applying rotary embeddings, implementing causal masking with a
   KV-cache), but making them all work together inside a pre-norm residual
   block adds another layer of difficulty.
4. **Directly practical**: These exact components power Llama 2/3, Mistral,
   and many other production LLMs.
5. **Verifiable without external data**: The judge only needs PyTorch and
   random inputs — no datasets or pre-trained weights.

---

## 2. Tools, Packages, and Data

| Requirement | Detail |
|---|---|
| **Python** | 3.10+ |
| **PyTorch** | ≥ 2.0 (CPU is sufficient; no GPU needed) |
| **Standard lib** | `math`, `typing` |
| **Data** | None — the judge generates random test tensors with fixed seeds |

No GPUs are required for this environment.  All tests run on CPU.

---

## 3. Prompt

See [`environment/prompt.md`](environment/prompt.md) for the full prompt.

In summary, the LLM is told to create `/solution/transformer_block.py`
containing five `nn.Module` classes (`RMSNorm`, `RotaryPositionEmbedding`,
`GroupedQueryAttention`, `SwiGLU`, `LlamaBlock`) and a helper function
(`apply_rotary_emb`), all implemented from scratch using only `torch`,
`torch.nn`, `torch.nn.functional`, and the Python standard library.

The prompt specifies exact class names, constructor signatures, forward-method
signatures, mathematical formulas, and architectural constraints (pre-norm
residual connections, causal masking, KV-cache support, etc.).

---

## 4. Judge Description

The judge (`environment/judge.py`) performs the following steps:

1. **File existence** — Check that `/solution/transformer_block.py` exists.
   If not, score = 0.
2. **Static import analysis** — Parse the AST and verify no disallowed
   imports (`transformers`, `xformers`, `flash_attn`, etc.).  If found,
   score = 0.
3. **Dynamic import** — Import the module.  If an exception occurs, score = 0.
4. **Structure suite** — Verify all five classes exist and are `nn.Module`
   subclasses, and that `apply_rotary_emb` is callable.
5. **RMSNorm suite** — Test output shape, value on constant input, and
   gradient flow.
6. **RoPE suite** — Test cos/sin shapes, `cos²+sin²=1` identity, norm
   preservation after rotation, distinct embeddings for distinct positions,
   and gradient flow.
7. **GQA suite** — Test output shape for MHA (num_kv_heads == num_heads) and
   MQA (num_kv_heads == 1), causal masking (prefix outputs match truncated
   inputs), KV-cache correctness (step-by-step decoding matches full-sequence
   output), and gradient flow.
8. **SwiGLU suite** — Test output shape, that the effective hidden dimension
   is a multiple of 256, manual formula verification (`W_down(SiLU(W_gate(x))
   * W_up(x))`), and gradient flow.
9. **LlamaBlock suite** — Test output shape, non-zero output, KV-cache
   step-by-step vs. full-sequence equivalence, float16 support, gradient
   flow, and that all parameters receive gradients.

Each suite gets a score = (tests passed) / (tests total).  The final score is
the arithmetic mean of all suite scores, yielding a continuous value in [0, 1].

**What causes the LLM to fail?**
- Missing solution file or disallowed imports → score 0.
- Any class or function missing → reduced structure score.
- Incorrect math (wrong RMSNorm formula, broken RoPE rotation, wrong
  attention scaling, bad causal mask) → reduced suite scores.
- Broken KV-cache (step-by-step ≠ full sequence) → GQA and LlamaBlock tests
  fail.
- Non-differentiable operations → gradient tests fail.

**What causes the LLM to succeed?**
A numerically correct, from-scratch implementation of all five components with
proper causal masking, KV-cache management, and differentiability.

---

## 5. Reward Hacking and Reward Denial Analysis

### Reward hacking (false positives)

| Attack vector | Mitigation |
|---|---|
| **Wrapping a library**: The LLM could import `transformers.models.llama` and wrap it. | The judge performs static AST analysis to detect disallowed imports. |
| **Hardcoded outputs**: The LLM could detect test inputs and return expected values. | Tests use `torch.manual_seed` with specific seeds, but also use random inputs without fixed seeds. The variety of test shapes and configurations (different num_heads, num_kv_heads, seq_len) makes hardcoding impractical. |
| **Partial implementation**: The LLM could implement only the tested properties (e.g., only shape correctness). | The judge tests mathematical correctness with formula verification (SwiGLU formula check, RMSNorm constant-input check) and functional equivalence (KV-cache vs. full-sequence). |

The risk of reward hacking is **low** because the judge tests fundamental
mathematical properties and functional equivalence across multiple
configurations, not just surface-level output format.

### Reward denial (false negatives)

| Risk | Mitigation |
|---|---|
| **Numerical precision**: Floating-point differences between implementations. | The judge uses `torch.allclose` with generous tolerances (atol=1e-4, rtol=1e-3) and even more generous for float16. |
| **Implementation-equivalent alternatives**: A correct solution that uses a different but mathematically equivalent formula. | The tests check *properties* (norm preservation, cos²+sin²=1, causal masking behavior, KV-cache equivalence) rather than bit-exact outputs, so equivalent formulations will pass. |
| **Constructor naming**: A correct solution with different attribute names. | The SwiGLU formula test directly accesses `w_gate`, `w_up`, `w_down` — if the LLM names them differently, this specific test will fail. However, the prompt explicitly specifies these names, and this is only one test among many. |

The risk of reward denial is **low** because the tests focus on behavioral
properties rather than implementation details, and use tolerant numerical
comparisons.

---

## 6. Why This Environment?

I chose this environment because:

1. **Direct relevance to my experience**: I have hands-on experience with
   transformer architectures, having worked on LLM training and inference
   optimization.  Implementing these components from scratch is something I
   have done in practice.
2. **Tests real understanding**: This task cannot be solved by pattern
   matching or memorization alone — it requires understanding the mathematics
   of rotary embeddings, the mechanics of grouped-query attention, and the
   subtleties of KV-cache management.
3. **Challenging for LLMs**: In my experience, current LLMs frequently make
   errors in RoPE implementation (especially the interleaving pattern),
   KV-cache management (off-by-one errors, wrong concatenation dimension),
   and GQA head broadcasting.
4. **Clean verification**: The judge can verify correctness entirely through
   property-based testing on random inputs, requiring no external datasets,
   pre-trained models, or GPU hardware.

**Other environments I might build**:
- **Implement speculative decoding**: Verify that the output distribution
  matches standard autoregressive sampling while achieving wall-clock speedup.
- **Implement GPTQ-style quantization**: Quantize a pre-trained model and
  verify quality–compression tradeoff.
- **Implement a custom Triton kernel for fused attention**: Verify numerical
  equivalence and memory efficiency compared to naive attention.

---

## 7. GitHub Profile

GitHub: [mathemage](https://github.com/mathemage)

---

## 8. Availability

[To be filled by the applicant.]

---

## 9. Additional Information

The complete working environment — including the prompt, judge, reference
solution, and tests — is provided in this repository:

- **Prompt**: [`environment/prompt.md`](environment/prompt.md)
- **Judge**: [`environment/judge.py`](environment/judge.py)
- **VM setup**: [`environment/setup.sh`](environment/setup.sh)
- **Reference solution**: [`reference_solution/transformer_block.py`](reference_solution/transformer_block.py)
- **Tests**: [`tests/test_judge.py`](tests/test_judge.py)
