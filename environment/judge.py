#!/usr/bin/env python3
"""
Automated judge for the Llama-style Transformer Decoder Block RL environment.

Scoring
-------
The judge runs a series of test suites and returns a continuous score in [0, 1].
Each test suite has equal weight.  Within a suite every test case has equal
weight.  The final score is the weighted average of all test-case results.

Exit codes
----------
- 0  : judging completed (score printed to stdout as last line)
- 1  : solution file missing or could not be imported (score = 0)
"""

import ast
import importlib
import importlib.util
import json
import math
import sys
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SOLUTION_PATH = "/solution/transformer_block.py"
DISALLOWED_IMPORTS = {"transformers", "xformers", "flash_attn", "flash_attention"}
TOLERANCE = {"atol": 1e-4, "rtol": 1e-3}
TOLERANCE_FP16 = {"atol": 5e-3, "rtol": 5e-3}


def _load_solution(path: str = SOLUTION_PATH):
    """Dynamically load the solution module from *path*."""
    spec = importlib.util.spec_from_file_location("solution", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _check_disallowed_imports(path: str = SOLUTION_PATH) -> List[str]:
    """Static analysis: check that the solution does not import disallowed packages."""
    with open(path) as f:
        tree = ast.parse(f.read(), filename=path)
    violations: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in DISALLOWED_IMPORTS:
                    violations.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in DISALLOWED_IMPORTS:
                violations.append(node.module)
    return violations


# ---------------------------------------------------------------------------
# Individual test suites
# ---------------------------------------------------------------------------

TestResult = Dict[str, Any]  # {"pass": bool, "detail": str}


def suite_structure(mod) -> List[TestResult]:
    """Check that all required classes / functions exist with expected signatures."""
    results: List[TestResult] = []
    required_classes = ["RMSNorm", "RotaryPositionEmbedding", "GroupedQueryAttention", "SwiGLU", "LlamaBlock"]
    for name in required_classes:
        cls = getattr(mod, name, None)
        ok = cls is not None and isinstance(cls, type) and issubclass(cls, nn.Module)
        results.append({"pass": ok, "detail": f"class {name} exists and is nn.Module subclass"})

    fn = getattr(mod, "apply_rotary_emb", None)
    results.append({"pass": callable(fn), "detail": "apply_rotary_emb is callable"})
    return results


def suite_rmsnorm(mod) -> List[TestResult]:
    """Test RMSNorm correctness."""
    results: List[TestResult] = []
    try:
        RMSNorm = mod.RMSNorm
        norm = RMSNorm(64)

        # --- shape test ---
        x = torch.randn(2, 8, 64)
        y = norm(x)
        results.append({"pass": y.shape == x.shape, "detail": "output shape matches input shape"})

        # --- value test (constant input) ---
        c = torch.full((1, 1, 64), 3.0)
        y_c = norm(c)
        expected = c / math.sqrt(9.0 + 1e-6)
        results.append({
            "pass": torch.allclose(y_c, expected, **TOLERANCE),
            "detail": "correct output for constant input",
        })

        # --- gradient test ---
        x2 = torch.randn(2, 4, 64, requires_grad=True)
        norm(x2).sum().backward()
        results.append({"pass": x2.grad is not None and x2.grad.shape == x2.shape, "detail": "gradient flows"})
    except Exception as exc:
        results.append({"pass": False, "detail": f"exception: {exc}"})
    return results


def suite_rope(mod) -> List[TestResult]:
    """Test Rotary Position Embeddings."""
    results: List[TestResult] = []
    try:
        RoPE = mod.RotaryPositionEmbedding
        apply_fn = mod.apply_rotary_emb
        rope = RoPE(dim=32, max_seq_len=128)

        dummy = torch.zeros(1)  # just for device inference
        cos, sin = rope(dummy, seq_len=16)

        # --- shape ---
        results.append({"pass": cos.shape == (16, 32) and sin.shape == (16, 32), "detail": "cos/sin shape correct"})

        # --- cos^2 + sin^2 == 1 ---
        identity = cos ** 2 + sin ** 2
        results.append({
            "pass": torch.allclose(identity, torch.ones_like(identity), atol=1e-5),
            "detail": "cos^2 + sin^2 == 1",
        })

        # --- apply_rotary_emb preserves norm ---
        x = torch.randn(1, 4, 16, 32)
        x_rot = apply_fn(x, cos, sin)
        norms_before = x.norm(dim=-1)
        norms_after = x_rot.norm(dim=-1)
        results.append({
            "pass": torch.allclose(norms_before, norms_after, atol=1e-4),
            "detail": "RoPE preserves vector norms",
        })

        # --- different positions produce different embeddings ---
        cos2, sin2 = rope(dummy, seq_len=32)
        results.append({
            "pass": not torch.allclose(cos2[0], cos2[1], atol=1e-6),
            "detail": "different positions have different embeddings",
        })

        # --- gradient through apply_rotary_emb ---
        xg = torch.randn(1, 4, 16, 32, requires_grad=True)
        apply_fn(xg, cos, sin).sum().backward()
        results.append({"pass": xg.grad is not None, "detail": "gradient flows through apply_rotary_emb"})
    except Exception as exc:
        results.append({"pass": False, "detail": f"exception: {exc}"})
    return results


def suite_gqa(mod) -> List[TestResult]:
    """Test Grouped-Query Attention."""
    results: List[TestResult] = []
    try:
        GQA = mod.GroupedQueryAttention
        RoPE = mod.RotaryPositionEmbedding
        dim = 64
        seq = 8
        batch = 2

        # --- MHA equivalence (num_kv_heads == num_heads) ---
        num_heads_mha = 8
        head_dim_mha = dim // num_heads_mha
        gqa_mha = GQA(dim, num_heads=num_heads_mha, num_kv_heads=num_heads_mha)
        rope_mha = RoPE(head_dim_mha, max_seq_len=64)
        x = torch.randn(batch, seq, dim)
        dummy = torch.zeros(1)
        cos, sin = rope_mha(dummy, seq_len=seq)
        out, _ = gqa_mha(x, cos, sin)
        results.append({"pass": out.shape == (batch, seq, dim), "detail": "MHA output shape correct"})

        # --- MQA equivalence (num_kv_heads == 1) ---
        gqa_mqa = GQA(dim, num_heads=num_heads_mha, num_kv_heads=1)
        out_mqa, _ = gqa_mqa(x, cos, sin)
        results.append({"pass": out_mqa.shape == (batch, seq, dim), "detail": "MQA output shape correct"})

        # --- Causal masking: output at position i should not depend on positions > i ---
        num_heads_c = 4
        num_kv_heads_c = 2
        head_dim_c = dim // num_heads_c
        torch.manual_seed(42)
        gqa_causal = GQA(dim, num_heads=num_heads_c, num_kv_heads=num_kv_heads_c)
        rope_c = RoPE(head_dim_c, max_seq_len=64)
        x1 = torch.randn(1, seq, dim)
        cos_c, sin_c = rope_c(dummy, seq_len=seq)
        out_full, _ = gqa_causal(x1, cos_c, sin_c)
        # Truncate input and compare prefix output
        x1_prefix = x1[:, :4, :]
        cos_p, sin_p = rope_c(dummy, seq_len=4)
        out_prefix, _ = gqa_causal(x1_prefix, cos_p, sin_p)
        results.append({
            "pass": torch.allclose(out_full[:, :4, :], out_prefix, **TOLERANCE),
            "detail": "causal masking: prefix outputs match truncated input",
        })

        # --- KV-cache: step-by-step decoding matches full-sequence output ---
        torch.manual_seed(123)
        gqa_kv = GQA(dim, num_heads=num_heads_c, num_kv_heads=num_kv_heads_c)
        x_full = torch.randn(1, 6, dim)
        cos_full, sin_full = rope_c(dummy, seq_len=6)
        out_full_kv, _ = gqa_kv(x_full, cos_full, sin_full)

        # Step-by-step with kv-cache
        cache = None
        outs = []
        for t in range(6):
            x_t = x_full[:, t : t + 1, :]
            cos_t = cos_full[t : t + 1]
            sin_t = sin_full[t : t + 1]
            out_t, cache = gqa_kv(x_t, cos_t, sin_t, kv_cache=cache)
            outs.append(out_t)
        out_step = torch.cat(outs, dim=1)
        results.append({
            "pass": torch.allclose(out_full_kv, out_step, **TOLERANCE),
            "detail": "KV-cache step-by-step matches full-sequence output",
        })

        # --- gradient ---
        xg = torch.randn(batch, seq, dim, requires_grad=True)
        gqa_mha(xg, cos, sin)[0].sum().backward()
        results.append({"pass": xg.grad is not None, "detail": "gradient flows through GQA"})
    except Exception as exc:
        results.append({"pass": False, "detail": f"exception: {exc}"})
    return results


def suite_swiglu(mod) -> List[TestResult]:
    """Test SwiGLU FFN."""
    results: List[TestResult] = []
    try:
        SwiGLU = mod.SwiGLU
        ffn = SwiGLU(dim=64, hidden_dim=256)

        # --- shape ---
        x = torch.randn(2, 8, 64)
        y = ffn(x)
        results.append({"pass": y.shape == x.shape, "detail": "output shape matches input shape"})

        # --- effective hidden dim is multiple of 256 ---
        for child in ffn.children():
            if isinstance(child, nn.Linear) and child.in_features == 64:
                results.append({
                    "pass": child.out_features % 256 == 0,
                    "detail": "effective hidden dim is multiple of 256",
                })
                break
        else:
            # Try checking weight shapes directly
            hidden_found = False
            for name, param in ffn.named_parameters():
                if "gate" in name or "up" in name:
                    if param.shape[0] % 256 == 0:
                        hidden_found = True
                        break
            results.append({"pass": hidden_found, "detail": "effective hidden dim is multiple of 256"})

        # --- SwiGLU formula check: manually compute for known input ---
        torch.manual_seed(0)
        ffn2 = SwiGLU(dim=16, hidden_dim=64)
        x2 = torch.randn(1, 1, 16)
        y2 = ffn2(x2)
        # Manually compute
        gate = F.silu(F.linear(x2, ffn2.w_gate.weight))
        up = F.linear(x2, ffn2.w_up.weight)
        expected = F.linear(gate * up, ffn2.w_down.weight)
        results.append({
            "pass": torch.allclose(y2, expected, **TOLERANCE),
            "detail": "SwiGLU formula is correct (SiLU(gate) * up then down)",
        })

        # --- gradient ---
        xg = torch.randn(2, 4, 64, requires_grad=True)
        ffn(xg).sum().backward()
        results.append({"pass": xg.grad is not None, "detail": "gradient flows through SwiGLU"})
    except Exception as exc:
        results.append({"pass": False, "detail": f"exception: {exc}"})
    return results


def suite_llama_block(mod) -> List[TestResult]:
    """Test the complete LlamaBlock."""
    results: List[TestResult] = []
    try:
        LlamaBlock = mod.LlamaBlock
        RoPE = mod.RotaryPositionEmbedding
        dim = 64
        num_heads = 4
        num_kv_heads = 2
        hidden_dim = 256
        head_dim = dim // num_heads

        block = LlamaBlock(dim, num_heads, num_kv_heads, hidden_dim)
        rope = RoPE(head_dim, max_seq_len=128)
        dummy = torch.zeros(1)

        # --- basic forward ---
        x = torch.randn(2, 8, dim)
        cos, sin = rope(dummy, seq_len=8)
        out, kv = block(x, cos, sin)
        results.append({"pass": out.shape == x.shape, "detail": "LlamaBlock output shape correct"})

        # --- residual: output != 0 when input != 0 ---
        results.append({
            "pass": bool(out.abs().sum() > 0),
            "detail": "LlamaBlock produces non-zero output",
        })

        # --- KV-cache step-by-step vs full ---
        torch.manual_seed(999)
        block2 = LlamaBlock(dim, num_heads, num_kv_heads, hidden_dim)
        x_full = torch.randn(1, 6, dim)
        cos_full, sin_full = rope(dummy, seq_len=6)
        out_full, _ = block2(x_full, cos_full, sin_full)

        cache = None
        outs = []
        for t in range(6):
            x_t = x_full[:, t : t + 1, :]
            cos_t = cos_full[t : t + 1]
            sin_t = sin_full[t : t + 1]
            out_t, cache = block2(x_t, cos_t, sin_t, kv_cache=cache)
            outs.append(out_t)
        out_step = torch.cat(outs, dim=1)
        results.append({
            "pass": torch.allclose(out_full, out_step, **TOLERANCE),
            "detail": "LlamaBlock KV-cache matches full-sequence output",
        })

        # --- float16 ---
        block_fp16 = LlamaBlock(dim, num_heads, num_kv_heads, hidden_dim).half()
        x_fp16 = torch.randn(1, 4, dim, dtype=torch.float16)
        cos16, sin16 = rope(x_fp16, seq_len=4)
        cos16, sin16 = cos16.half(), sin16.half()
        out16, _ = block_fp16(x_fp16, cos16, sin16)
        results.append({"pass": out16.dtype == torch.float16, "detail": "LlamaBlock works in float16"})

        # --- gradient ---
        xg = torch.randn(2, 4, dim, requires_grad=True)
        cos_g, sin_g = rope(dummy, seq_len=4)
        block(xg, cos_g, sin_g)[0].sum().backward()
        results.append({"pass": xg.grad is not None, "detail": "gradient flows through LlamaBlock"})

        # --- all parameters receive gradients ---
        block3 = LlamaBlock(dim, num_heads, num_kv_heads, hidden_dim)
        xb = torch.randn(1, 4, dim)
        cos_b, sin_b = rope(dummy, seq_len=4)
        block3(xb, cos_b, sin_b)[0].sum().backward()
        all_grads = all(p.grad is not None for p in block3.parameters())
        results.append({"pass": all_grads, "detail": "all LlamaBlock parameters receive gradients"})
    except Exception as exc:
        results.append({"pass": False, "detail": f"exception: {exc}"})
    return results


# ---------------------------------------------------------------------------
# Main judge logic
# ---------------------------------------------------------------------------

SUITES: List[Tuple[str, Callable]] = [
    ("structure", suite_structure),
    ("rmsnorm", suite_rmsnorm),
    ("rope", suite_rope),
    ("gqa", suite_gqa),
    ("swiglu", suite_swiglu),
    ("llama_block", suite_llama_block),
]


def run_judge(solution_path: str = SOLUTION_PATH) -> dict:
    """Run all suites and return a report dict with per-suite and overall scores."""
    import os

    report: Dict[str, Any] = {"suites": {}, "score": 0.0, "errors": []}

    # --- File existence ---
    if not os.path.isfile(solution_path):
        report["errors"].append(f"Solution file not found at {solution_path}")
        return report

    # --- Disallowed imports ---
    violations = _check_disallowed_imports(solution_path)
    if violations:
        report["errors"].append(f"Disallowed imports found: {violations}")
        return report

    # --- Load module ---
    try:
        mod = _load_solution(solution_path)
    except Exception as exc:
        report["errors"].append(f"Failed to import solution: {exc}")
        return report

    # --- Run suites ---
    suite_scores: List[float] = []
    for suite_name, suite_fn in SUITES:
        try:
            results = suite_fn(mod)
        except Exception as exc:
            results = [{"pass": False, "detail": f"suite crashed: {exc}"}]
        passed = sum(1 for r in results if r["pass"])
        total = len(results)
        score = passed / total if total > 0 else 0.0
        suite_scores.append(score)
        report["suites"][suite_name] = {
            "score": score,
            "passed": passed,
            "total": total,
            "details": results,
        }

    report["score"] = sum(suite_scores) / len(suite_scores) if suite_scores else 0.0
    return report


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Judge for Llama Transformer Block environment")
    parser.add_argument("--solution", default=SOLUTION_PATH, help="Path to solution file")
    parser.add_argument("--json", action="store_true", help="Output full JSON report")
    args = parser.parse_args()

    report = run_judge(args.solution)

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        for name, suite in report.get("suites", {}).items():
            status = f"{suite['passed']}/{suite['total']}"
            print(f"  [{status}] {name}")
            for d in suite["details"]:
                mark = "✓" if d["pass"] else "✗"
                print(f"        {mark} {d['detail']}")
        if report.get("errors"):
            for e in report["errors"]:
                print(f"  ERROR: {e}")
        print(f"\nFinal score: {report['score']:.4f}")

    # Exit code 0 always – the score is in stdout
    sys.exit(0)


if __name__ == "__main__":
    main()
