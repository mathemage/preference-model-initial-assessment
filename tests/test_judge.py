"""
Tests for the judge and the reference solution.

Running these tests validates that:
1. The judge correctly scores the reference solution as ~1.0 (all tests pass).
2. The judge correctly rejects bad / missing solutions.
3. Individual judge test suites work as expected.
"""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest
import torch

# ---------------------------------------------------------------------------
# Locate project root so we can import the judge and reference solution
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "environment"))
sys.path.insert(0, str(PROJECT_ROOT / "reference_solution"))

import judge  # noqa: E402
import transformer_block as ref  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def ref_solution_path():
    """Return the path to the reference solution."""
    return str(PROJECT_ROOT / "reference_solution" / "transformer_block.py")


@pytest.fixture()
def tmp_dir():
    """Provide a temporary directory, cleaned up after the test."""
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


# ---------------------------------------------------------------------------
# Judge integration tests
# ---------------------------------------------------------------------------

class TestJudgeWithReferenceSolution:
    """The judge should give a perfect (or near-perfect) score to the reference."""

    def test_perfect_score(self, ref_solution_path):
        report = judge.run_judge(ref_solution_path)
        assert report["score"] >= 0.95, (
            f"Reference solution scored {report['score']:.4f}, expected >= 0.95.\n"
            f"Errors: {report.get('errors')}\n"
            f"Suites: {json.dumps(report.get('suites'), indent=2, default=str)}"
        )

    def test_no_errors(self, ref_solution_path):
        report = judge.run_judge(ref_solution_path)
        assert report["errors"] == [], f"Unexpected errors: {report['errors']}"

    def test_all_suites_present(self, ref_solution_path):
        report = judge.run_judge(ref_solution_path)
        expected_suites = {"structure", "rmsnorm", "rope", "gqa", "swiglu", "llama_block"}
        assert set(report["suites"].keys()) == expected_suites


class TestJudgeRejectsBadSolutions:
    """The judge should give score 0 for missing / broken solutions."""

    def test_missing_file(self):
        report = judge.run_judge("/nonexistent/path.py")
        assert report["score"] == 0.0

    def test_empty_file(self, tmp_dir):
        path = os.path.join(tmp_dir, "empty.py")
        with open(path, "w") as f:
            f.write("")
        report = judge.run_judge(path)
        # An empty file has none of the required classes
        assert report["score"] < 0.2

    def test_disallowed_import(self, tmp_dir):
        path = os.path.join(tmp_dir, "bad_import.py")
        with open(path, "w") as f:
            f.write("import transformers\n")
        report = judge.run_judge(path)
        assert report["score"] == 0.0
        assert any("Disallowed" in e for e in report["errors"])


# ---------------------------------------------------------------------------
# Reference solution unit tests
# ---------------------------------------------------------------------------

class TestRMSNorm:
    def test_shape(self):
        norm = ref.RMSNorm(64)
        x = torch.randn(2, 8, 64)
        assert norm(x).shape == x.shape

    def test_constant_input(self):
        norm = ref.RMSNorm(32)
        c = torch.full((1, 1, 32), 2.0)
        y = norm(c)
        expected = c / (4.0 + 1e-6) ** 0.5
        assert torch.allclose(y, expected, atol=1e-5)


class TestRoPE:
    def test_cos_sin_shapes(self):
        rope = ref.RotaryPositionEmbedding(dim=32)
        cos, sin = rope(torch.zeros(1), seq_len=16)
        assert cos.shape == (16, 32)
        assert sin.shape == (16, 32)

    def test_identity(self):
        rope = ref.RotaryPositionEmbedding(dim=32)
        cos, sin = rope(torch.zeros(1), seq_len=8)
        identity = cos ** 2 + sin ** 2
        assert torch.allclose(identity, torch.ones_like(identity), atol=1e-5)

    def test_preserves_norm(self):
        rope = ref.RotaryPositionEmbedding(dim=16)
        cos, sin = rope(torch.zeros(1), seq_len=4)
        x = torch.randn(1, 2, 4, 16)
        x_rot = ref.apply_rotary_emb(x, cos, sin)
        assert torch.allclose(x.norm(dim=-1), x_rot.norm(dim=-1), atol=1e-4)


class TestGQA:
    def test_output_shape(self):
        gqa = ref.GroupedQueryAttention(64, num_heads=4, num_kv_heads=2)
        rope = ref.RotaryPositionEmbedding(16)
        cos, sin = rope(torch.zeros(1), seq_len=8)
        x = torch.randn(2, 8, 64)
        out, _ = gqa(x, cos, sin)
        assert out.shape == (2, 8, 64)

    def test_kv_cache_shapes(self):
        gqa = ref.GroupedQueryAttention(64, num_heads=4, num_kv_heads=2)
        rope = ref.RotaryPositionEmbedding(16)
        cos, sin = rope(torch.zeros(1), seq_len=1)
        x = torch.randn(1, 1, 64)
        _, cache = gqa(x, cos, sin)
        assert cache is not None
        assert cache[0].shape[2] == 1  # seq_len dimension


class TestSwiGLU:
    def test_shape(self):
        ffn = ref.SwiGLU(64, 256)
        x = torch.randn(2, 8, 64)
        assert ffn(x).shape == x.shape

    def test_hidden_dim_multiple_of_256(self):
        ffn = ref.SwiGLU(64, 256)
        assert ffn.w_gate.out_features % 256 == 0


class TestLlamaBlock:
    def test_shape(self):
        block = ref.LlamaBlock(64, 4, 2, 256)
        rope = ref.RotaryPositionEmbedding(16)
        cos, sin = rope(torch.zeros(1), seq_len=8)
        x = torch.randn(2, 8, 64)
        out, _ = block(x, cos, sin)
        assert out.shape == x.shape

    def test_gradient(self):
        block = ref.LlamaBlock(64, 4, 2, 256)
        rope = ref.RotaryPositionEmbedding(16)
        cos, sin = rope(torch.zeros(1), seq_len=4)
        x = torch.randn(1, 4, 64, requires_grad=True)
        block(x, cos, sin)[0].sum().backward()
        assert x.grad is not None
