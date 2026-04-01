# Preference Model Initial Assessment

[![Validation](https://github.com/mathemage/preference-model-initial-assessment/actions/workflows/validation.yml/badge.svg)](https://github.com/mathemage/preference-model-initial-assessment/actions/workflows/validation.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://choosealicense.com/licenses/mit/)

Solution for the [technical assessment](https://docs.google.com/forms/d/e/1FAIpQLSf4mqNST9kC6P41EDtaEYk65k0DptjuSyA_iRBoN10FwalSpw/viewform).

Automated validation runs on every push, pull request, and manual workflow run.
It executes the
RL-style judge against `reference_solution/transformer_block.py` and runs
`pytest tests/ -v` to confirm the environment and reference implementation work
end to end.

## Environment: Implement a Llama-Style Transformer Decoder Block from Scratch

An RL environment where the LLM agent must implement five core components of
modern transformer architectures — **RMSNorm**, **Rotary Position Embeddings
(RoPE)**, **Grouped-Query Attention (GQA)**, **SwiGLU**, and a complete
**LlamaBlock** — entirely from scratch in PyTorch.

### Repository Structure

```
├── assessment_answers.md              # Full assessment answers
├── environment/
│   ├── prompt.md                      # Exact prompt given to the LLM agent
│   ├── judge.py                       # Automated judge (continuous score 0–1)
│   └── setup.sh                       # VM setup script
├── reference_solution/
│   └── transformer_block.py           # Working reference implementation
├── tests/
│   └── test_judge.py                  # Tests for the judge + reference solution
└── requirements.txt                   # Python dependencies
```

### Quick Start

```bash
pip install -r requirements.txt

# Run the judge against the reference solution
python environment/judge.py --solution reference_solution/transformer_block.py

# Run tests
pytest tests/ -v
```

### Assessment Answers

See [assessment_answers.md](assessment_answers.md) for detailed answers to all
assessment questions.
