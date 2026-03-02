#!/usr/bin/env bash
# Setup script for the RL environment VM.
# Installs all dependencies needed by both the LLM agent and the judge.
set -euo pipefail

echo "=== Setting up RL environment for Llama Transformer Block task ==="

# Install Python dependencies
pip install torch>=2.0.0

# Create the solution directory where the LLM agent will write its code
mkdir -p /solution

echo "=== Environment setup complete ==="
echo "The LLM agent should create /solution/transformer_block.py"
