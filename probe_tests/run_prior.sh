#!/bin/bash
# Prior-pressure experiments on the GPU box: token-forced scoring of natural rollouts, free generation under each system
# prompt (vLLM), then scoring of the generated rollouts. Each stage resumes from what is on disk.
# Usage: bash probe_tests/run_prior.sh [N_PER_TASK]
set -euo pipefail
cd "$(dirname "$0")/.."
export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=url.https://github.com/.insteadOf GIT_CONFIG_VALUE_0=ssh://git@github.com/
uv sync
echo "== stage 1: natural rollouts, token-forced under each system prompt"
PYTHONPATH=. uv run python probe_tests/prior_score.py natural
echo "== stage 2: free generation"
VENV=/workspace/venvs/vllm
if [ ! -x $VENV/bin/vllm ]; then
    uv venv $VENV --python 3.12
    uv pip install --python $VENV/bin/python --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu129 \
        "vllm @ https://github.com/vllm-project/vllm/releases/download/v0.26.0/vllm-0.26.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl"
fi
# as setup_vast.sh: the CUDA 12.9 index has no torchcodec for this torch; vllm only decodes video with it
$VENV/bin/python -c "import torchcodec.decoders" 2>/dev/null || uv pip install --python $VENV/bin/python --reinstall-package torchcodec --index-url https://download.pytorch.org/whl/cpu torchcodec==0.17.0
uv pip install --python $VENV/bin/python -q ninja
export PATH=$VENV/bin:$PATH  # vLLM shells out to ninja for kernel builds
$VENV/bin/python -c "import torch, vllm; assert torch.version.cuda.startswith('12.'), torch.version.cuda; torch.zeros(1).cuda()"
$VENV/bin/python probe_tests/prior_generate.py "${1:-256}"
echo "== stage 3: generated rollouts, scored under their own system prompt and under none"
PYTHONPATH=. uv run python probe_tests/prior_score.py gen
echo "== all done"
