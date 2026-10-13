#!/bin/bash
# Step 2 on the GPU box: install ortho's env (mechtools over https; the pyproject's ssh URL needs a forwarded key), then score.
# Usage: bash probe_tests/run_score_probe.sh [LIMIT]
set -euo pipefail
cd "$(dirname "$0")/.."
export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=url.https://github.com/.insteadOf GIT_CONFIG_VALUE_0=ssh://git@github.com/
uv sync
PYTHONPATH=. uv run python probe_tests/score_probe.py "$@"
