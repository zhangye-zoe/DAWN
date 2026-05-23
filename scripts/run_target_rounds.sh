#!/usr/bin/env bash
set -euo pipefail
CONFIG=${1:-configs/pipeline_tnbc.yaml}
python -m dawn.pipeline.run_target_rounds --config "$CONFIG"
