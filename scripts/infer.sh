#!/usr/bin/env bash
set -euo pipefail
CONFIG=${1:-configs/infer_tnbc.yaml}
python -m dawn.postprocess.infer --config "$CONFIG"
