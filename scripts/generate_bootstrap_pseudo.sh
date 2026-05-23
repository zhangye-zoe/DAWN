#!/usr/bin/env bash
set -euo pipefail
CONFIG=${1:?"Usage: bash scripts/generate_bootstrap_pseudo.sh configs/bootstrap_pseudo_tnbc.yaml"}
python -m dawn.postprocess.bootstrap_pseudo --config "$CONFIG"
