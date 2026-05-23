#!/usr/bin/env bash
set -euo pipefail
CONFIG=${1:-configs/target_tnbc.yaml}
python -m dawn.training.target_train --config "$CONFIG"
