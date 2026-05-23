#!/usr/bin/env bash
set -euo pipefail
CONFIG=${1:-configs/source_pannuke.yaml}
python -m dawn.training.source_train --config "$CONFIG"
