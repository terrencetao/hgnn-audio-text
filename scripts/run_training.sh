#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-config/yemba.yaml}

python -m training.train --config "$CONFIG"
