#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export YOLO_CONFIG_DIR="$PWD/.ultralytics"
exec /home/zy0002-hlw/miniconda3/envs/yolo/bin/python camera_depth.py "$@"
