#!/usr/bin/env bash
set -euo pipefail
python -m drone_pur.evaluate --mode open "$@"
