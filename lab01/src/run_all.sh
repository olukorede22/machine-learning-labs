#!/usr/bin/env bash
# Run from repo root: bash lab01/src/run_all.sh
set -e
cd "$(dirname "$0")"
mkdir -p ../results
python print_versions.py > ../results/versions.txt
python baseline.py
python measure.py
python budget.py
