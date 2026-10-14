#!/usr/bin/env bash
# Run from anywhere: bash lab02/src/run_all.sh
set -e
cd "$(dirname "$0")"
mkdir -p ../results
python ../../lab01/src/print_versions.py > ../results/versions.txt
python check_forward_loss.py | tee ../results/forward_loss_check.txt
python grad_check.py
python train.py > ../results/train_log.txt          # init std 0.01 (as in manual)
python train.py 0.1 > ../results/train_log_std0.1.txt  # comparison run
python timing.py
