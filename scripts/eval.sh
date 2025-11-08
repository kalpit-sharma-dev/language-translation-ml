#!/bin/bash

# Evaluation script for NMT model

CHECKPOINT=${1:-checkpoints/best.pt}
CONFIG=${2:-}
OUTPUT=${3:-results/eval_results.txt}

mkdir -p results

python -m src.evaluate --checkpoint "$CHECKPOINT" --config "$CONFIG" --output "$OUTPUT"

