#!/bin/bash

# Training script for NMT model

CONFIG=${1:-configs/transformer_base.yaml}
RESUME=${2:-}

if [ -z "$RESUME" ]; then
    python -m src.trainer --config "$CONFIG"
else
    python -m src.trainer --config "$CONFIG" --resume "$RESUME"
fi

