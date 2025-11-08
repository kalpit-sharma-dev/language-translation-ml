#!/bin/bash

# Complete data preparation script

echo "Step 1: Downloading data..."
python scripts/download_data.py --use_sample --sample_size 1000 --split

echo "Step 2: Training tokenizer..."
bash scripts/train_tokenizer.sh \
    data/cleaned/train.en \
    data/cleaned/train.de \
    data/tokenized \
    32000 \
    true

echo "Data preparation complete!"
echo "Next step: Update configs/transformer_base.yaml and run training"

