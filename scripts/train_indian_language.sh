#!/bin/bash

# Complete training script for Indian languages

TARGET_LANG=${1:-hi}
SAMPLE_SIZE=${2:-1000}
VOCAB_SIZE=${3:-32000}

echo "Training English → ${TARGET_LANG^^} Translation Model"
echo "=================================================="

# Step 1: Download data
echo "Step 1: Downloading ${TARGET_LANG^^} dataset..."
python scripts/download_indian_languages.py \
    --target_lang "$TARGET_LANG" \
    --use_sample \
    --sample_size "$SAMPLE_SIZE" \
    --split

# Step 2: Train tokenizer
echo ""
echo "Step 2: Training tokenizer..."
bash scripts/train_tokenizer.sh \
    "data/cleaned/train.en" \
    "data/cleaned/train.${TARGET_LANG}" \
    "data/tokenized" \
    "$VOCAB_SIZE" \
    true

# Step 3: Update config (create language-specific config)
echo ""
echo "Step 3: Creating config file..."
CONFIG_FILE="configs/transformer_${TARGET_LANG}.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    # Create config from template
    sed "s/\.de/\.${TARGET_LANG}/g; s/target_lang: de/target_lang: ${TARGET_LANG}/g" \
        configs/transformer_base.yaml > "$CONFIG_FILE"
    echo "Created config: $CONFIG_FILE"
fi

# Step 4: Train model
echo ""
echo "Step 4: Training model..."
python -m src.trainer --config "$CONFIG_FILE"

echo ""
echo "Training complete! Model saved to checkpoints/best.pt"
echo ""
echo "To translate, run:"
echo "python -m src.infer --checkpoint checkpoints/best.pt --input 'Your text here' --beam_size 4"

