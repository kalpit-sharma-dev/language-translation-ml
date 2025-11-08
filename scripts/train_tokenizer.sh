#!/bin/bash

# Train tokenizer script

SRC_FILE=${1:-data/cleaned/src_cleaned.txt}
TGT_FILE=${2:-data/cleaned/tgt_cleaned.txt}
OUTPUT_DIR=${3:-data/tokenized}
VOCAB_SIZE=${4:-32000}
JOINT=${5:-true}

mkdir -p "$OUTPUT_DIR"

# Auto-adjust vocab size for small datasets
# Count lines in source file to estimate dataset size
if [ -f "$SRC_FILE" ]; then
    LINE_COUNT=$(wc -l < "$SRC_FILE" | tr -d ' ')
    if [ "$LINE_COUNT" -lt 10000 ]; then
        # For small datasets, use smaller vocab size
        AUTO_VOCAB=$((LINE_COUNT / 4))
        if [ "$AUTO_VOCAB" -lt 500 ]; then
            AUTO_VOCAB=500
        elif [ "$AUTO_VOCAB" -gt "$VOCAB_SIZE" ]; then
            AUTO_VOCAB=$VOCAB_SIZE
        fi
        echo "Dataset has $LINE_COUNT lines. Using vocab_size=$AUTO_VOCAB (auto-adjusted from $VOCAB_SIZE)"
        VOCAB_SIZE=$AUTO_VOCAB
    fi
fi

if [ "$JOINT" = "true" ]; then
    python -m src.tokenizer_train \
        --src_file "$SRC_FILE" \
        --tgt_file "$TGT_FILE" \
        --output_dir "$OUTPUT_DIR" \
        --vocab_size "$VOCAB_SIZE" \
        --joint
else
    python -m src.tokenizer_train \
        --src_file "$SRC_FILE" \
        --tgt_file "$TGT_FILE" \
        --output_dir "$OUTPUT_DIR" \
        --vocab_size "$VOCAB_SIZE"
fi

