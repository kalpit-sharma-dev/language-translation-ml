#!/bin/bash
# Script to train models for all supported languages: German, Hindi, Tamil, Telugu, Gujarati, Bengali

set -e  # Exit on error

echo "=========================================="
echo "Training Models for All Languages"
echo "=========================================="
echo ""
echo "This script will train models for:"
echo "  - German (de)"
echo "  - Hindi (hi)"
echo "  - Tamil (ta)"
echo "  - Telugu (te)"
echo "  - Gujarati (gu)"
echo "  - Bengali (bn)"
echo ""
echo "Each model will be saved as: checkpoints/best_<lang>.pt"
echo ""

SAMPLE_SIZE=${1:-1000}  # Default to 1000 if not provided
echo "Using sample size: $SAMPLE_SIZE sentences per language"
echo ""

# Create checkpoints directory if it doesn't exist
mkdir -p checkpoints

# Function to train a language model
train_language() {
    local lang_code=$1
    local lang_name=$2
    local sample_size=$3
    
    echo "=========================================="
    echo "Training $lang_name ($lang_code) model..."
    echo "=========================================="
    
    # Step 1: Download/Create dataset
    if [ "$lang_code" == "de" ]; then
        echo "Step 1: Downloading German dataset..."
        python scripts/download_data.py --use_sample --sample_size $sample_size --split
        SRC_FILE="data/cleaned/train.en"
        TGT_FILE="data/cleaned/train.de"
        CONFIG_FILE="configs/transformer_base.yaml"
    else
        echo "Step 1: Downloading $lang_name dataset..."
        python scripts/download_indian_languages.py \
            --target_lang $lang_code \
            --use_sample \
            --sample_size $sample_size \
            --split
        SRC_FILE="data/cleaned/train.en"
        TGT_FILE="data/cleaned/train.$lang_code"
        
        # Check if language-specific config exists, otherwise use multilingual template
        if [ -f "configs/transformer_$lang_code.yaml" ]; then
            CONFIG_FILE="configs/transformer_$lang_code.yaml"
        else
            CONFIG_FILE="configs/transformer_multilingual.yaml"
            # Update config for this language
            echo "Creating config for $lang_name..."
            python -c "
import yaml
with open('configs/transformer_multilingual.yaml', 'r') as f:
    config = yaml.safe_load(f)
config['data']['train_tgt'] = 'data/cleaned/train.$lang_code'
config['data']['dev_tgt'] = 'data/cleaned/dev.$lang_code'
config['data']['test_tgt'] = 'data/cleaned/test.$lang_code'
config['data']['target_lang'] = '$lang_code'
with open('configs/transformer_$lang_code.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
"
            CONFIG_FILE="configs/transformer_$lang_code.yaml"
        fi
    fi
    
    # Step 2: Train tokenizer
    echo "Step 2: Training tokenizer for $lang_name..."
    TOKENIZER_DIR="data/tokenized_$lang_code"
    mkdir -p $TOKENIZER_DIR
    
    bash scripts/train_tokenizer.sh \
        $SRC_FILE \
        $TGT_FILE \
        $TOKENIZER_DIR \
        32000 \
        true
    
    # Update config with correct tokenizer directory
    python -c "
import yaml
with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)
config['data']['tokenizer_dir'] = '$TOKENIZER_DIR'
with open('$CONFIG_FILE', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
"
    
    # Step 3: Train model
    echo "Step 3: Training $lang_name model..."
    python -m src.trainer --config $CONFIG_FILE
    
    # Step 4: Save with language-specific name
    echo "Step 4: Saving $lang_name model..."
    if [ -f "checkpoints/best.pt" ]; then
        cp checkpoints/best.pt "checkpoints/best_$lang_code.pt"
        echo "✓ $lang_name model saved to checkpoints/best_$lang_code.pt"
    else
        echo "✗ Warning: checkpoints/best.pt not found for $lang_name"
    fi
    
    echo ""
    echo "✓ $lang_name ($lang_code) training complete!"
    echo ""
}

# Train all languages
train_language "de" "German" $SAMPLE_SIZE
train_language "hi" "Hindi" $SAMPLE_SIZE
train_language "ta" "Tamil" $SAMPLE_SIZE
train_language "te" "Telugu" $SAMPLE_SIZE
train_language "gu" "Gujarati" $SAMPLE_SIZE
train_language "bn" "Bengali" $SAMPLE_SIZE

echo "=========================================="
echo "All Models Trained Successfully!"
echo "=========================================="
echo ""
echo "Models saved:"
ls -lh checkpoints/best_*.pt 2>/dev/null || echo "No models found"
echo ""
echo "To start the web UI with all languages:"
echo "  python app.py"
echo ""
echo "The UI will automatically detect all models and show them in the dropdown!"

