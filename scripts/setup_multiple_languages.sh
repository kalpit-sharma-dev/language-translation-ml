#!/bin/bash
# Script to help set up multiple language models for the web UI

echo "=== Multi-Language Setup Guide ==="
echo ""
echo "This script helps you set up multiple language models for the web UI."
echo ""

# Check if models exist
echo "Checking for existing models..."
echo ""

LANGUAGES=("de" "hi" "bn" "te" "ta" "gu")
LANG_NAMES=("German" "Hindi" "Bengali" "Telugu" "Tamil" "Gujarati")

for i in "${!LANGUAGES[@]}"; do
    lang=${LANGUAGES[$i]}
    name=${LANG_NAMES[$i]}
    
    # Check multiple possible locations
    found=false
    if [ -f "checkpoints/best_${lang}.pt" ]; then
        echo "✓ ${name} (${lang}): Found at checkpoints/best_${lang}.pt"
        found=true
    elif [ -f "checkpoints/${lang}/best.pt" ]; then
        echo "✓ ${name} (${lang}): Found at checkpoints/${lang}/best.pt"
        found=true
    elif [ -f "checkpoints/best.pt" ] && [ "$lang" == "de" ]; then
        echo "✓ ${name} (${lang}): Found at checkpoints/best.pt (default)"
        found=true
    else
        echo "✗ ${name} (${lang}): Not found"
    fi
done

echo ""
echo "=== Setup Instructions ==="
echo ""
echo "To use multiple languages in the web UI:"
echo ""
echo "1. Train models for each language:"
echo "   For German:"
echo "     python scripts/download_data.py --use_sample --sample_size 1000 --split"
echo "     bash scripts/train_tokenizer.sh data/cleaned/train.en data/cleaned/train.de data/tokenized 32000 true"
echo "     python -m src.trainer --config configs/transformer_base.yaml"
echo "     cp checkpoints/best.pt checkpoints/best_de.pt"
echo ""
echo "   For Hindi:"
echo "     bash scripts/train_indian_language.sh hi 1000"
echo "     cp checkpoints/best.pt checkpoints/best_hi.pt"
echo ""
echo "   For other languages, replace 'hi' with language code (bn, te, ta, gu, etc.)"
echo ""
echo "2. Start web UI with environment variables:"
echo "   export CHECKPOINT_PATH_DE=checkpoints/best_de.pt"
echo "   export CONFIG_PATH_DE=configs/transformer_base.yaml"
echo "   export CHECKPOINT_PATH_HI=checkpoints/best_hi.pt"
echo "   export CONFIG_PATH_HI=configs/transformer_hindi.yaml"
echo "   python app.py"
echo ""
echo "3. Or the UI will auto-detect models in these locations:"
echo "   - checkpoints/best_<lang>.pt (e.g., checkpoints/best_hi.pt)"
echo "   - checkpoints/<lang>/best.pt (e.g., checkpoints/hi/best.pt)"
echo "   - checkpoints/best.pt (default, used for German)"
echo ""
echo "See quickstart.md for complete instructions!"

