#!/usr/bin/env python3
"""
Script to train models for all supported languages: German, Hindi, Tamil, Telugu, Gujarati, Bengali
Usage: python scripts/train_all_languages.py [sample_size]
"""

import os
import sys
import subprocess
import yaml
import shutil
from pathlib import Path

# Languages to train
LANGUAGES = [
    ('de', 'German'),
    ('hi', 'Hindi'),
    ('ta', 'Tamil'),
    ('te', 'Telugu'),
    ('gu', 'Gujarati'),
    ('bn', 'Bengali'),
]

def run_command(cmd, check=True):
    """Run a shell command."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, check=check)
    return result.returncode == 0

def train_language(lang_code, lang_name, sample_size=1000):
    """Train a model for a specific language."""
    print(f"\n{'='*50}")
    print(f"Training {lang_name} ({lang_code}) model...")
    print(f"{'='*50}\n")
    
    # Step 1: Download/Create dataset
    if lang_code == 'de':
        print(f"Step 1: Downloading {lang_name} dataset...")
        run_command(f"python scripts/download_data.py --use_sample --sample_size {sample_size} --split")
        src_file = "data/cleaned/train.en"
        tgt_file = "data/cleaned/train.de"
        config_file = "configs/transformer_base.yaml"
    else:
        print(f"Step 1: Downloading {lang_name} dataset...")
        run_command(f"python scripts/download_indian_languages.py --target_lang {lang_code} --use_sample --sample_size {sample_size} --split")
        src_file = f"data/cleaned/train.en"
        tgt_file = f"data/cleaned/train.{lang_code}"
        
        # Create or use language-specific config
        config_file = f"configs/transformer_{lang_code}.yaml"
        if not os.path.exists(config_file):
            # Use multilingual template
            template_file = "configs/transformer_multilingual.yaml"
            if not os.path.exists(template_file):
                template_file = "configs/transformer_hindi.yaml"
            
            print(f"Creating config for {lang_name}...")
            with open(template_file, 'r') as f:
                config = yaml.safe_load(f)
            
            config['data']['train_tgt'] = f"data/cleaned/train.{lang_code}"
            config['data']['dev_tgt'] = f"data/cleaned/dev.{lang_code}"
            config['data']['test_tgt'] = f"data/cleaned/test.{lang_code}"
            config['data']['target_lang'] = lang_code
            
            with open(config_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
    
    # Step 2: Train tokenizer
    print(f"Step 2: Training tokenizer for {lang_name}...")
    tokenizer_dir = f"data/tokenized_{lang_code}"
    os.makedirs(tokenizer_dir, exist_ok=True)
    
    # Train tokenizer using Python module (cross-platform)
    run_command(f"python -m src.tokenizer_train --src_file {src_file} --tgt_file {tgt_file} --output_dir {tokenizer_dir} --vocab_size 32000 --joint")
    
    # Update config with correct tokenizer directory
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    config['data']['tokenizer_dir'] = tokenizer_dir
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Step 3: Train model
    print(f"Step 3: Training {lang_name} model...")
    run_command(f"python -m src.trainer --config {config_file}")
    
    # Step 4: Save with language-specific name
    print(f"Step 4: Saving {lang_name} model...")
    if os.path.exists("checkpoints/best.pt"):
        target_file = f"checkpoints/best_{lang_code}.pt"
        shutil.copy("checkpoints/best.pt", target_file)
        print(f"✓ {lang_name} model saved to {target_file}")
    else:
        print(f"✗ Warning: checkpoints/best.pt not found for {lang_name}")
    
    print(f"\n✓ {lang_name} ({lang_code}) training complete!\n")


def main():
    sample_size = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    
    print("="*50)
    print("Training Models for All Languages")
    print("="*50)
    print(f"\nLanguages: {', '.join([f'{name} ({code})' for code, name in LANGUAGES])}")
    print(f"Sample size: {sample_size} sentences per language")
    print()
    
    # Create checkpoints directory
    os.makedirs("checkpoints", exist_ok=True)
    
    # Train all languages
    for lang_code, lang_name in LANGUAGES:
        try:
            train_language(lang_code, lang_name, sample_size)
        except Exception as e:
            print(f"✗ Error training {lang_name}: {e}")
            print(f"Continuing with next language...\n")
            continue
    
    print("="*50)
    print("All Models Trained Successfully!")
    print("="*50)
    print("\nModels saved:")
    for file in Path("checkpoints").glob("best_*.pt"):
        size = file.stat().st_size / (1024 * 1024)  # Size in MB
        print(f"  - {file.name} ({size:.1f} MB)")
    
    print("\nTo start the web UI with all languages:")
    print("  python app.py")
    print("\nThe UI will automatically detect all models and show them in the dropdown!")


if __name__ == "__main__":
    main()

