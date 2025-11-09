#!/usr/bin/env python3
"""Create config files for all languages."""

import yaml
import os

LANGUAGES = [
    ('de', 'German', 'transformer_base.yaml'),
    ('hi', 'Hindi', 'transformer_hindi.yaml'),
    ('ta', 'Tamil', 'transformer_multilingual.yaml'),
    ('te', 'Telugu', 'transformer_multilingual.yaml'),
    ('gu', 'Gujarati', 'transformer_multilingual.yaml'),
    ('bn', 'Bengali', 'transformer_multilingual.yaml'),
]

def create_config(lang_code, lang_name, template_file):
    """Create a config file for a language."""
    config_file = f"configs/transformer_{lang_code}.yaml"
    
    # Read template
    with open(template_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update for this language
    config['data']['train_tgt'] = f"data/cleaned/train.{lang_code}"
    config['data']['dev_tgt'] = f"data/cleaned/dev.{lang_code}"
    config['data']['test_tgt'] = f"data/cleaned/test.{lang_code}"
    config['data']['target_lang'] = lang_code
    config['data']['tokenizer_dir'] = f"data/tokenized_{lang_code}"
    
    # Write config
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Created: {config_file}")

if __name__ == "__main__":
    os.makedirs("configs", exist_ok=True)
    
    for lang_code, lang_name, template in LANGUAGES:
        if os.path.exists(template):
            create_config(lang_code, lang_name, template)
        else:
            print(f"Warning: Template {template} not found, skipping {lang_name}")

