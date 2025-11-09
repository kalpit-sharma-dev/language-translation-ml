"""
Flask web application for Neural Machine Translation.
"""
import os
import torch
import sentencepiece as spm
import yaml
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from src.model import SimpleTransformerNMT
from src.infer import translate_sentence

app = Flask(__name__)
CORS(app)

# Global variables for models and tokenizers
# Store multiple models: {language_code: {'model': model, 'src_tokenizer': ..., 'tgt_tokenizer': ..., 'config': ...}}
models = {}
current_language = None
device = None

# Language configurations - map language codes to checkpoint/config paths
# Each language uses its own checkpoint and config file
LANGUAGE_CONFIGS = {
    'de': {
        'checkpoint': os.getenv('CHECKPOINT_PATH_DE', 'checkpoints/best_de.pt'),
        'config': os.getenv('CONFIG_PATH_DE', 'configs/transformer_base.yaml'),
        'name': 'German',
        'flag': '🇩🇪'
    },
    'hi': {
        'checkpoint': os.getenv('CHECKPOINT_PATH_HI', 'checkpoints/best_hi.pt'),
        'config': os.getenv('CONFIG_PATH_HI', 'configs/transformer_hindi.yaml'),
        'name': 'Hindi',
        'flag': '🇮🇳'
    },
    'bn': {
        'checkpoint': os.getenv('CHECKPOINT_PATH_BN', 'checkpoints/best_bn.pt'),
        'config': os.getenv('CONFIG_PATH_BN', 'configs/transformer_bn.yaml'),
        'name': 'Bengali',
        'flag': '🇧🇩'
    },
    'te': {
        'checkpoint': os.getenv('CHECKPOINT_PATH_TE', 'checkpoints/best_te.pt'),
        'config': os.getenv('CONFIG_PATH_TE', 'configs/transformer_te.yaml'),
        'name': 'Telugu',
        'flag': '🇮🇳'
    },
    'ta': {
        'checkpoint': os.getenv('CHECKPOINT_PATH_TA', 'checkpoints/best_ta.pt'),
        'config': os.getenv('CONFIG_PATH_TA', 'configs/transformer_ta.yaml'),
        'name': 'Tamil',
        'flag': '🇮🇳'
    },
    'gu': {
        'checkpoint': os.getenv('CHECKPOINT_PATH_GU', 'checkpoints/best_gu.pt'),
        'config': os.getenv('CONFIG_PATH_GU', 'configs/transformer_gu.yaml'),
        'name': 'Gujarati',
        'flag': '🇮🇳'
    }
}


def find_checkpoint_for_language(language_code: str):
    """Find checkpoint file for a language, checking multiple possible locations."""
    # Check environment variable first
    env_checkpoint = os.getenv(f'CHECKPOINT_PATH_{language_code.upper()}')
    if env_checkpoint and os.path.exists(env_checkpoint):
        print(f"  Found checkpoint from environment: {env_checkpoint}")
        return env_checkpoint
    
    # Check language config
    if language_code in LANGUAGE_CONFIGS:
        config_checkpoint = LANGUAGE_CONFIGS[language_code]['checkpoint']
        if config_checkpoint and os.path.exists(config_checkpoint):
            print(f"  Found checkpoint from config: {config_checkpoint}")
            return config_checkpoint
    
    # Check common checkpoint locations (prioritize language-specific)
    possible_paths = [
        f'checkpoints/best_{language_code}.pt',  # Language-specific (preferred)
        f'checkpoints/{language_code}/best.pt',   # Language-specific subdirectory
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"  Found checkpoint at: {path}")
            return path
    
    # Only use generic best.pt as last resort for German (de) if no language-specific exists
    # But first verify it's actually for German by checking its config
    if language_code == 'de' and os.path.exists('checkpoints/best.pt'):
        # Verify it's actually for German before using it
        try:
            checkpoint = torch.load('checkpoints/best.pt', map_location='cpu')
            saved_config = checkpoint.get('config', {})
            saved_target_lang = saved_config.get('data', {}).get('target_lang', 'de')
            if saved_target_lang == 'de':
                print(f"  Using default checkpoint for German: checkpoints/best.pt")
                return 'checkpoints/best.pt'
            else:
                print(f"  WARNING: checkpoints/best.pt is for {saved_target_lang}, not German. Skipping.")
        except Exception as e:
            print(f"  Could not verify checkpoints/best.pt: {e}, skipping")
    
    print(f"  No checkpoint found for {language_code}")
    return None


def find_config_for_language(language_code: str):
    """Find config file for a language."""
    # Check environment variable first
    env_config = os.getenv(f'CONFIG_PATH_{language_code.upper()}')
    if env_config and os.path.exists(env_config):
        return env_config
    
    # Check language config
    if language_code in LANGUAGE_CONFIGS:
        config_path = LANGUAGE_CONFIGS[language_code]['config']
        if config_path and os.path.exists(config_path):
            return config_path
    
    # Check common config locations
    possible_paths = [
        f'configs/transformer_{language_code}.yaml',
        f'configs/transformer_multilingual.yaml',
        f'configs/transformer_base.yaml',  # Default fallback
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None


def load_model(language_code: str, checkpoint_path: str = None, config_path: str = None):
    """Load the trained model and tokenizers for a specific language."""
    global models, device, current_language
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
    
    # Get checkpoint and config from language config if not provided
    if language_code not in LANGUAGE_CONFIGS:
        raise ValueError(f"Language {language_code} not configured")
    
    lang_config = LANGUAGE_CONFIGS[language_code]
    
    # Find checkpoint - prioritize provided path, then language config, then auto-detect
    if checkpoint_path is None:
        # First try language-specific checkpoint from config
        if lang_config['checkpoint'] and os.path.exists(lang_config['checkpoint']):
            checkpoint_path = lang_config['checkpoint']
            print(f"Using checkpoint from language config: {checkpoint_path}")
        else:
            # Auto-detect checkpoint
            checkpoint_path = find_checkpoint_for_language(language_code)
    
    # Find config - prioritize provided path, then language config, then auto-detect
    if config_path is None:
        # First try language-specific config from config
        if lang_config['config'] and os.path.exists(lang_config['config']):
            config_path = lang_config['config']
            print(f"Using config from language config: {config_path}")
        else:
            # Auto-detect config
            config_path = find_config_for_language(language_code)
    
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        # Provide helpful error message
        lang_name = LANGUAGE_CONFIGS.get(language_code, {}).get('name', language_code)
        checked_paths = [
            f'checkpoints/best_{language_code}.pt',
            f'checkpoints/{language_code}/best.pt',
            f'checkpoints/best.pt',
        ]
        error_msg = (
            f"Model checkpoint not found for {lang_name} ({language_code}).\n"
            f"Please either:\n"
            f"1. Train a model for {lang_name} first\n"
            f"2. Set CHECKPOINT_PATH_{language_code.upper()} environment variable\n"
            f"3. Place checkpoint at: checkpoints/best_{language_code}.pt\n"
            f"Checked paths: {', '.join(checked_paths)}"
        )
        raise FileNotFoundError(error_msg)
    
    print(f"Loading model for {language_code} ({LANGUAGE_CONFIGS[language_code]['name']})...")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Config: {config_path}")
    
    # Load checkpoint first to get saved config and vocab_size
    checkpoint = torch.load(checkpoint_path, map_location=device)
    saved_config = checkpoint.get('config')
    
    # Verify this checkpoint is for the correct language
    if saved_config and 'data' in saved_config:
        saved_target_lang = saved_config['data'].get('target_lang')
        if saved_target_lang and saved_target_lang != language_code:
            print(f"  ERROR: Checkpoint target_lang ({saved_target_lang}) != requested language ({language_code})")
            print(f"  This checkpoint is for {saved_target_lang}, not {language_code}!")
            raise ValueError(
                f"Checkpoint mismatch! {checkpoint_path} is for {saved_target_lang}, "
                f"but you're trying to load it as {language_code}. "
                f"Expected checkpoint: checkpoints/best_{language_code}.pt"
            )
    
    # Load config file - file config takes precedence, but verify target_lang matches
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            file_config = yaml.safe_load(f)
        
        # Verify config target_lang matches requested language
        config_target_lang = file_config.get('data', {}).get('target_lang')
        if config_target_lang and config_target_lang != language_code:
            print(f"  ERROR: Config target_lang ({config_target_lang}) != requested language ({language_code})")
            raise ValueError(
                f"Config mismatch! {config_path} is for {config_target_lang}, "
                f"but you're trying to load it as {language_code}. "
                f"Expected config: configs/transformer_{language_code}.yaml"
            )
        
        # Use file config, but prefer tokenizer_dir from saved config if available (more accurate)
        config = file_config
        if saved_config and 'data' in saved_config:
            # Update tokenizer_dir from saved config if it exists (checkpoint knows the actual tokenizer used)
            if 'tokenizer_dir' in saved_config.get('data', {}):
                saved_tokenizer_dir = saved_config['data']['tokenizer_dir']
                print(f"  Using tokenizer_dir from checkpoint: {saved_tokenizer_dir}")
                config['data']['tokenizer_dir'] = saved_tokenizer_dir
    else:
        # Use saved config from checkpoint
        config = saved_config
        if config is None:
            raise ValueError(f"Config not found. Please provide config_path or ensure checkpoint has config. Expected: configs/transformer_{language_code}.yaml")
    
    # Get tokenizer directory - try multiple locations
    tokenizer_dir = config.get('data', {}).get('tokenizer_dir')
    if not tokenizer_dir:
        # Try language-specific tokenizer directory first
        tokenizer_dir = f"data/tokenized_{language_code}"
        if not os.path.exists(tokenizer_dir):
            # Don't fallback to generic - each language should have its own tokenizer
            raise FileNotFoundError(
                f"Tokenizer directory not found for {language_code}. "
                f"Expected: data/tokenized_{language_code} or tokenizer_dir in config. "
                f"This ensures each language uses its correct tokenizer."
            )
    
    print(f"  Tokenizer directory: {tokenizer_dir}")
    
    # Load tokenizers to get actual vocab_size
    if config.get('data', {}).get('joint_tokenizer', True):
        tokenizer_path = os.path.join(tokenizer_dir, 'tokenizer.model')
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}. Expected directory: {tokenizer_dir}")
        src_tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_path)
        tgt_tokenizer = src_tokenizer
        vocab_size = len(src_tokenizer)
    else:
        src_tokenizer_path = os.path.join(tokenizer_dir, 'src_tokenizer.model')
        tgt_tokenizer_path = os.path.join(tokenizer_dir, 'tgt_tokenizer.model')
        if not os.path.exists(src_tokenizer_path) or not os.path.exists(tgt_tokenizer_path):
            raise FileNotFoundError(f"Tokenizers not found in {tokenizer_dir}")
        src_tokenizer = spm.SentencePieceProcessor(model_file=src_tokenizer_path)
        tgt_tokenizer = spm.SentencePieceProcessor(model_file=tgt_tokenizer_path)
        vocab_size = len(tgt_tokenizer)
    
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Tokenizer directory: {tokenizer_dir}")
    
    # Get vocab_size from checkpoint if available (most reliable)
    checkpoint_vocab_size = None
    if 'model_state_dict' in checkpoint:
        # Try to infer vocab_size from checkpoint weights
        if 'tok_embed.weight' in checkpoint['model_state_dict']:
            checkpoint_vocab_size = checkpoint['model_state_dict']['tok_embed.weight'].shape[0]
            print(f"  Checkpoint vocab size: {checkpoint_vocab_size}")
    
    # Use checkpoint vocab_size if available and different (most accurate)
    if checkpoint_vocab_size and checkpoint_vocab_size != vocab_size:
        print(f"  Warning: Tokenizer vocab_size ({vocab_size}) != checkpoint vocab_size ({checkpoint_vocab_size})")
        print(f"  Using checkpoint vocab_size: {checkpoint_vocab_size}")
        vocab_size = checkpoint_vocab_size
    
    # Create model with correct vocab_size
    model = SimpleTransformerNMT(
        vocab_size=vocab_size,
        d_model=config['model']['d_model'],
        nhead=config['model']['nhead'],
        num_encoder_layers=config['model']['num_encoder_layers'],
        num_decoder_layers=config['model']['num_decoder_layers'],
        dim_feedforward=config['model']['dim_feedforward'],
        dropout=config['model']['dropout'],
        pad_idx=0
    ).to(device)
    
    # Load state dict with strict=False to handle minor mismatches
    try:
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    except RuntimeError as e:
        # If strict loading fails, try to load with strict=False and print warning
        print(f"  Warning: Strict loading failed, attempting flexible loading...")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if missing_keys:
            print(f"  Missing keys: {missing_keys[:5]}...")  # Show first 5
        if unexpected_keys:
            print(f"  Unexpected keys: {unexpected_keys[:5]}...")
        # If vocab_size mismatch, this won't work - need to fix vocab_size
        if 'tok_embed.weight' in str(e) or 'generator.weight' in str(e):
            raise RuntimeError(
                f"Vocabulary size mismatch! Checkpoint was trained with vocab_size={checkpoint_vocab_size}, "
                f"but tokenizer has vocab_size={len(src_tokenizer)}. "
                f"Make sure you're using the correct tokenizer for this model. "
                f"Tokenizer directory: {tokenizer_dir}"
            )
        raise
    
    model.eval()
    
    # Get target language from config to verify
    model_target_lang = config.get('data', {}).get('target_lang', language_code)
    if model_target_lang != language_code:
        print(f"  WARNING: Config target_lang ({model_target_lang}) != language_code ({language_code})")
        print(f"  This model may be for a different language!")
    
    # Store model with language verification
    models[language_code] = {
        'model': model,
        'src_tokenizer': src_tokenizer,
        'tgt_tokenizer': tgt_tokenizer,
        'config': config,
        'target_lang': model_target_lang  # Store actual target language from config
    }
    
    # Set as current if first model or if explicitly requested
    if current_language is None:
        current_language = language_code
    
    print(f"Model for {language_code} loaded successfully!")
    print(f"  Model target_lang: {model_target_lang}")
    return True


@app.route('/')
def index():
    """Render the main page."""
    # Always show all configured languages (even if checkpoints don't exist yet)
    # This allows users to see all options and models will load on-demand
    available_languages = []
    for lang_code, lang_config in LANGUAGE_CONFIGS.items():
        available_languages.append({
            'code': lang_code,
            'name': lang_config['name'],
            'flag': lang_config['flag']
        })
    
    return render_template('index.html', languages=available_languages)


@app.route('/translate', methods=['POST'])
def translate():
    """Translate a sentence."""
    global current_language
    
    try:
        data = request.get_json()
        sentence = data.get('sentence', '').strip()
        beam_size = data.get('beam_size', 4)
        # Get language from request, fallback to current_language only if not provided
        language = data.get('language')
        if not language:
            language = current_language
        
        if not sentence:
            return jsonify({'error': 'No sentence provided'}), 400
        
        if language is None:
            return jsonify({'error': 'No language selected and no default model loaded'}), 500
        
        if language not in models:
            # Try to load the model
            try:
                load_model(language)
            except FileNotFoundError as e:
                lang_name = LANGUAGE_CONFIGS.get(language, {}).get('name', language)
                return jsonify({
                    'error': f'Model for {lang_name} not found. Please train a model first or set CHECKPOINT_PATH_{language.upper()} environment variable.',
                    'details': str(e)
                }), 404
            except Exception as e:
                return jsonify({'error': f'Could not load model for {language}: {str(e)}'}), 500
        
        # Update current_language to the one being used
        current_language = language
        
        # Verify model exists for this language
        if language not in models:
            return jsonify({
                'error': f'Model for {language} not loaded. Please switch to this language first.',
                'language': language
            }), 500
        
        print(f"Translating to {language} using model: {list(models.keys())}")
        print(f"  Requested language: {language}")
        print(f"  Available models: {list(models.keys())}")
        
        if language not in models:
            return jsonify({
                'error': f'Model for {language} not loaded. Available models: {list(models.keys())}',
                'language': language
            }), 500
        
        model_data = models[language]
        
        # Verify we're using the correct model
        model_target_lang = model_data.get('target_lang') or model_data.get('config', {}).get('data', {}).get('target_lang', 'unknown')
        print(f"  Model target_lang: {model_target_lang}")
        print(f"  Model tokenizer dir: {model_data.get('config', {}).get('data', {}).get('tokenizer_dir', 'unknown')}")
        
        if model_target_lang != language:
            print(f"  ERROR: Model target_lang ({model_target_lang}) != requested language ({language})!")
            return jsonify({
                'error': f'Model mismatch! Model stored for "{language}" is actually for "{model_target_lang}". This means the wrong checkpoint was loaded. Please check that checkpoints/best_{language}.pt exists and is correct.',
                'language': language,
                'model_lang': model_target_lang,
                'help': f'Expected checkpoint: checkpoints/best_{language}.pt should contain a model for {language}'
            }), 500
        
        translation = translate_sentence(
            model_data['model'],
            sentence,
            model_data['src_tokenizer'],
            model_data['tgt_tokenizer'],
            device,
            beam_size
        )
        
        print(f"  Translation completed for {language}")
        
        return jsonify({
            'source': sentence,
            'translation': translation,
            'language': language,
            'success': True
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/languages', methods=['GET'])
def get_languages():
    """Get list of available languages."""
    available = []
    for lang_code, lang_config in LANGUAGE_CONFIGS.items():
        is_loaded = lang_code in models
        checkpoint_path = lang_config['checkpoint']
        checkpoint_exists = checkpoint_path and os.path.exists(checkpoint_path)
        
        available.append({
            'code': lang_code,
            'name': lang_config['name'],
            'flag': lang_config['flag'],
            'loaded': is_loaded,
            'available': checkpoint_exists or is_loaded
        })
    
    return jsonify({'languages': available, 'current': current_language})


@app.route('/switch_language', methods=['POST'])
def switch_language():
    """Switch to a different language model."""
    global current_language
    
    try:
        data = request.get_json()
        language = data.get('language')
        
        if not language:
            return jsonify({'error': 'No language specified'}), 400
        
        if language not in LANGUAGE_CONFIGS:
            return jsonify({'error': f'Language {language} not supported'}), 400
        
        # Load model if not already loaded
        if language not in models:
            try:
                load_model(language)
            except FileNotFoundError as e:
                # Provide user-friendly error message
                lang_name = LANGUAGE_CONFIGS[language]['name']
                error_msg = (
                    f"Model for {lang_name} not found. "
                    f"Please train a model first or set CHECKPOINT_PATH_{language.upper()} environment variable. "
                    f"See quickstart.md for instructions."
                )
                return jsonify({
                    'error': error_msg,
                    'details': str(e),
                    'help': f'To train a model for {lang_name}, run: bash scripts/train_indian_language.sh {language} 1000'
                }), 404
            except Exception as e:
                return jsonify({
                    'error': f'Could not load model: {str(e)}',
                    'help': 'Check server logs for details'
                }), 500
        
        current_language = language
        
        return jsonify({
            'success': True,
            'language': language,
            'name': LANGUAGE_CONFIGS[language]['name']
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'models_loaded': list(models.keys()),
        'current_language': current_language,
        'device': str(device) if device else None
    })


if __name__ == '__main__':
    # Try to load default models on startup
    print("Checking for available models...")
    
    # Try to load German model (default)
    if 'de' in LANGUAGE_CONFIGS and LANGUAGE_CONFIGS['de']['checkpoint']:
        if os.path.exists(LANGUAGE_CONFIGS['de']['checkpoint']):
            try:
                load_model('de')
                print("✓ German model loaded")
            except Exception as e:
                print(f"⚠ Could not load German model: {e}")
    
    # Try to load Hindi model if available
    if 'hi' in LANGUAGE_CONFIGS and LANGUAGE_CONFIGS['hi']['checkpoint']:
        if os.path.exists(LANGUAGE_CONFIGS['hi']['checkpoint']):
            try:
                load_model('hi')
                print("✓ Hindi model loaded")
            except Exception as e:
                print(f"⚠ Could not load Hindi model: {e}")
    
    # Try to load from environment variables (backward compatibility)
    checkpoint_path = os.getenv('CHECKPOINT_PATH')
    config_path = os.getenv('CONFIG_PATH')
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            # Try to detect language from config or default to 'de'
            lang = 'de'
            if config_path and 'hindi' in config_path.lower():
                lang = 'hi'
            load_model(lang, checkpoint_path, config_path)
            print(f"✓ Model loaded from environment variables ({lang})")
        except Exception as e:
            print(f"⚠ Could not load model from environment: {e}")
    
    if not models:
        print("⚠ No models loaded. Translation will not work until a model is loaded.")
        print("Available languages:", list(LANGUAGE_CONFIGS.keys()))
        print("Set CHECKPOINT_PATH_<LANG> and CONFIG_PATH_<LANG> environment variables")
    
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

