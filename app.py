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

# Global variables for model and tokenizers
model = None
src_tokenizer = None
tgt_tokenizer = None
device = None
config = None


def load_model(checkpoint_path: str, config_path: str = None):
    """Load the trained model and tokenizers."""
    global model, src_tokenizer, tgt_tokenizer, device, config
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config')
    
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    
    if config is None:
        raise ValueError("Config not found in checkpoint and config_path not provided")
    
    # Load tokenizers
    tokenizer_dir = config['data']['tokenizer_dir']
    if config['data']['joint_tokenizer']:
        tokenizer_path = os.path.join(tokenizer_dir, 'tokenizer.model')
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
        src_tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_path)
        tgt_tokenizer = src_tokenizer
        vocab_size = len(src_tokenizer)
    else:
        src_tokenizer_path = os.path.join(tokenizer_dir, 'src_tokenizer.model')
        tgt_tokenizer_path = os.path.join(tokenizer_dir, 'tgt_tokenizer.model')
        if not os.path.exists(src_tokenizer_path) or not os.path.exists(tgt_tokenizer_path):
            raise FileNotFoundError(f"Tokenizers not found")
        src_tokenizer = spm.SentencePieceProcessor(model_file=src_tokenizer_path)
        tgt_tokenizer = spm.SentencePieceProcessor(model_file=tgt_tokenizer_path)
        vocab_size = len(tgt_tokenizer)
    
    # Create model
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
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("Model loaded successfully!")


@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')


@app.route('/translate', methods=['POST'])
def translate():
    """Translate a sentence."""
    try:
        data = request.get_json()
        sentence = data.get('sentence', '').strip()
        beam_size = data.get('beam_size', 4)
        
        if not sentence:
            return jsonify({'error': 'No sentence provided'}), 400
        
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        translation = translate_sentence(
            model, sentence, src_tokenizer, tgt_tokenizer, device, beam_size
        )
        
        return jsonify({
            'source': sentence,
            'translation': translation,
            'success': True
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device) if device else None
    })


if __name__ == '__main__':
    # Load model on startup
    checkpoint_path = os.getenv('CHECKPOINT_PATH', 'checkpoints/best.pt')
    config_path = os.getenv('CONFIG_PATH', 'configs/transformer_base.yaml')
    
    if os.path.exists(checkpoint_path):
        try:
            load_model(checkpoint_path, config_path)
        except Exception as e:
            print(f"Warning: Could not load model: {e}")
            print("You can still start the server, but translation will not work until model is loaded.")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        print("Set CHECKPOINT_PATH environment variable or place model at checkpoints/best.pt")
    
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

