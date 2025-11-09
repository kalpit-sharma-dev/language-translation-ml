# Deep Learning-based Neural Machine Translation

A complete implementation of a Transformer-based Neural Machine Translation (NMT) system supporting:
- **English → German** translation (WMT dataset)
- **English → Indian Languages** (Hindi, Bengali, Telugu, Tamil, Gujarati, and more)
- **Any language pair** with parallel corpus data

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Supported Languages](#supported-languages)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Indian Languages Support](#indian-languages-support)
- [Usage](#usage)
- [Configuration](#configuration)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [Web UI](#web-ui)
- [Experiments](#experiments)
- [Results](#results)
- [Timeline & Milestones](#timeline--milestones)
- [References](#references)

## 🎯 Project Overview

This project implements a complete Neural Machine Translation system using the Transformer architecture. The system supports:
- **English → German** translation (WMT dataset)
- **English → Indian Languages** (Hindi, Bengali, Telugu, Tamil, Gujarati, Kannada, Malayalam, Marathi, Punjabi, Urdu, and more)
- **Any language pair** - the architecture is language-agnostic

The system can be trained on any parallel corpus and translates between language pairs with state-of-the-art performance.

### Key Components

- **Transformer Architecture**: 6-layer encoder-decoder with multi-head attention
- **SentencePiece Tokenization**: BPE-based subword tokenization
- **Training Pipeline**: Complete training loop with validation and checkpointing
- **Evaluation Metrics**: BLEU, chrF scores using sacreBLEU
- **Inference**: Beam search decoding with configurable beam width
- **Web Interface**: Flask-based web UI for interactive translation

## ✨ Features

- ✅ Complete Transformer implementation from scratch
- ✅ SentencePiece tokenization (joint or separate vocabularies)
- ✅ Label smoothing and learning rate scheduling
- ✅ Beam search inference with length penalty
- ✅ Comprehensive evaluation with BLEU and chrF metrics
- ✅ Web-based translation interface
- ✅ Configurable hyperparameters via YAML
- ✅ TensorBoard logging
- ✅ Model checkpointing and resuming
- ✅ Support for multiple experiment configurations
- ✅ **Indian Languages Support**: Hindi, Bengali, Telugu, Tamil, Gujarati, and more
- ✅ **Multilingual Ready**: Easy to extend to any language pair

## 🌏 Supported Languages

### European Languages
- ✅ English → German (WMT dataset)
- ✅ Can be extended to: French, Spanish, Italian, etc.

### Indian Languages
- ✅ **Hindi** (Devanagari script)
- ✅ **Bengali** (Bengali script)
- ✅ **Telugu** (Telugu script)
- ✅ **Tamil** (Tamil script)
- ✅ **Gujarati** (Gujarati script)
- ✅ **Kannada** (Kannada script)
- ✅ **Malayalam** (Malayalam script)
- ✅ **Marathi** (Devanagari script)
- ✅ **Punjabi** (Gurmukhi script)
- ✅ **Urdu** (Perso-Arabic script)
- ✅ **Odia** (Odia script)
- ✅ **Assamese** (Assamese script)

**Language-Specific Considerations:**
- **Script Handling**: SentencePiece handles all Indian scripts (Devanagari, Bengali, Dravidian, etc.)
- **Vocabulary Size**: Auto-adjusts for small datasets (500-2000 for <10K sentences, 32000+ for large datasets)
- **Data Sources**: IIT Bombay corpus, Samanantar, OPUS
- **Character Coverage**: Uses 0.9995 coverage which works well for Indian scripts

**For Other Indian Languages:** Replace `hi` with `bn` (Bengali), `te` (Telugu), `ta` (Tamil), `gu` (Gujarati), etc.

## 📁 Project Structure

```
wmt-en-de-nmt/
├── data/
│   ├── raw/                  # Downloaded raw files
│   ├── cleaned/              # Preprocessed data
│   └── tokenized/            # Tokenizer models
├── src/
│   ├── __init__.py
│   ├── model.py              # Transformer model
│   ├── positional_encoding.py
│   ├── data_loader.py        # Data loading utilities
│   ├── tokenizer_train.py   # Tokenizer training
│   ├── trainer.py            # Training script
│   ├── evaluate.py           # Evaluation script
│   └── infer.py              # Inference script
├── configs/
│   └── transformer_base.yaml # Configuration file
├── scripts/
│   ├── train.sh              # Training script
│   ├── eval.sh               # Evaluation script
│   └── train_tokenizer.sh    # Tokenizer training script
├── experiments/              # Experiment outputs
├── notebooks/                # Jupyter notebooks for analysis
├── checkpoints/              # Model checkpoints
├── logs/                     # TensorBoard logs
├── results/                  # Evaluation results
├── templates/                # Web UI templates
├── static/                   # Web UI static files
├── app.py                    # Flask web application
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── quickstart.md            # Quick start guide
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended) or CPU
- 8GB+ RAM
- 10GB+ disk space for data and models

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd wmt-en-de-nmt
```

### Step 2: Create Virtual Environment

```bash
# Using conda
conda create -n nmt python=3.9
conda activate nmt

# Or using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download NLTK Data (Optional)

```bash
python -c "import nltk; nltk.download('punkt')"
```

## 🏃 Quick Start

For a detailed quick start guide, see [quickstart.md](quickstart.md).

### For English → German
Follow the standard quick start guide.

### For English → Indian Languages
See [Indian Languages Support](#indian-languages-support) section below or [docs/INDIAN_LANGUAGES.md](docs/INDIAN_LANGUAGES.md).

## 🌏 Indian Languages Support

The project fully supports translation to and from Indian languages. Here's a quick example for **English → Hindi**:

```bash
# 1. Download Hindi dataset
python scripts/download_indian_languages.py \
    --target_lang hi \
    --use_sample \
    --sample_size 1000 \
    --split

# 2. Train tokenizer
bash scripts/train_tokenizer.sh \
    data/cleaned/train.en \
    data/cleaned/train.hi \
    data/tokenized \
    32000 \
    true

# 3. Train model (use Hindi config)
python -m src.trainer --config configs/transformer_hindi.yaml

# 4. Translate
python -m src.infer \
    --checkpoint checkpoints/best.pt \
    --input "Hello, how are you?" \
    --beam_size 4
```

**Supported Languages:** Hindi, Bengali, Telugu, Tamil, Gujarati, Kannada, Malayalam, Marathi, Punjabi, Urdu, Odia, Assamese

**Language Codes:** `hi`, `bn`, `te`, `ta`, `gu`, `kn`, `ml`, `mr`, `pa`, `ur`, `or`, `as`

**One-Command Training for All Languages:**
```bash
# Train models for all 6 languages at once
python scripts/train_all_languages.py 1000
```

This automatically:
- Downloads/creates datasets for German, Hindi, Tamil, Telugu, Gujarati, Bengali
- Trains tokenizers for each language
- Creates config files for each language
- Trains models for all languages
- Saves each model as `checkpoints/best_<lang>.pt`

**Train Individual Languages:**
```bash
# For Hindi
bash scripts/train_indian_language.sh hi 1000

# For Bengali
bash scripts/train_indian_language.sh bn 1000

# For Tamil, Telugu, Gujarati - replace 'bn' with 'ta', 'te', 'gu'
```

### 1. Prepare Data

```bash
# Download sample data for testing
python scripts/download_data.py --use_sample --sample_size 1000 --split

# Or download full WMT dataset
python scripts/download_data.py --split
```

### 2. Train Tokenizer

```bash
# Train joint tokenizer
bash scripts/train_tokenizer.sh \
    data/cleaned/train.en \
    data/cleaned/train.de \
    data/tokenized \
    32000 \
    true
```

### 3. Train Model

```bash
# Update configs/transformer_base.yaml with your data paths
bash scripts/train.sh configs/transformer_base.yaml
```

### 4. Evaluate

```bash
bash scripts/eval.sh checkpoints/best.pt
```

### 5. Run Web UI

```bash
# Set checkpoint path
export CHECKPOINT_PATH=checkpoints/best.pt
python app.py
```

Then open `http://localhost:5000` in your browser.

## 📖 Usage

### Training

```bash
python -m src.trainer --config configs/transformer_base.yaml
```

Resume from checkpoint:
```bash
python -m src.trainer --config configs/transformer_base.yaml --resume checkpoints/latest.pt
```

### Evaluation

```bash
python -m src.evaluate \
    --checkpoint checkpoints/best.pt \
    --config configs/transformer_base.yaml \
    --output results/eval_results.txt
```

### Inference

Translate a single sentence:
```bash
python -m src.infer \
    --checkpoint checkpoints/best.pt \
    --input "Hello, how are you?" \
    --beam_size 4
```

Translate a file:
```bash
python -m src.infer \
    --checkpoint checkpoints/best.pt \
    --input data/test_sentences.txt \
    --output translations.txt \
    --beam_size 4
```

## ⚙️ Configuration

Configuration files are in YAML format. See `configs/transformer_base.yaml` for an example.

### Key Parameters

**Model:**
- `d_model`: Model dimension (default: 512)
- `nhead`: Number of attention heads (default: 8)
- `num_encoder_layers`: Encoder layers (default: 6)
- `num_decoder_layers`: Decoder layers (default: 6)
- `dim_feedforward`: Feedforward dimension (default: 2048)
- `dropout`: Dropout rate (default: 0.1)

**Training:**
- `batch_size`: Batch size (default: 32)
- `num_epochs`: Number of epochs (default: 30)
- `lr`: Learning rate (default: 5e-4)
- `warmup_steps`: Warmup steps (default: 4000)
- `label_smoothing`: Label smoothing (default: 0.1)
- `clip_grad`: Gradient clipping (default: 1.0)

**Data:**
- `max_len`: Maximum sequence length (default: 250)
- `joint_tokenizer`: Use shared vocabulary (default: true)
- `vocab_size`: Vocabulary size (default: 32000)

## 🎓 Training

### Training Process

1. **Data Preparation**: Clean and preprocess parallel corpora
2. **Tokenization**: Train SentencePiece tokenizer
3. **Model Training**: Train Transformer model with validation
4. **Evaluation**: Evaluate on test set with BLEU/chrF

### Monitoring Training

View TensorBoard logs:
```bash
tensorboard --logdir logs
```

### Checkpointing

- Best model (by dev BLEU): `checkpoints/best.pt`
- Latest model: `checkpoints/latest.pt`

## 📊 Evaluation

### Metrics

- **BLEU**: Bilingual Evaluation Understudy (sacreBLEU)
- **chrF**: Character-level F-score

### Evaluation on Test Set

```bash
python -m src.evaluate \
    --checkpoint checkpoints/best.pt \
    --test_src data/cleaned/test.en \
    --test_tgt data/cleaned/test.de \
    --output results/test_results.txt
```

## 🔮 Inference

### Beam Search

The inference uses beam search with configurable parameters:
- `beam_size`: Number of beams (default: 4)
- `length_penalty`: Length penalty factor (default: 0.6)
- `max_len`: Maximum target length (default: 250)

### Example

```python
from src.infer import translate_sentence
import sentencepiece as spm
import torch

# Load model and tokenizers (see infer.py for details)
translation = translate_sentence(
    model, "Hello, how are you?", 
    src_tokenizer, tgt_tokenizer, device
)
print(translation)
```

## 🌐 Web UI

The project includes a Flask-based web interface for interactive translation with **multi-language support**.

### Start Server

**Single Model:**
```bash
export CHECKPOINT_PATH=checkpoints/best.pt
export CONFIG_PATH=configs/transformer_base.yaml
python app.py
```

**Multiple Models (German + Hindi):**
```bash
export CHECKPOINT_PATH_DE=checkpoints/best.pt
export CONFIG_PATH_DE=configs/transformer_base.yaml
export CHECKPOINT_PATH_HI=checkpoints/best_hi.pt
export CONFIG_PATH_HI=configs/transformer_hindi.yaml
python app.py
```

### Access UI

Open `http://localhost:5000` in your browser.

### Features

- ✅ **Language Selection**: Dropdown to select target language (🇩🇪 German, 🇮🇳 Hindi, 🇧🇩 Bengali, 🇮🇳 Telugu, 🇮🇳 Tamil, 🇮🇳 Gujarati)
- ✅ **Multi-Model Support**: Load and switch between multiple language models simultaneously
- ✅ **Real-time Translation**: Instant translation as you type
- ✅ **Configurable Beam Size**: Choose between greedy (1) or beam search (4, 8)
- ✅ **Example Sentences**: Click to try pre-loaded examples
- ✅ **Dynamic Loading**: Models load automatically when selected
- ✅ **Clean, Modern Interface**: Beautiful gradient design with language flags

**Language Dropdown:** The UI shows all 6 configured languages (German, Hindi, Bengali, Telugu, Tamil, Gujarati). Models load automatically when selected.

**Setting Up Multiple Languages:**

1. **Train models for each language** and save with language-specific names:
   ```bash
   # Train German model
   python scripts/download_data.py --use_sample --sample_size 1000 --split
   bash scripts/train_tokenizer.sh data/cleaned/train.en data/cleaned/train.de data/tokenized 32000 true
   python -m src.trainer --config configs/transformer_base.yaml
   cp checkpoints/best.pt checkpoints/best_de.pt
   
   # Train Hindi model
   bash scripts/train_indian_language.sh hi 1000
   cp checkpoints/best.pt checkpoints/best_hi.pt
   ```

2. **Start web UI** - models are auto-detected if named correctly:
   ```bash
   # Option A: Auto-detection (if models named checkpoints/best_<lang>.pt)
   python app.py
   
   # Option B: Environment variables (recommended)
   export CHECKPOINT_PATH_DE=checkpoints/best_de.pt
   export CONFIG_PATH_DE=configs/transformer_base.yaml
   export CHECKPOINT_PATH_HI=checkpoints/best_hi.pt
   export CONFIG_PATH_HI=configs/transformer_hindi.yaml
   python app.py
   ```

3. **Auto-detection paths checked:**
   - `checkpoints/best_<lang>.pt` (e.g., `checkpoints/best_hi.pt`)
   - `checkpoints/<lang>/best.pt` (e.g., `checkpoints/hi/best.pt`)
   - Environment variables: `CHECKPOINT_PATH_<LANG>`

**If a model isn't found:** The UI shows a helpful error message with instructions on how to train the model.

## 🧪 Experiments

### Suggested Experiments

1. **Baseline**: 6/6 layers, 32k vocab, joint tokenizer
2. **Deeper Model**: 12/12 layers
3. **Separate Vocabularies**: Source and target tokenizers
4. **Vocabulary Size**: 16k, 32k, 50k
5. **Label Smoothing**: On/off
6. **Learning Rate Schedulers**: Inverse sqrt vs linear

### Running Experiments

Create experiment configs in `configs/` and run:

```bash
python -m src.trainer --config configs/experiment_1.yaml
```

## 📈 Results

### Expected Performance

- **Baseline (6/6 layers)**: ~28-30 BLEU on WMT En-De
- **Large (12/12 layers)**: ~30-32 BLEU

### Results Table

| Model | Vocab | Layers | Params | Dev BLEU | Test BLEU | Notes |
|-------|-------|--------|--------|----------|-----------|-------|
| Baseline | 32k (joint) | 6/6 | ~65M | - | - | Initial baseline |
| Large | 32k | 12/12 | ~200M | - | - | More layers |

*Replace with your actual results*

## 📅 Timeline & Milestones

- **Week 1**: Setup, data download, tokenizer experiments
- **Week 2**: Baseline model implementation, training loop
- **Week 3**: Full training, validation, logging
- **Week 4**: Ablation studies, variants
- **Week 5**: Evaluation, error analysis
- **Week 6**: Inference demo, web UI, documentation

## 📚 References

1. Vaswani et al. (2017). "Attention is All You Need". NeurIPS.
2. Kudo & Richardson (2018). "SentencePiece: A simple and language independent subword tokenizer". EMNLP.
3. Post (2018). "A Call for Clarity in Reporting BLEU Scores". WMT.
4. WMT: http://www.statmt.org/wmt14/

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- WMT for providing the parallel corpus
- Hugging Face for datasets and transformers libraries
- SentencePiece team for tokenization tools

---

For detailed instructions, see [quickstart.md](quickstart.md).

