# Deep Learning-based English → German Translation

A complete implementation of a Transformer-based Neural Machine Translation (NMT) system for English to German translation using the WMT parallel corpus or other datasets.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
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

This project implements a complete Neural Machine Translation system using the Transformer architecture. The system is trained on parallel English-German corpora and can translate English sentences to German with state-of-the-art performance.

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

The project includes a Flask-based web interface for interactive translation.

### Start Server

```bash
export CHECKPOINT_PATH=checkpoints/best.pt
export CONFIG_PATH=configs/transformer_base.yaml
python app.py
```

### Access UI

Open `http://localhost:5000` in your browser.

### Features

- Real-time translation
- Configurable beam size
- Example sentences
- Clean, modern interface

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

