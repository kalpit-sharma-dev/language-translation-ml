# Quick Start Guide

This guide will help you get started with the Neural Machine Translation project quickly.

## Prerequisites

- Python 3.8+
- pip or conda
- GPU (optional but recommended for training)
- CUDA-capable GPU (if using GPU)

## Step 1: Installation

### 1.1 Create and Activate Virtual Environment

**Using conda:**
```bash
conda create -n nmt python=3.9
conda activate nmt
```

**Using venv:**
```bash
python -m venv venv
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 1.2 Install Dependencies

```bash
pip install -r requirements.txt
```

**Verify installation:**
```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

## Step 2: Prepare Data

### Option A: Use Sample Data (Quick Testing) - RECOMMENDED FOR FIRST RUN

For quick testing, create a small sample dataset:

```bash
python scripts/download_data.py --use_sample --sample_size 1000 --split
```

This creates:
- `data/raw/train.en` and `data/raw/train.de` (1000 examples)
- `data/cleaned/train.en`, `train.de` (~980 examples after filtering)
- `data/cleaned/dev.en`, `dev.de` (~10 examples)
- `data/cleaned/test.en`, `test.de` (~10 examples)

**Note:** The actual number of examples may be slightly less after length filtering.

### Option B: Download Full WMT Dataset

For full training, download the WMT dataset:

```bash
# Download from Hugging Face
python scripts/download_data.py --split

# Or manually download from:
# http://www.statmt.org/wmt14/translation-task.html
# Then place files in data/raw/ and run:
python scripts/download_data.py --split
```

### Option C: Use Your Own Data

1. Place your parallel files in `data/raw/`:
   - `train.en` (English source)
   - `train.de` (German target)

2. Split into train/dev/test:
```bash
python scripts/download_data.py --split --split_dir data/cleaned
```

## Step 3: Train Tokenizer

Train a SentencePiece tokenizer on your data:

```bash
# Train joint tokenizer (shared vocabulary)
# Note: Vocab size will be auto-adjusted for small datasets
bash scripts/train_tokenizer.sh \
    data/cleaned/train.en \
    data/cleaned/train.de \
    data/tokenized \
    32000 \
    true
```

**Important Notes:**
- The script automatically adjusts vocabulary size for small datasets
- For datasets < 10,000 sentences, vocab size is reduced proportionally
- Minimum vocab size is 500
- For large datasets (WMT), you can use 32000 vocab size
- You'll see output like: "Dataset has 980 lines. Using vocab_size=500 (auto-adjusted from 32000)"

**Alternative: Train separate tokenizers**
```bash
bash scripts/train_tokenizer.sh \
    data/cleaned/train.en \
    data/cleaned/train.de \
    data/tokenized \
    32000 \
    false
```

This creates:
- `data/tokenized/tokenizer.model` and `data/tokenized/tokenizer.vocab` (if joint)
- `data/tokenized/src_tokenizer.model` and `tgt_tokenizer.model` (if separate)

**Verify tokenizer was created:**
```bash
ls -lh data/tokenized/
```

## Step 4: Configure Training

The config file `configs/transformer_base.yaml` is already set up with correct paths. Verify it matches your setup:

```yaml
data:
  train_src: data/cleaned/train.en
  train_tgt: data/cleaned/train.de
  dev_src: data/cleaned/dev.en
  dev_tgt: data/cleaned/dev.de
  test_src: data/cleaned/test.en
  test_tgt: data/cleaned/test.de
  tokenizer_dir: data/tokenized
  joint_tokenizer: true  # Set to false if using separate tokenizers
  max_len: 250
```

**For small datasets**, the config is already optimized:

```yaml
training:
  batch_size: 16  # Reduced for small datasets
  num_epochs: 10  # Reduced for quick testing
  warmup_steps: 400  # Reduced proportionally
  lr: 5e-4
  label_smoothing: 0.1
```

**For large datasets (WMT)**, you may want to increase:
- `batch_size: 32` or higher
- `num_epochs: 30`
- `warmup_steps: 4000`

## Step 5: Train Model

### Quick Training (Small Dataset)

The config is already optimized for small datasets. Simply run:

```bash
bash scripts/train.sh configs/transformer_base.yaml
```

Or directly:

```bash
python -m src.trainer --config configs/transformer_base.yaml
```

**What happens during training:**
- Model loads with vocabulary size from tokenizer (auto-adjusted)
- Training progress shown with loss and learning rate
- Dev BLEU evaluated after each epoch
- Best checkpoint saved to `checkpoints/best.pt`
- Latest checkpoint saved to `checkpoints/latest.pt`
- TensorBoard logs saved to `logs/`

**Expected output:**
```
Using device: cuda
Vocabulary size: 500  # Auto-adjusted for small dataset
Loaded 980 sentence pairs
Model parameters: 44,653,044

Epoch 1/10
Training: 100%|...| loss=2.20, lr=0.00034
Train Loss: 2.2048
Dev BLEU: 96.83
New best BLEU: 96.83, checkpoint saved!
```

**Note:** High BLEU scores on small datasets are normal due to limited vocabulary and repetitive patterns.

### Full Training

For full training on WMT dataset:

```bash
python -m src.trainer --config configs/transformer_base.yaml
```

Training will:
- Save checkpoints to `checkpoints/`
- Log to TensorBoard in `logs/`
- Save best model based on dev BLEU

### Monitor Training

In another terminal:

```bash
tensorboard --logdir logs
```

Open `http://localhost:6006` to view training progress.

**Training time estimates:**
- Small dataset (1000 sentences): ~4-5 seconds per epoch on GPU
- Large dataset (WMT): Several hours depending on GPU

## Step 6: Evaluate

Evaluate the trained model:

```bash
bash scripts/eval.sh checkpoints/best.pt
```

Or:

```bash
python -m src.evaluate \
    --checkpoint checkpoints/best.pt \
    --config configs/transformer_base.yaml \
    --output results/eval_results.txt
```

This computes BLEU and chrF scores on the test set.

**Expected output:**
```
Computing BLEU score...
Computing chrF score...

Results:
BLEU: 28.59
chrF: 45.23
```

## Step 7: Translate

### Command Line

Translate a single sentence:

```bash
python -m src.infer \
    --checkpoint checkpoints/best.pt \
    --input "Hello, how are you?" \
    --beam_size 4
```

**Output:**
```
Translation: Hallo, wie geht es dir?
```

Translate a file:

```bash
python -m src.infer \
    --checkpoint checkpoints/best.pt \
    --input data/test_sentences.txt \
    --output translations.txt \
    --beam_size 4
```

### Python API

```python
from src.infer import translate_sentence
import sentencepiece as spm
import torch
import yaml

# Load config
with open('configs/transformer_base.yaml') as f:
    config = yaml.safe_load(f)

# Load tokenizers
tokenizer = spm.SentencePieceProcessor(
    model_file='data/tokenized/tokenizer.model'
)

# Load model (see src/infer.py for full example)
# ...

# Translate
translation = translate_sentence(
    model, "Hello, how are you?",
    tokenizer, tokenizer, device
)
print(translation)
```

## Step 8: Web UI

Start the web interface:

**On Linux/Mac:**
```bash
export CHECKPOINT_PATH=checkpoints/best.pt
export CONFIG_PATH=configs/transformer_base.yaml
python app.py
```

**On Windows:**
```cmd
set CHECKPOINT_PATH=checkpoints/best.pt
set CONFIG_PATH=configs/transformer_base.yaml
python app.py
```

**Or set environment variables inline (cross-platform):**
```bash
CHECKPOINT_PATH=checkpoints/best.pt CONFIG_PATH=configs/transformer_base.yaml python app.py
```

Open `http://localhost:5000` in your browser.

**Note:** If the model checkpoint doesn't exist yet, the server will start but translation won't work until you complete training.

### Using Web UI

1. Enter English text in the left textarea
2. Select beam size (1, 4, or 8)
3. Click "Translate" or press Ctrl+Enter (Cmd+Enter on Mac)
4. View German translation in the right textarea
5. Click example sentences to try them quickly

## Complete End-to-End Example

Here's a complete example from scratch:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Create sample data
python scripts/download_data.py --use_sample --sample_size 1000 --split

# 3. Train tokenizer
bash scripts/train_tokenizer.sh data/cleaned/train.en data/cleaned/train.de data/tokenized 32000 true

# 4. Train model
bash scripts/train.sh configs/transformer_base.yaml

# 5. Evaluate
bash scripts/eval.sh checkpoints/best.pt

# 6. Test translation
python -m src.infer --checkpoint checkpoints/best.pt --input "Hello, how are you?" --beam_size 4

# 7. Start web UI
CHECKPOINT_PATH=checkpoints/best.pt CONFIG_PATH=configs/transformer_base.yaml python app.py
```

## Troubleshooting

### Common Issues

**1. Out of Memory (OOM)**
- Reduce `batch_size` in config (e.g., `batch_size: 8`)
- Reduce `max_len` (e.g., `max_len: 128`)
- Use gradient accumulation (not implemented yet, but can be added)

**2. Tokenizer Not Found or Vocabulary Size Error**
- Check `tokenizer_dir` in config
- Ensure tokenizer files exist in `data/tokenized/`
- If you see "Vocabulary size too high" error, the script now auto-fixes this
- For very small datasets (< 100 sentences), manually set smaller vocab size (e.g., 500)
- Verify tokenizer was created: `ls data/tokenized/`

**3. Data Files Not Found**
- Verify file paths in config match actual file locations
- Check that data files exist: `ls data/cleaned/`
- Ensure you ran the data preparation step

**4. CUDA Out of Memory**
- Use CPU: The code automatically falls back to CPU if CUDA is unavailable
- Reduce batch size in config
- Use smaller model (reduce `d_model`, `num_encoder_layers`, etc.)

**5. Import Errors**
- Ensure virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt`
- Check Python version: `python --version` (should be 3.8+)

**6. Learning Rate Type Error**
- This is now fixed automatically - the code converts YAML string to float
- If you see this error, update to latest code version

**7. Mask Shape Mismatch Error**
- This is now fixed - masks are properly aligned with sequence lengths
- If you see this error, update to latest code version

**8. File Path Issues on Windows**
- Use forward slashes in config files: `data/cleaned/train.en`
- Paths work correctly on both Windows and Linux/Mac

### Quick Test (End-to-End Verification)

To verify everything works from start to finish:

```bash
# 1. Create sample data (100 sentences for quick test)
python scripts/download_data.py --use_sample --sample_size 100 --split

# 2. Train tokenizer (vocab size auto-adjusted to ~500 for 100 sentences)
bash scripts/train_tokenizer.sh \
    data/cleaned/train.en \
    data/cleaned/train.de \
    data/tokenized \
    32000 \
    true

# 3. Update config for quick test (optional - edit configs/transformer_base.yaml)
# Set: num_epochs: 2, batch_size: 8

# 4. Train model (2 epochs for quick test)
python -m src.trainer --config configs/transformer_base.yaml

# 5. Test inference
python -m src.infer \
    --checkpoint checkpoints/best.pt \
    --input "Hello, how are you?" \
    --beam_size 4
```

**Expected results:**
- Tokenizer trains successfully (vocab size auto-adjusted)
- Model trains without errors
- Inference produces German translation

## Next Steps

1. **Experiment with Hyperparameters**: Try different model sizes, learning rates
2. **Ablation Studies**: Test joint vs separate vocabularies, label smoothing
3. **Evaluate on Test Set**: Get final BLEU scores
4. **Error Analysis**: Analyze translation errors
5. **Deploy**: Export model for production use
6. **Scale Up**: Use full WMT dataset for production-quality model

## Tips

- **Start with small data** (100-1000 sentences) for quick iteration and testing
- **Monitor training** with TensorBoard: `tensorboard --logdir logs`
- **Checkpoints are saved automatically** - best model saved to `checkpoints/best.pt`
- **Use beam search** (beam_size=4) for better translations than greedy decoding
- **Vocabulary size auto-adjusts** for small datasets - no need to manually tune
- **For production**, use full WMT dataset with 32000 vocab size
- **Training time**: ~4-5 seconds per epoch for 980 sentences on GPU
- **Evaluation time**: ~60-90 seconds per epoch (BLEU computation)
- **Resume training**: Use `--resume checkpoints/latest.pt` to continue from last checkpoint

## Getting Help

- Check `README.md` for detailed documentation
- Review code comments in `src/` directory
- Check TensorBoard logs for training issues
- Verify data format matches expected structure
- Check that all file paths in config are correct

---

Happy translating! 🚀
