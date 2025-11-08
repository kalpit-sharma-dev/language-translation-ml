"""
Evaluation script with BLEU, chrF, and other metrics.
"""
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
import sentencepiece as spm
from sacrebleu import BLEU, CHRF
import yaml
import os

from .model import SimpleTransformerNMT
from .data_loader import create_dataloader
from .infer import translate_batch


def evaluate_bleu(
    model: nn.Module,
    data_loader,
    tgt_tokenizer: spm.SentencePieceProcessor,
    device: torch.device,
    beam_size: int = 4,
    max_len: int = 250
) -> float:
    """Evaluate model and compute BLEU score."""
    model.eval()
    hypotheses = []
    references = []
    
    bos_id = 1
    eos_id = 2
    pad_id = 0
    
    with torch.no_grad():
        for src, tgt, src_mask, tgt_mask in tqdm(data_loader, desc='Evaluating'):
            src = src.to(device)
            src_mask = src_mask.to(device)
            
            # Translate
            batch_hypotheses = translate_batch(
                model, src, src_mask, tgt_tokenizer, device,
                beam_size=beam_size, max_len=max_len, bos_id=bos_id, eos_id=eos_id
            )
            
            # Decode references
            for i in range(tgt.size(1)):
                tgt_ids = tgt[:, i].cpu().tolist()
                # Remove padding and special tokens
                tgt_ids = [tid for tid in tgt_ids if tid not in [pad_id, bos_id, eos_id]]
                ref_text = tgt_tokenizer.decode(tgt_ids)
                references.append(ref_text)
            
            hypotheses.extend(batch_hypotheses)
    
    # Compute BLEU
    bleu = BLEU()
    bleu_score = bleu.corpus_score(hypotheses, [references]).score
    
    return bleu_score


def evaluate_chrf(
    model: nn.Module,
    data_loader,
    tgt_tokenizer: spm.SentencePieceProcessor,
    device: torch.device,
    beam_size: int = 4,
    max_len: int = 250
) -> float:
    """Evaluate model and compute chrF score."""
    model.eval()
    hypotheses = []
    references = []
    
    bos_id = 1
    eos_id = 2
    pad_id = 0
    
    with torch.no_grad():
        for src, tgt, src_mask, tgt_mask in tqdm(data_loader, desc='Evaluating'):
            src = src.to(device)
            src_mask = src_mask.to(device)
            
            # Translate
            batch_hypotheses = translate_batch(
                model, src, src_mask, tgt_tokenizer, device,
                beam_size=beam_size, max_len=max_len, bos_id=bos_id, eos_id=eos_id
            )
            
            # Decode references
            for i in range(tgt.size(1)):
                tgt_ids = tgt[:, i].cpu().tolist()
                tgt_ids = [tid for tid in tgt_ids if tid not in [pad_id, bos_id, eos_id]]
                ref_text = tgt_tokenizer.decode(tgt_ids)
                references.append(ref_text)
            
            hypotheses.extend(batch_hypotheses)
    
    # Compute chrF
    chrf = CHRF()
    chrf_score = chrf.corpus_score(hypotheses, [references]).score
    
    return chrf_score


def evaluate(
    checkpoint_path: str,
    config_path: str = None,
    test_src: str = None,
    test_tgt: str = None,
    output_file: str = None
):
    """Main evaluation function."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config')
    
    if config_path:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    
    if config is None:
        raise ValueError("Config not found in checkpoint and config_path not provided")
    
    # Load tokenizers
    tokenizer_dir = config['data']['tokenizer_dir']
    if config['data']['joint_tokenizer']:
        tokenizer_path = os.path.join(tokenizer_dir, 'tokenizer.model')
        src_tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_path)
        tgt_tokenizer = src_tokenizer
        vocab_size = len(src_tokenizer)
    else:
        src_tokenizer_path = os.path.join(tokenizer_dir, 'src_tokenizer.model')
        tgt_tokenizer_path = os.path.join(tokenizer_dir, 'tgt_tokenizer.model')
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
    
    print(f"Model loaded from {checkpoint_path}")
    
    # Create data loader
    if test_src and test_tgt:
        test_loader = create_dataloader(
            src_file=test_src,
            tgt_file=test_tgt,
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer,
            batch_size=config['training']['batch_size'],
            max_len=config['data']['max_len'],
            shuffle=False
        )
    else:
        test_loader = create_dataloader(
            src_file=config['data']['test_src'],
            tgt_file=config['data']['test_tgt'],
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer,
            batch_size=config['training']['batch_size'],
            max_len=config['data']['max_len'],
            shuffle=False
        )
    
    # Evaluate
    print("Computing BLEU score...")
    bleu_score = evaluate_bleu(model, test_loader, tgt_tokenizer, device)
    
    print("Computing chrF score...")
    chrf_score = evaluate_chrf(model, test_loader, tgt_tokenizer, device)
    
    print(f"\nResults:")
    print(f"BLEU: {bleu_score:.2f}")
    print(f"chrF: {chrf_score:.2f}")
    
    # Save results
    if output_file:
        with open(output_file, 'w') as f:
            f.write(f"BLEU: {bleu_score:.2f}\n")
            f.write(f"chrF: {chrf_score:.2f}\n")
        print(f"Results saved to {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate NMT model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None, help='Path to config YAML file')
    parser.add_argument('--test_src', type=str, default=None, help='Test source file')
    parser.add_argument('--test_tgt', type=str, default=None, help='Test target file')
    parser.add_argument('--output', type=str, default=None, help='Output file for results')
    
    args = parser.parse_args()
    evaluate(args.checkpoint, args.config, args.test_src, args.test_tgt, args.output)

