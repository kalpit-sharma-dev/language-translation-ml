"""
Inference script for translation.
"""
import argparse
import torch
import torch.nn as nn
import sentencepiece as spm
import yaml
import os
import math

from .model import SimpleTransformerNMT


def translate_batch(
    model: nn.Module,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    tgt_tokenizer: spm.SentencePieceProcessor,
    device: torch.device,
    beam_size: int = 4,
    max_len: int = 250,
    length_penalty: float = 0.6,
    bos_id: int = 1,
    eos_id: int = 2,
    pad_id: int = 0
) -> list:
    """
    Translate a batch of source sequences using beam search.
    
    Args:
        model: Trained NMT model
        src: Source sequences [src_len, batch_size]
        src_mask: Source padding mask [batch_size, src_len]
        tgt_tokenizer: Target tokenizer
        device: Device
        beam_size: Beam search width
        max_len: Maximum target length
        length_penalty: Length penalty for beam search
        bos_id: Beginning of sentence token ID
        eos_id: End of sentence token ID
        pad_id: Padding token ID
    
    Returns:
        List of translated sentences
    """
    model.eval()
    batch_size = src.size(1)
    hypotheses = []
    
    with torch.no_grad():
        # Encode source
        src_emb = model.pos_encoder(model.tok_embed(src) * math.sqrt(model.d_model))
        memory = model.transformer.encoder(src_emb, src_key_padding_mask=src_mask)
        
        for i in range(batch_size):
            src_seq = src[:, i:i+1]
            src_mask_seq = src_mask[i:i+1, :]
            memory_seq = memory[:, i:i+1, :]
            
            # Beam search
            translation = beam_search(
                model, src_seq, src_mask_seq, memory_seq, tgt_tokenizer, device,
                beam_size, max_len, length_penalty, bos_id, eos_id, pad_id
            )
            hypotheses.append(translation)
    
    return hypotheses


def beam_search(
    model: nn.Module,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    memory: torch.Tensor,
    tgt_tokenizer: spm.SentencePieceProcessor,
    device: torch.device,
    beam_size: int,
    max_len: int,
    length_penalty: float,
    bos_id: int,
    eos_id: int,
    pad_id: int
) -> str:
    """Beam search decoding."""
    # Initialize beam
    beams = [([bos_id], 0.0)]  # (sequence, score)
    finished = []
    
    for step in range(max_len):
        candidates = []
        
        for seq, score in beams:
            if seq[-1] == eos_id:
                finished.append((seq, score / (len(seq) ** length_penalty)))
                continue
            
            # Prepare input
            tgt_input = torch.tensor([seq], dtype=torch.long).to(device).transpose(0, 1)
            tgt_mask = model.generate_square_subsequent_mask(len(seq)).to(device)
            
            # Decode
            tgt_emb = model.pos_encoder(model.tok_embed(tgt_input) * math.sqrt(model.d_model))
            output = model.transformer.decoder(
                tgt_emb,
                memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_mask
            )
            logits = model.generator(output[-1, 0, :])  # [vocab_size]
            log_probs = torch.log_softmax(logits, dim=0)
            
            # Get top k candidates
            top_k_probs, top_k_indices = torch.topk(log_probs, beam_size)
            
            for prob, idx in zip(top_k_probs, top_k_indices):
                new_seq = seq + [idx.item()]
                new_score = score + prob.item()
                candidates.append((new_seq, new_score))
        
        # Keep top beam_size candidates
        candidates.sort(key=lambda x: x[1], reverse=True)
        beams = candidates[:beam_size]
        
        if not beams:
            break
    
    # Add unfinished beams
    for seq, score in beams:
        finished.append((seq, score / (len(seq) ** length_penalty)))
    
    # Get best translation
    if finished:
        finished.sort(key=lambda x: x[1], reverse=True)
        best_seq = finished[0][0]
    else:
        best_seq = [bos_id, eos_id]
    
    # Decode
    best_seq = [tid for tid in best_seq if tid not in [pad_id, bos_id, eos_id]]
    translation = tgt_tokenizer.decode(best_seq)
    
    return translation


def translate_sentence(
    model: nn.Module,
    sentence: str,
    src_tokenizer: spm.SentencePieceProcessor,
    tgt_tokenizer: spm.SentencePieceProcessor,
    device: torch.device,
    beam_size: int = 4,
    max_len: int = 250
) -> str:
    """Translate a single sentence."""
    # Tokenize source
    src_ids = [1] + src_tokenizer.encode(sentence, out_type=int) + [2]
    src = torch.tensor([src_ids], dtype=torch.long).to(device).transpose(0, 1)
    src_mask = torch.zeros(1, src.size(0), dtype=torch.bool).to(device)
    
    # Translate
    translation = translate_batch(
        model, src, src_mask, tgt_tokenizer, device,
        beam_size=beam_size, max_len=max_len
    )[0]
    
    return translation


def translate_file(
    checkpoint_path: str,
    input_file: str,
    output_file: str,
    config_path: str = None,
    beam_size: int = 4
):
    """Translate sentences from a file."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config')
    
    if config_path:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    
    if config is None:
        raise ValueError("Config not found")
    
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
    
    # Translate
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            sentence = line.strip()
            if sentence:
                translation = translate_sentence(
                    model, sentence, src_tokenizer, tgt_tokenizer, device, beam_size
                )
                f_out.write(translation + '\n')
            else:
                f_out.write('\n')
    
    print(f"Translations saved to {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Translate using trained NMT model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True, help='Input file or sentence')
    parser.add_argument('--output', type=str, default=None, help='Output file (if input is file)')
    parser.add_argument('--config', type=str, default=None, help='Path to config YAML file')
    parser.add_argument('--beam_size', type=int, default=4, help='Beam search width')
    
    args = parser.parse_args()
    
    if os.path.isfile(args.input):
        translate_file(args.checkpoint, args.input, args.output or 'output.txt', args.config, args.beam_size)
    else:
        # Single sentence translation
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(args.checkpoint, map_location=device)
        config = checkpoint.get('config')
        
        if args.config:
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
        
        tokenizer_dir = config['data']['tokenizer_dir']
        if config['data']['joint_tokenizer']:
            tokenizer_path = os.path.join(tokenizer_dir, 'tokenizer.model')
            src_tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_path)
            tgt_tokenizer = src_tokenizer
        else:
            src_tokenizer_path = os.path.join(tokenizer_dir, 'src_tokenizer.model')
            tgt_tokenizer_path = os.path.join(tokenizer_dir, 'tgt_tokenizer.model')
            src_tokenizer = spm.SentencePieceProcessor(model_file=src_tokenizer_path)
            tgt_tokenizer = spm.SentencePieceProcessor(model_file=tgt_tokenizer_path)
        
        vocab_size = len(tgt_tokenizer)
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
        
        translation = translate_sentence(model, args.input, src_tokenizer, tgt_tokenizer, device, args.beam_size)
        print(f"Translation: {translation}")

