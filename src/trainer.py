"""
Training script for NMT model.
"""
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml
import math
import sentencepiece as spm

from .model import SimpleTransformerNMT
from .data_loader import create_dataloader


class LabelSmoothingLoss(nn.Module):
    """Label smoothing loss."""
    
    def __init__(self, vocab_size: int, padding_idx: int, smoothing: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [tgt_len, batch_size, vocab_size]
            target: [tgt_len, batch_size]
        """
        pred = pred.reshape(-1, self.vocab_size)
        target = target.reshape(-1)
        
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (self.vocab_size - 2))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = (target == self.padding_idx)
        true_dist[mask] = 0
        
        return nn.functional.kl_div(
            nn.functional.log_softmax(pred, dim=1),
            true_dist,
            reduction='sum'
        ) / (target != self.padding_idx).sum().float()


class WarmupInverseSqrtScheduler:
    """Learning rate scheduler with warmup and inverse sqrt decay."""
    
    def __init__(self, optimizer, d_model: int, warmup_steps: int = 4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.current_step = 0
    
    def step(self):
        """Update learning rate."""
        self.current_step += 1
        lr = self.d_model ** (-0.5) * min(
            self.current_step ** (-0.5),
            self.current_step * self.warmup_steps ** (-1.5)
        )
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr
    
    def get_lr(self):
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']


def train_epoch(
    model: nn.Module,
    train_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    clip_grad: float = 1.0
):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for src, tgt, src_mask, tgt_mask in pbar:
        src = src.to(device)
        tgt = tgt.to(device)
        src_mask = src_mask.to(device)
        tgt_mask = tgt_mask.to(device)
        
        # Prepare input and target for teacher forcing
        tgt_input = tgt[:-1, :]  # Remove last token
        tgt_output = tgt[1:, :]  # Remove first token (BOS)
        
        # Adjust masks: tgt_mask is [batch_size, tgt_len], need [batch_size, tgt_len-1]
        tgt_input_mask = tgt_mask[:, :-1]  # Remove last column to match tgt_input length
        
        # Forward pass
        optimizer.zero_grad()
        output = model(src, tgt_input, src_key_padding_mask=src_mask, tgt_key_padding_mask=tgt_input_mask)
        
        # Compute loss
        loss = criterion(output, tgt_output)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': loss.item(), 'lr': scheduler.get_lr()})
    
    return total_loss / num_batches


def train(
    config_path: str,
    resume_from: str = None
):
    """Main training function."""
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
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
        vocab_size = len(tgt_tokenizer)  # Use target vocab size
    
    print(f"Vocabulary size: {vocab_size}")
    
    # Create data loaders
    train_loader = create_dataloader(
        src_file=config['data']['train_src'],
        tgt_file=config['data']['train_tgt'],
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        batch_size=config['training']['batch_size'],
        max_len=config['data']['max_len'],
        shuffle=True
    )
    
    dev_loader = create_dataloader(
        src_file=config['data']['dev_src'],
        tgt_file=config['data']['dev_tgt'],
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        batch_size=config['training']['batch_size'],
        max_len=config['data']['max_len'],
        shuffle=False
    )
    
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
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    if config['training']['label_smoothing'] > 0:
        criterion = LabelSmoothingLoss(vocab_size, padding_idx=0, smoothing=config['training']['label_smoothing'])
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # Ensure learning rate is a float
    learning_rate = float(config['training']['lr'])
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.98),
        eps=1e-9
    )
    
    scheduler = WarmupInverseSqrtScheduler(
        optimizer,
        d_model=config['model']['d_model'],
        warmup_steps=config['training']['warmup_steps']
    )
    
    # Resume from checkpoint if provided
    start_epoch = 0
    best_bleu = 0.0
    if resume_from and os.path.exists(resume_from):
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_bleu = checkpoint.get('best_bleu', 0.0)
        print(f"Resumed from epoch {start_epoch}, best BLEU: {best_bleu:.2f}")
    
    # TensorBoard writer
    log_dir = config.get('log_dir', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    
    # Checkpoint directory
    checkpoint_dir = config.get('checkpoint_dir', 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training loop
    num_epochs = config['training']['num_epochs']
    eval_every = config['training'].get('eval_every', 1)
    
    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, device,
                                clip_grad=config['training'].get('clip_grad', 1.0))
        
        writer.add_scalar('Train/Loss', train_loss, epoch)
        writer.add_scalar('Train/LearningRate', scheduler.get_lr(), epoch)
        
        print(f"Train Loss: {train_loss:.4f}")
        
        # Evaluate
        if (epoch + 1) % eval_every == 0:
            from .evaluate import evaluate_bleu
            bleu_score = evaluate_bleu(model, dev_loader, tgt_tokenizer, device)
            writer.add_scalar('Dev/BLEU', bleu_score, epoch)
            print(f"Dev BLEU: {bleu_score:.2f}")
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_bleu': max(best_bleu, bleu_score),
                'config': config
            }
            
            # Save latest
            torch.save(checkpoint, os.path.join(checkpoint_dir, 'latest.pt'))
            
            # Save best
            if bleu_score > best_bleu:
                best_bleu = bleu_score
                torch.save(checkpoint, os.path.join(checkpoint_dir, 'best.pt'))
                print(f"New best BLEU: {best_bleu:.2f}, checkpoint saved!")
    
    writer.close()
    print("Training completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train NMT model')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    train(args.config, args.resume)

