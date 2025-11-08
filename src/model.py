"""
Transformer-based Neural Machine Translation Model.
"""
import torch
import torch.nn as nn
import math
from .positional_encoding import PositionalEncoding


class SimpleTransformerNMT(nn.Module):
    """Simple Transformer model for Neural Machine Translation."""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_len: int = 5000,
        pad_idx: int = 0
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx
        
        # Embedding layers
        self.tok_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False  # Use (seq_len, batch, features) format
        )
        
        # Output projection
        self.generator = nn.Linear(d_model, vocab_size)
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize parameters."""
        initrange = 0.1
        self.tok_embed.weight.data.uniform_(-initrange, initrange)
        self.generator.bias.data.zero_()
        self.generator.weight.data.uniform_(-initrange, initrange)
    
    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate a square mask for the sequence."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None,
        src_key_padding_mask: torch.Tensor = None,
        tgt_key_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            src: Source sequence [src_len, batch_size]
            tgt: Target sequence [tgt_len, batch_size]
            src_mask: Source attention mask
            tgt_mask: Target attention mask
            src_key_padding_mask: Source padding mask
            tgt_key_padding_mask: Target padding mask
        
        Returns:
            Output logits [tgt_len, batch_size, vocab_size]
        """
        # Embeddings
        src_emb = self.pos_encoder(self.tok_embed(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.tok_embed(tgt) * math.sqrt(self.d_model))
        
        # Generate masks if not provided
        if tgt_mask is None and tgt.size(0) > 0:
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(0)).to(tgt.device)
        
        # Transformer forward
        out = self.transformer(
            src_emb,
            tgt_emb,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        # Generate logits
        return self.generator(out)
    
    def encode(self, src: torch.Tensor, src_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """Encode source sequence."""
        src_emb = self.pos_encoder(self.tok_embed(src) * math.sqrt(self.d_model))
        return self.transformer.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
    
    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor = None,
        tgt_key_padding_mask: torch.Tensor = None,
        memory_key_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """Decode target sequence."""
        tgt_emb = self.pos_encoder(self.tok_embed(tgt) * math.sqrt(self.d_model))
        if tgt_mask is None and tgt.size(0) > 0:
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(0)).to(tgt.device)
        return self.transformer.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )

