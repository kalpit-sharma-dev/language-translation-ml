"""
Data loading and preprocessing utilities.
"""
import os
import re
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional
import sentencepiece as spm


class TranslationDataset(Dataset):
    """Dataset for parallel translation pairs."""
    
    def __init__(
        self,
        src_file: str,
        tgt_file: str,
        src_tokenizer: spm.SentencePieceProcessor,
        tgt_tokenizer: spm.SentencePieceProcessor,
        max_len: int = 250,
        bos_id: int = 1,
        eos_id: int = 2,
        pad_id: int = 0
    ):
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_len = max_len
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        
        # Load parallel sentences
        self.src_sentences = self._load_file(src_file)
        self.tgt_sentences = self._load_file(tgt_file)
        
        # Filter by length
        self._filter_by_length()
        
        print(f"Loaded {len(self.src_sentences)} sentence pairs")
    
    def _load_file(self, filepath: str) -> List[str]:
        """Load sentences from file."""
        sentences = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    sentences.append(line)
        return sentences
    
    def _filter_by_length(self):
        """Filter sentences that are too long."""
        filtered_src = []
        filtered_tgt = []
        
        for src, tgt in zip(self.src_sentences, self.tgt_sentences):
            src_ids = self.src_tokenizer.encode(src, out_type=int)
            tgt_ids = self.tgt_tokenizer.encode(tgt, out_type=int)
            
            if len(src_ids) <= self.max_len and len(tgt_ids) <= self.max_len:
                filtered_src.append(src)
                filtered_tgt.append(tgt)
        
        self.src_sentences = filtered_src
        self.tgt_sentences = filtered_tgt
        print(f"After filtering: {len(self.src_sentences)} sentence pairs")
    
    def __len__(self) -> int:
        return len(self.src_sentences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single translation pair."""
        src_text = self.src_sentences[idx]
        tgt_text = self.tgt_sentences[idx]
        
        # Tokenize
        src_ids = [self.bos_id] + self.src_tokenizer.encode(src_text, out_type=int) + [self.eos_id]
        tgt_ids = [self.bos_id] + self.tgt_tokenizer.encode(tgt_text, out_type=int) + [self.eos_id]
        
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]], pad_id: int = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function for DataLoader.
    
    Returns:
        src: [src_len, batch_size]
        tgt: [tgt_len, batch_size]
        src_mask: [batch_size, src_len]
        tgt_mask: [batch_size, tgt_len]
    """
    src_batch, tgt_batch = zip(*batch)
    
    # Pad sequences
    src_len = max(len(s) for s in src_batch)
    tgt_len = max(len(t) for t in tgt_batch)
    
    src_padded = []
    tgt_padded = []
    
    for src, tgt in zip(src_batch, tgt_batch):
        src_pad = torch.cat([src, torch.full((src_len - len(src),), pad_id, dtype=torch.long)])
        tgt_pad = torch.cat([tgt, torch.full((tgt_len - len(tgt),), pad_id, dtype=torch.long)])
        src_padded.append(src_pad)
        tgt_padded.append(tgt_pad)
    
    src_tensor = torch.stack(src_padded).transpose(0, 1)  # [src_len, batch_size]
    tgt_tensor = torch.stack(tgt_padded).transpose(0, 1)  # [tgt_len, batch_size]
    
    # Create padding masks (True for padding tokens)
    src_mask = (src_tensor == pad_id).transpose(0, 1)  # [batch_size, src_len]
    tgt_mask = (tgt_tensor == pad_id).transpose(0, 1)  # [batch_size, tgt_len]
    
    return src_tensor, tgt_tensor, src_mask, tgt_mask


def create_dataloader(
    src_file: str,
    tgt_file: str,
    src_tokenizer: spm.SentencePieceProcessor,
    tgt_tokenizer: spm.SentencePieceProcessor,
    batch_size: int = 32,
    max_len: int = 250,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """Create a DataLoader for translation data."""
    dataset = TranslationDataset(src_file, tgt_file, src_tokenizer, tgt_tokenizer, max_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda b: collate_fn(b, pad_id=0),
        num_workers=num_workers
    )


def preprocess_text(text: str, lowercase: bool = False) -> str:
    """Preprocess text: normalize whitespace, punctuation."""
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    if lowercase:
        text = text.lower()
    
    return text


def prepare_data_files(src_file: str, tgt_file: str, output_dir: str, lowercase: bool = False):
    """Preprocess and save cleaned data files."""
    os.makedirs(output_dir, exist_ok=True)
    
    src_output = os.path.join(output_dir, 'src_cleaned.txt')
    tgt_output = os.path.join(output_dir, 'tgt_cleaned.txt')
    
    with open(src_file, 'r', encoding='utf-8') as f_src, \
         open(tgt_file, 'r', encoding='utf-8') as f_tgt, \
         open(src_output, 'w', encoding='utf-8') as out_src, \
         open(tgt_output, 'w', encoding='utf-8') as out_tgt:
        
        for src_line, tgt_line in zip(f_src, f_tgt):
            src_cleaned = preprocess_text(src_line, lowercase)
            tgt_cleaned = preprocess_text(tgt_line, lowercase)
            
            if src_cleaned and tgt_cleaned:
                out_src.write(src_cleaned + '\n')
                out_tgt.write(tgt_cleaned + '\n')
    
    print(f"Cleaned data saved to {output_dir}")
    return src_output, tgt_output

