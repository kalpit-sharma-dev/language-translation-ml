"""
Train SentencePiece tokenizer for NMT.
"""
import argparse
import sentencepiece as spm
import os
from typing import Optional


def train_tokenizer(
    input_files: list,
    model_prefix: str,
    vocab_size: int = 32000,
    model_type: str = 'bpe',
    character_coverage: float = 0.9995,
    input_sentence_size: int = 1000000,
    shuffle_input_sentence: bool = True
):
    """
    Train a SentencePiece tokenizer.
    
    Args:
        input_files: List of input text files
        model_prefix: Prefix for output model files
        vocab_size: Vocabulary size
        model_type: 'bpe', 'unigram', 'char', or 'word'
        character_coverage: Character coverage (0.0-1.0)
        input_sentence_size: Maximum number of sentences to use
        shuffle_input_sentence: Whether to shuffle input sentences
    """
    # Combine input files into a single string
    input_str = ','.join(input_files)
    
    # Estimate maximum possible vocab size from data
    # Count unique characters and words to estimate upper bound
    try:
        import os
        total_chars = 0
        total_words = 0
        for file_path in input_files:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        total_chars += len(line)
                        total_words += len(line.split())
        
        # Rough estimate: vocab size should be less than unique subwords possible
        # For small datasets, use a more conservative estimate
        max_reasonable_vocab = min(vocab_size, max(500, total_words // 10))
        
        if vocab_size > max_reasonable_vocab:
            print(f"Warning: Requested vocab_size ({vocab_size}) may be too large for the dataset.")
            print(f"Adjusting to {max_reasonable_vocab} based on data size.")
            vocab_size = max_reasonable_vocab
    except Exception as e:
        print(f"Could not estimate optimal vocab size: {e}")
        # If dataset is very small, use a smaller default
        if vocab_size > 5000:
            print(f"Warning: Large vocab_size ({vocab_size}) for potentially small dataset.")
            print("Consider using a smaller vocab_size (e.g., 1000-5000) for small datasets.")
    
    # SentencePiece training parameters
    try:
        spm.SentencePieceTrainer.train(
            input=input_str,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type=model_type,
            character_coverage=character_coverage,
            input_sentence_size=input_sentence_size,
            shuffle_input_sentence=shuffle_input_sentence,
            normalization_rule_name='nmt_nfkc_cf',  # Normalization for NMT
            pad_id=0,
            bos_id=1,
            eos_id=2,
            unk_id=3,
            user_defined_symbols=['<mask>']
        )
        
        print(f"Tokenizer trained successfully!")
        print(f"Model files: {model_prefix}.model, {model_prefix}.vocab")
    except RuntimeError as e:
        error_msg = str(e)
        if "Vocabulary size too high" in error_msg:
            # Extract suggested vocab size from error message
            import re
            match = re.search(r'value <= (\d+)', error_msg)
            if match:
                suggested_size = int(match.group(1))
                print(f"\nError: Vocabulary size {vocab_size} is too large for the dataset.")
                print(f"Suggested maximum vocabulary size: {suggested_size}")
                print(f"\nRetrying with vocabulary size: {suggested_size}")
                
                # Retry with suggested size
                spm.SentencePieceTrainer.train(
                    input=input_str,
                    model_prefix=model_prefix,
                    vocab_size=suggested_size,
                    model_type=model_type,
                    character_coverage=character_coverage,
                    input_sentence_size=input_sentence_size,
                    shuffle_input_sentence=shuffle_input_sentence,
                    normalization_rule_name='nmt_nfkc_cf',
                    pad_id=0,
                    bos_id=1,
                    eos_id=2,
                    unk_id=3,
                    user_defined_symbols=['<mask>']
                )
                
                print(f"Tokenizer trained successfully with vocab_size={suggested_size}!")
                print(f"Model files: {model_prefix}.model, {model_prefix}.vocab")
            else:
                raise
        else:
            raise


def train_joint_tokenizer(
    src_file: str,
    tgt_file: str,
    output_prefix: str,
    vocab_size: int = 32000,
    model_type: str = 'bpe'
):
    """Train a joint tokenizer for source and target languages."""
    train_tokenizer(
        input_files=[src_file, tgt_file],
        model_prefix=output_prefix,
        vocab_size=vocab_size,
        model_type=model_type
    )


def train_separate_tokenizers(
    src_file: str,
    tgt_file: str,
    src_output_prefix: str,
    tgt_output_prefix: str,
    vocab_size: int = 32000,
    model_type: str = 'bpe'
):
    """Train separate tokenizers for source and target languages."""
    print("Training source tokenizer...")
    train_tokenizer(
        input_files=[src_file],
        model_prefix=src_output_prefix,
        vocab_size=vocab_size,
        model_type=model_type
    )
    
    print("Training target tokenizer...")
    train_tokenizer(
        input_files=[tgt_file],
        model_prefix=tgt_output_prefix,
        vocab_size=vocab_size,
        model_type=model_type
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SentencePiece tokenizer')
    parser.add_argument('--src_file', type=str, required=True, help='Source language file')
    parser.add_argument('--tgt_file', type=str, required=True, help='Target language file')
    parser.add_argument('--output_dir', type=str, default='data/tokenized', help='Output directory')
    parser.add_argument('--vocab_size', type=int, default=32000, help='Vocabulary size')
    parser.add_argument('--model_type', type=str, default='bpe', choices=['bpe', 'unigram', 'char', 'word'],
                       help='SentencePiece model type')
    parser.add_argument('--joint', action='store_true', help='Train joint tokenizer (shared vocab)')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.joint:
        output_prefix = os.path.join(args.output_dir, 'tokenizer')
        print("Training joint tokenizer...")
        train_joint_tokenizer(
            args.src_file,
            args.tgt_file,
            output_prefix,
            args.vocab_size,
            args.model_type
        )
    else:
        src_prefix = os.path.join(args.output_dir, 'src_tokenizer')
        tgt_prefix = os.path.join(args.output_dir, 'tgt_tokenizer')
        print("Training separate tokenizers...")
        train_separate_tokenizers(
            args.src_file,
            args.tgt_file,
            src_prefix,
            tgt_prefix,
            args.vocab_size,
            args.model_type
        )

