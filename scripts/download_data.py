"""
Script to download and prepare WMT English-German dataset.
This script provides instructions and can download from Hugging Face datasets.
"""
import argparse
import os
from datasets import load_dataset
from tqdm import tqdm


def download_wmt_dataset(output_dir: str = 'data/raw', split_size: int = None):
    """
    Download WMT English-German dataset from Hugging Face.
    
    Args:
        output_dir: Directory to save the data
        split_size: Optional limit on number of examples to download
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("Downloading WMT English-German dataset...")
    print("This may take a while...")
    
    try:
        # Try to load WMT dataset
        dataset = load_dataset("wmt14", "de-en", split="train")
        
        print(f"Dataset loaded. Total examples: {len(dataset)}")
        
        if split_size:
            dataset = dataset.select(range(min(split_size, len(dataset))))
            print(f"Using {len(dataset)} examples")
        
        # Save to files
        src_file = os.path.join(output_dir, 'train.en')
        tgt_file = os.path.join(output_dir, 'train.de')
        
        with open(src_file, 'w', encoding='utf-8') as f_src, \
             open(tgt_file, 'w', encoding='utf-8') as f_tgt:
            for example in tqdm(dataset, desc="Saving"):
                f_src.write(example['translation']['en'] + '\n')
                f_tgt.write(example['translation']['de'] + '\n')
        
        print(f"Data saved to {src_file} and {tgt_file}")
        
    except Exception as e:
        print(f"Error downloading from Hugging Face: {e}")
        print("\nAlternative: Download manually from:")
        print("1. http://www.statmt.org/wmt14/translation-task.html")
        print("2. https://huggingface.co/datasets/wmt14")
        print("\nOr use a smaller dataset for testing:")
        print("  python scripts/download_data.py --use_sample")


def create_sample_dataset(output_dir: str = 'data/raw', num_examples: int = 1000):
    """Create a small sample dataset for testing."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample English-German pairs for testing
    sample_pairs = [
        ("Hello, how are you?", "Hallo, wie geht es dir?"),
        ("The weather is nice today.", "Das Wetter ist heute schön."),
        ("I love learning new languages.", "Ich liebe es, neue Sprachen zu lernen."),
        ("Can you help me with this problem?", "Kannst du mir bei diesem Problem helfen?"),
        ("Machine learning is fascinating.", "Maschinelles Lernen ist faszinierend."),
        ("The cat is sleeping on the sofa.", "Die Katze schläft auf dem Sofa."),
        ("We are going to the park tomorrow.", "Wir gehen morgen in den Park."),
        ("This is a beautiful day.", "Das ist ein schöner Tag."),
        ("I need to buy some groceries.", "Ich muss Lebensmittel einkaufen."),
        ("The book is on the table.", "Das Buch liegt auf dem Tisch."),
    ]
    
    # Repeat to create more examples
    src_file = os.path.join(output_dir, 'train.en')
    tgt_file = os.path.join(output_dir, 'train.de')
    
    with open(src_file, 'w', encoding='utf-8') as f_src, \
         open(tgt_file, 'w', encoding='utf-8') as f_tgt:
        for i in range(num_examples):
            pair = sample_pairs[i % len(sample_pairs)]
            f_src.write(pair[0] + '\n')
            f_tgt.write(pair[1] + '\n')
    
    print(f"Sample dataset created with {num_examples} examples")
    print(f"Files: {src_file}, {tgt_file}")


def split_data(src_file: str, tgt_file: str, output_dir: str, train_ratio: float = 0.98, dev_ratio: float = 0.01):
    """
    Split data into train/dev/test sets.
    
    Args:
        src_file: Source language file
        tgt_file: Target language file
        output_dir: Output directory
        train_ratio: Ratio for training set
        dev_ratio: Ratio for dev set (test = 1 - train - dev)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Read all lines
    with open(src_file, 'r', encoding='utf-8') as f:
        src_lines = f.readlines()
    with open(tgt_file, 'r', encoding='utf-8') as f:
        tgt_lines = f.readlines()
    
    assert len(src_lines) == len(tgt_lines), "Source and target files must have same number of lines"
    
    total = len(src_lines)
    train_end = int(total * train_ratio)
    dev_end = train_end + int(total * dev_ratio)
    
    # Split
    train_src = src_lines[:train_end]
    train_tgt = tgt_lines[:train_end]
    dev_src = src_lines[train_end:dev_end]
    dev_tgt = tgt_lines[train_end:dev_end]
    test_src = src_lines[dev_end:]
    test_tgt = tgt_lines[dev_end:]
    
    # Write splits
    for split_name, src_data, tgt_data in [
        ('train', train_src, train_tgt),
        ('dev', dev_src, dev_tgt),
        ('test', test_src, test_tgt)
    ]:
        with open(os.path.join(output_dir, f'{split_name}.en'), 'w', encoding='utf-8') as f:
            f.writelines(src_data)
        with open(os.path.join(output_dir, f'{split_name}.de'), 'w', encoding='utf-8') as f:
            f.writelines(tgt_data)
        print(f"{split_name}: {len(src_data)} examples")
    
    print(f"\nData splits saved to {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download and prepare WMT dataset')
    parser.add_argument('--output_dir', type=str, default='data/raw', help='Output directory')
    parser.add_argument('--use_sample', action='store_true', help='Create sample dataset')
    parser.add_argument('--sample_size', type=int, default=1000, help='Size of sample dataset')
    parser.add_argument('--split', action='store_true', help='Split data into train/dev/test')
    parser.add_argument('--split_dir', type=str, default='data/cleaned', help='Directory for split data')
    
    args = parser.parse_args()
    
    if args.use_sample:
        create_sample_dataset(args.output_dir, args.sample_size)
    else:
        download_wmt_dataset(args.output_dir)
    
    if args.split:
        src_file = os.path.join(args.output_dir, 'train.en')
        tgt_file = os.path.join(args.output_dir, 'train.de')
        if os.path.exists(src_file) and os.path.exists(tgt_file):
            split_data(src_file, tgt_file, args.split_dir)
        else:
            print("Error: Source files not found. Run download first.")

