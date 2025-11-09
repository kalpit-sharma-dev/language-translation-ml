"""
Script to download and prepare Indian language datasets.
Supports multiple Indian languages for translation.
"""
import argparse
import os
from datasets import load_dataset
from tqdm import tqdm


# Language code mappings
LANGUAGE_CODES = {
    'hindi': 'hi',
    'bengali': 'bn',
    'telugu': 'te',
    'tamil': 'ta',
    'gujarati': 'gu',
    'kannada': 'kn',
    'malayalam': 'ml',
    'marathi': 'mr',
    'punjabi': 'pa',
    'urdu': 'ur',
    'odia': 'or',
    'assamese': 'as'
}

# Dataset mappings for Hugging Face
DATASET_MAPPINGS = {
    'hi': 'cfilt/iitb-english-hindi',  # IIT Bombay English-Hindi
    'bn': 'cfilt/iitb-english-bengali',
    'te': 'cfilt/iitb-english-telugu',
    'ta': 'cfilt/iitb-english-tamil',
    'gu': 'cfilt/iitb-english-gujarati',
    'kn': 'cfilt/iitb-english-kannada',
    'ml': 'cfilt/iitb-english-malayalam',
    'mr': 'cfilt/iitb-english-marathi',
    'pa': 'cfilt/iitb-english-punjabi',
    'ur': 'cfilt/iitb-english-urdu',
}


def download_indian_language_dataset(
    target_lang: str,
    output_dir: str = 'data/raw',
    split_size: int = None,
    use_sample: bool = False
):
    """
    Download Indian language dataset from Hugging Face.
    
    Args:
        target_lang: Target language code (hi, bn, te, etc.)
        output_dir: Directory to save the data
        split_size: Optional limit on number of examples
        use_sample: Create sample dataset for testing
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if use_sample:
        create_sample_indian_dataset(target_lang, output_dir, split_size or 1000)
        return
    
    lang_code = target_lang.lower()
    if lang_code not in DATASET_MAPPINGS:
        print(f"Warning: Dataset not found for {target_lang}")
        print(f"Available languages: {list(DATASET_MAPPINGS.keys())}")
        print("Creating sample dataset instead...")
        create_sample_indian_dataset(target_lang, output_dir, split_size or 1000)
        return
    
    dataset_name = DATASET_MAPPINGS[lang_code]
    
    print(f"Downloading {target_lang.upper()} dataset from {dataset_name}...")
    print("This may take a while...")
    
    try:
        dataset = load_dataset(dataset_name, split="train")
        
        print(f"Dataset loaded. Total examples: {len(dataset)}")
        
        if split_size:
            dataset = dataset.select(range(min(split_size, len(dataset))))
            print(f"Using {len(dataset)} examples")
        
        # Save to files
        src_file = os.path.join(output_dir, 'train.en')
        tgt_file = os.path.join(output_dir, f'train.{lang_code}')
        
        with open(src_file, 'w', encoding='utf-8') as f_src, \
             open(tgt_file, 'w', encoding='utf-8') as f_tgt:
            for example in tqdm(dataset, desc="Saving"):
                # Handle different dataset formats
                if 'translation' in example:
                    en_text = example['translation'].get('en', '')
                    tgt_text = example['translation'].get(lang_code, '')
                elif 'en' in example and lang_code in example:
                    en_text = example['en']
                    tgt_text = example[lang_code]
                else:
                    # Try to find English and target language columns
                    keys = list(example.keys())
                    if len(keys) >= 2:
                        en_text = example[keys[0]]
                        tgt_text = example[keys[1]]
                    else:
                        continue
                
                if en_text and tgt_text:
                    f_src.write(en_text.strip() + '\n')
                    f_tgt.write(tgt_text.strip() + '\n')
        
        print(f"Data saved to {src_file} and {tgt_file}")
        
    except Exception as e:
        print(f"Error downloading from Hugging Face: {e}")
        print(f"\nCreating sample dataset for {target_lang} instead...")
        create_sample_indian_dataset(target_lang, output_dir, split_size or 1000)


def create_sample_indian_dataset(target_lang: str, output_dir: str, num_examples: int = 1000):
    """Create a small sample dataset for testing with Indian languages."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample translations for different Indian languages
    sample_pairs = {
        'hi': [
            ("Hello, how are you?", "नमस्ते, आप कैसे हैं?"),
            ("The weather is nice today.", "आज मौसम अच्छा है।"),
            ("I love learning new languages.", "मुझे नई भाषाएं सीखना पसंद है।"),
            ("Can you help me with this problem?", "क्या आप मेरी इस समस्या में मदद कर सकते हैं?"),
            ("Machine learning is fascinating.", "मशीन लर्निंग आकर्षक है।"),
        ],
        'bn': [
            ("Hello, how are you?", "হ্যালো, আপনি কেমন আছেন?"),
            ("The weather is nice today.", "আজ আবহাওয়া ভালো।"),
            ("I love learning new languages.", "আমি নতুন ভাষা শিখতে ভালোবাসি।"),
        ],
        'te': [
            ("Hello, how are you?", "హలో, మీరు ఎలా ఉన్నారు?"),
            ("The weather is nice today.", "ఈరోజు వాతావరణం బాగుంది।"),
        ],
        'ta': [
            ("Hello, how are you?", "வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?"),
            ("The weather is nice today.", "இன்று வானிலை நன்றாக உள்ளது।"),
        ],
        'gu': [
            ("Hello, how are you?", "હેલો, તમે કેમ છો?"),
            ("The weather is nice today.", "આજે હવામાન સારું છે।"),
        ],
    }
    
    lang_code = target_lang.lower()
    pairs = sample_pairs.get(lang_code, sample_pairs['hi'])  # Default to Hindi
    
    src_file = os.path.join(output_dir, 'train.en')
    tgt_file = os.path.join(output_dir, f'train.{lang_code}')
    
    with open(src_file, 'w', encoding='utf-8') as f_src, \
         open(tgt_file, 'w', encoding='utf-8') as f_tgt:
        for i in range(num_examples):
            pair = pairs[i % len(pairs)]
            f_src.write(pair[0] + '\n')
            f_tgt.write(pair[1] + '\n')
    
    print(f"Sample {target_lang.upper()} dataset created with {num_examples} examples")
    print(f"Files: {src_file}, {tgt_file}")


def split_data(src_file: str, tgt_file: str, output_dir: str, 
               train_ratio: float = 0.98, dev_ratio: float = 0.01):
    """Split data into train/dev/test sets."""
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
    
    # Get file extension for target language
    # Handle both .hi and train.hi formats
    tgt_basename = os.path.basename(tgt_file)
    if '.' in tgt_basename:
        tgt_ext = tgt_basename.split('.')[-1]
    else:
        tgt_ext = 'hi'  # Default to Hindi
    
    # Ensure we have a valid extension
    if not tgt_ext or tgt_ext == 'en':
        tgt_ext = 'hi'  # Default
    
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
        with open(os.path.join(output_dir, f'{split_name}.{tgt_ext}'), 'w', encoding='utf-8') as f:
            f.writelines(tgt_data)
        print(f"{split_name}: {len(src_data)} examples")
    
    print(f"\nData splits saved to {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download and prepare Indian language datasets')
    parser.add_argument('--target_lang', type=str, required=True, 
                   help='Target language code (hi, bn, te, ta, gu, kn, ml, mr, pa, ur)')
    parser.add_argument('--output_dir', type=str, default='data/raw', help='Output directory')
    parser.add_argument('--use_sample', action='store_true', help='Create sample dataset')
    parser.add_argument('--sample_size', type=int, default=1000, help='Size of sample dataset')
    parser.add_argument('--split', action='store_true', help='Split data into train/dev/test')
    parser.add_argument('--split_dir', type=str, default='data/cleaned', help='Directory for split data')
    
    args = parser.parse_args()
    
    download_indian_language_dataset(
        args.target_lang,
        args.output_dir,
        args.sample_size,
        args.use_sample
    )
    
    if args.split:
        src_file = os.path.join(args.output_dir, 'train.en')
        tgt_file = os.path.join(args.output_dir, f'train.{args.target_lang.lower()}')
        if os.path.exists(src_file) and os.path.exists(tgt_file):
            split_data(src_file, tgt_file, args.split_dir)
        else:
            print("Error: Source files not found. Run download first.")

