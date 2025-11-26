"""
Data Converter for Memories Retriever Training
Converts queries.json to TSV format for LongMatrix training
"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any


def load_json_data(json_path: str) -> List[Dict[str, Any]]:
    """Load the queries.json file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def convert_to_tsv_rows(data: List[Dict[str, Any]], 
                        all_captions: List[str] = None,
                        add_hard_negatives: bool = True,
                        samples_per_query: int = 1) -> List[str]:
    """
    Convert JSON records to TSV format.
    
    Args:
        data: List of memory records with id, caption, queries, negatives
        all_captions: All captions from dataset (for mining negatives)
        add_hard_negatives: Whether to add provided hard negatives
        samples_per_query: Number of TSV rows to generate per query
    
    Returns:
        List of TSV formatted strings
    """
    tsv_rows = []
    
    # Collect all captions for negative mining if not provided
    if all_captions is None:
        all_captions = [record['caption'] for record in data]
    
    for record in data:
        caption = record['caption']
        queries = record['queries']
        provided_negatives = record.get('negatives', [])
        
        # For each query variation
        for query in queries:
            for _ in range(samples_per_query):
                # Start with query and positive passage
                row_parts = [query, caption]
                
                # Add hard negatives if available
                if add_hard_negatives and provided_negatives:
                    row_parts.extend(provided_negatives)
                
                # Add random negatives from other captions (not the current one)
                available_negatives = [c for c in all_captions if c != caption]
                if available_negatives:
                    # Sample additional negatives (aim for 7-10 total negatives)
                    num_additional = max(0, 7 - len(provided_negatives))
                    if num_additional > 0 and num_additional <= len(available_negatives):
                        additional_negs = random.sample(available_negatives, num_additional)
                        row_parts.extend(additional_negs)
                
                # Create TSV row (tab-separated)
                tsv_row = '\t'.join(row_parts)
                tsv_rows.append(tsv_row)
    
    return tsv_rows


def split_train_dev(rows: List[str], dev_ratio: float = 0.02, shuffle: bool = True) -> tuple:
    """Split data into train and dev sets."""
    if shuffle:
        random.shuffle(rows)
    
    split_idx = int(len(rows) * (1 - dev_ratio))
    train_rows = rows[:split_idx]
    dev_rows = rows[split_idx:]
    
    return train_rows, dev_rows


def save_tsv(rows: List[str], output_path: str):
    """Save TSV rows to file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(row + '\n')
    print(f"✓ Saved {len(rows):,} rows to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Convert queries.json to TSV format for training')
    parser.add_argument('--input', type=str, required=True, 
                        help='Path to queries.json file')
    parser.add_argument('--output_dir', type=str, default='data/processed',
                        help='Output directory for TSV files')
    parser.add_argument('--dev_ratio', type=float, default=0.02,
                        help='Ratio of data to use for validation (0.02 = 2%%)')
    parser.add_argument('--shuffle', action='store_true', default=True,
                        help='Shuffle data before split')
    parser.add_argument('--no_shuffle', dest='shuffle', action='store_false',
                        help='Do not shuffle data')
    parser.add_argument('--samples_per_query', type=int, default=1,
                        help='Number of training samples per query variant')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--add_hard_negatives', action='store_true', default=True,
                        help='Include provided hard negatives')
    parser.add_argument('--no_hard_negatives', dest='add_hard_negatives', action='store_false',
                        help='Skip provided hard negatives')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading data from {args.input}...")
    data = load_json_data(args.input)
    print(f"✓ Loaded {len(data):,} memory records")
    
    # Count total queries
    total_queries = sum(len(record['queries']) for record in data)
    print(f"✓ Total query variations: {total_queries:,}")
    
    # Convert to TSV format
    print(f"\nConverting to TSV format...")
    all_captions = [record['caption'] for record in data]
    tsv_rows = convert_to_tsv_rows(
        data, 
        all_captions=all_captions,
        add_hard_negatives=args.add_hard_negatives,
        samples_per_query=args.samples_per_query
    )
    print(f"✓ Generated {len(tsv_rows):,} training samples")
    
    # Split train/dev
    print(f"\nSplitting data (dev_ratio={args.dev_ratio})...")
    train_rows, dev_rows = split_train_dev(tsv_rows, args.dev_ratio, args.shuffle)
    print(f"✓ Train: {len(train_rows):,} samples")
    print(f"✓ Dev: {len(dev_rows):,} samples")
    
    # Save files
    train_path = output_dir / 'train.tsv'
    dev_path = output_dir / 'dev.tsv'
    
    print(f"\nSaving files...")
    save_tsv(train_rows, str(train_path))
    save_tsv(dev_rows, str(dev_path))
    
    print(f"\n{'='*60}")
    print(f"✓ Conversion complete!")
    print(f"{'='*60}")
    print(f"\nOutput files:")
    print(f"  - Train: {train_path}")
    print(f"  - Dev:   {dev_path}")
    print(f"\nNext steps:")
    print(f"  1. Update config.yaml with:")
    print(f"     train_tsv: {train_path}")
    print(f"     dev_tsv: {dev_path}")
    print(f"  2. Run training:")
    print(f"     python finetune/run_longmatrix.py --config finetune/config_used.yaml")


if __name__ == '__main__':
    main()
