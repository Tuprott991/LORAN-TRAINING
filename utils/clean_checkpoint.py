"""
Clean Checkpoint Tool
Removes unnecessary data from checkpoint files to reduce size.
"""
import torch
import os
import sys
import shutil
from pathlib import Path

def clean_checkpoint(input_path: str, output_path: str = None, backup: bool = True):
    """
    Clean a checkpoint by removing large unnecessary fields.
    
    Args:
        input_path: Path to the checkpoint file
        output_path: Path for cleaned checkpoint (default: overwrites input)
        backup: Whether to create a backup of the original file
    """
    if not os.path.exists(input_path):
        print(f"Error: File not found: {input_path}")
        return False
    
    # Get original size
    original_size_mb = os.path.getsize(input_path) / (1024 * 1024)
    print(f"Original file size: {original_size_mb:.2f} MB")
    
    # Load checkpoint
    print("Loading checkpoint...")
    try:
        checkpoint = torch.load(input_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return False
    
    # Check what will be removed
    removed_items = []
    
    if isinstance(checkpoint, dict):
        if 'args' in checkpoint and isinstance(checkpoint['args'], dict):
            args = checkpoint['args']
            
            # Check for _dev_rows
            if '_dev_rows' in args:
                import pickle
                dev_rows_size_mb = len(pickle.dumps(args['_dev_rows'])) / (1024 * 1024)
                print(f"Found _dev_rows: {len(args['_dev_rows'])} items, ~{dev_rows_size_mb:.2f} MB")
                removed_items.append(('_dev_rows', dev_rows_size_mb))
                del args['_dev_rows']
            
            # Check for other potentially large fields
            for key in ['_ema_obj', '_teacher_model', '_optimizer', '_scheduler']:
                if key in args:
                    print(f"Found {key}, removing...")
                    removed_items.append((key, 0))
                    del args[key]
    
    if not removed_items:
        print("No large fields found to remove. Checkpoint is already clean.")
        return True
    
    # Create backup if requested
    if backup and (output_path is None or output_path == input_path):
        backup_path = str(Path(input_path).with_suffix('.pt.backup'))
        print(f"Creating backup: {backup_path}")
        shutil.copy2(input_path, backup_path)
    
    # Determine output path
    if output_path is None:
        output_path = input_path
    
    # Save cleaned checkpoint
    print(f"Saving cleaned checkpoint to: {output_path}")
    try:
        torch.save(checkpoint, output_path)
    except Exception as e:
        print(f"Error saving checkpoint: {e}")
        return False
    
    # Get new size
    new_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    saved_mb = original_size_mb - new_size_mb
    saved_pct = (saved_mb / original_size_mb) * 100
    
    print(f"\n{'='*70}")
    print("Cleaning complete!")
    print(f"{'='*70}")
    print(f"Original size: {original_size_mb:.2f} MB")
    print(f"New size: {new_size_mb:.2f} MB")
    print(f"Saved: {saved_mb:.2f} MB ({saved_pct:.1f}%)")
    print(f"\nRemoved fields:")
    for name, size in removed_items:
        if size > 0:
            print(f"  - {name}: ~{size:.2f} MB")
        else:
            print(f"  - {name}")
    
    return True

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Clean PyTorch checkpoint files by removing unnecessary data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Clean in-place with backup (default)
  python clean_checkpoint.py longmatrix.pt
  
  # Clean without backup
  python clean_checkpoint.py longmatrix.pt --no-backup
  
  # Save to a new file
  python clean_checkpoint.py longmatrix.pt -o longmatrix_clean.pt
  
  # Clean multiple files
  python clean_checkpoint.py *.pt
        """
    )
    parser.add_argument('checkpoints', nargs='+', help='Checkpoint file(s) to clean')
    parser.add_argument('-o', '--output', help='Output path (only valid with single file)')
    parser.add_argument('--no-backup', action='store_true', help='Do not create backup')
    args = parser.parse_args()
    
    if args.output and len(args.checkpoints) > 1:
        print("Error: --output can only be used with a single checkpoint file")
        return 1
    
    success = True
    for ckpt_path in args.checkpoints:
        print(f"\n{'='*70}")
        print(f"Processing: {ckpt_path}")
        print(f"{'='*70}")
        
        output_path = args.output if args.output else None
        backup = not args.no_backup
        
        if not clean_checkpoint(ckpt_path, output_path, backup):
            success = False
    
    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main())
