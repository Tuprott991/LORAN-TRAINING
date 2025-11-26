"""
Checkpoint Analysis Tool
Analyzes PyTorch checkpoint files to understand their size and contents.
"""
import torch
import os
import sys
from pathlib import Path

def analyze_checkpoint(path: str):
    """Analyze a checkpoint file and print detailed information."""
    if not os.path.exists(path):
        print(f"Error: File not found: {path}")
        return
    
    # Get file size
    file_size_bytes = os.path.getsize(path)
    file_size_mb = file_size_bytes / (1024 * 1024)
    
    print(f"\n{'='*70}")
    print(f"Analyzing: {path}")
    print(f"{'='*70}")
    print(f"File size: {file_size_mb:.2f} MB ({file_size_bytes:,} bytes)")
    
    # Load checkpoint
    try:
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        print(f"\nCheckpoint type: {type(checkpoint)}")
        
        if isinstance(checkpoint, dict):
            print(f"\nTop-level keys: {list(checkpoint.keys())}")
            
            for key, value in checkpoint.items():
                print(f"\n--- Key: '{key}' ---")
                print(f"Type: {type(value)}")
                
                if isinstance(value, dict):
                    print(f"Number of items: {len(value)}")
                    
                    # Calculate size of each tensor in the dict
                    total_params = 0
                    total_size_mb = 0
                    
                    print("\nDetailed breakdown:")
                    for k, v in value.items():
                        if isinstance(v, torch.Tensor):
                            numel = v.numel()
                            dtype_size = v.element_size()
                            size_mb = (numel * dtype_size) / (1024 * 1024)
                            total_params += numel
                            total_size_mb += size_mb
                            print(f"  {k}:")
                            print(f"    Shape: {tuple(v.shape)}")
                            print(f"    Dtype: {v.dtype}")
                            print(f"    Params: {numel:,}")
                            print(f"    Size: {size_mb:.2f} MB")
                        else:
                            print(f"  {k}: {type(v)}")
                    
                    print(f"\nSummary for '{key}':")
                    print(f"  Total parameters: {total_params:,}")
                    print(f"  Total tensor size: {total_size_mb:.2f} MB")
                    
                elif isinstance(value, torch.Tensor):
                    numel = value.numel()
                    dtype_size = value.element_size()
                    size_mb = (numel * dtype_size) / (1024 * 1024)
                    print(f"Shape: {tuple(value.shape)}")
                    print(f"Dtype: {value.dtype}")
                    print(f"Params: {numel:,}")
                    print(f"Size: {size_mb:.2f} MB")
                    
                elif isinstance(value, (list, tuple)):
                    print(f"Length: {len(value)}")
                    if len(value) > 0:
                        print(f"First item type: {type(value[0])}")
                        
                else:
                    print(f"Value: {value}")
            
            # Calculate overhead (file size - tensor sizes)
            if 'model' in checkpoint:
                model_dict = checkpoint['model']
                if isinstance(model_dict, dict):
                    tensor_size = sum(
                        v.numel() * v.element_size() 
                        for v in model_dict.values() 
                        if isinstance(v, torch.Tensor)
                    ) / (1024 * 1024)
                    overhead = file_size_mb - tensor_size
                    print(f"\n{'='*70}")
                    print(f"File size overhead: {overhead:.2f} MB ({overhead/file_size_mb*100:.1f}%)")
                    print(f"Tensor data: {tensor_size:.2f} MB ({tensor_size/file_size_mb*100:.1f}%)")
                    
    except Exception as e:
        print(f"\nError loading checkpoint: {e}")
        import traceback
        traceback.print_exc()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Analyze PyTorch checkpoint files')
    parser.add_argument('checkpoints', nargs='+', help='Checkpoint file paths')
    args = parser.parse_args()
    
    for ckpt_path in args.checkpoints:
        analyze_checkpoint(ckpt_path)
    
    # Compare if multiple files
    if len(args.checkpoints) > 1:
        print(f"\n\n{'='*70}")
        print("COMPARISON SUMMARY")
        print(f"{'='*70}")
        for ckpt_path in args.checkpoints:
            if os.path.exists(ckpt_path):
                size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
                print(f"{Path(ckpt_path).name}: {size_mb:.2f} MB")

if __name__ == '__main__':
    main()
