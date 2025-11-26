"""Check the _dev_rows field in checkpoint files"""
import torch
import sys

def check_dev_rows(path):
    print(f"\nAnalyzing: {path}")
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    
    if 'args' in ckpt and '_dev_rows' in ckpt['args']:
        dev_rows = ckpt['args']['_dev_rows']
        print(f"Type of _dev_rows: {type(dev_rows)}")
        print(f"Length of _dev_rows: {len(dev_rows)}")
        
        if len(dev_rows) > 0:
            print(f"Type of first item: {type(dev_rows[0])}")
            print(f"First item: {dev_rows[0]}")
            
            # Calculate approximate size
            import pickle
            size_bytes = len(pickle.dumps(dev_rows))
            size_mb = size_bytes / (1024 * 1024)
            print(f"Approximate size of _dev_rows: {size_mb:.2f} MB")
    else:
        print("No _dev_rows found in args")

if __name__ == '__main__':
    for path in sys.argv[1:]:
        check_dev_rows(path)
