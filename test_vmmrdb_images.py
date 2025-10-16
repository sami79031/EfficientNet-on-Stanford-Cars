#!/usr/bin/env python3
"""
Test script to examine VMMRdb images and understand PIL issues.
"""

import os
from pathlib import Path
from PIL import Image
import pandas as pd

def test_vmmrdb_images():
    """Test a few images from VMMRdb to understand the PIL issues."""
    vmmrdb_path = Path("/Users/samiali/Downloads/VMMRdb")
    
    print("Testing VMMRdb images...")
    
    # Test a few directories
    test_dirs = ["acura_cl_1997", "bmw_325i_1995", "mercedes_benz_slk280_1995"]
    
    for test_dir in test_dirs:
        dir_path = vmmrdb_path / test_dir
        if not dir_path.exists():
            print(f"Directory {test_dir} not found")
            continue
            
        print(f"\nTesting directory: {test_dir}")
        print(f"Files in directory: {len(list(dir_path.glob('*')))}")
        
        # Test first few images
        image_files = list(dir_path.glob("*.jpg"))[:5]
        
        for img_file in image_files:
            print(f"\nTesting file: {img_file.name}")
            print(f"File size: {img_file.stat().st_size} bytes")
            
            try:
                # Try to open the image
                with Image.open(img_file) as img:
                    print(f"✓ Successfully opened image")
                    print(f"  Format: {img.format}")
                    print(f"  Mode: {img.mode}")
                    print(f"  Size: {img.size}")
                    
                    # Try to convert to RGB
                    rgb_img = img.convert('RGB')
                    print(f"✓ Successfully converted to RGB")
                    
            except Exception as e:
                print(f"✗ Error: {e}")
                print(f"  Error type: {type(e).__name__}")
                
                # Try to read the file as binary to see if it's corrupted
                try:
                    with open(img_file, 'rb') as f:
                        header = f.read(20)
                        print(f"  File header (first 20 bytes): {header}")
                except Exception as read_error:
                    print(f"  Cannot read file: {read_error}")

def test_specific_problematic_file():
    """Test the specific file that's causing issues."""
    problematic_path = "/Users/samiali/Downloads/VMMRdb/acura_cl_1997/acura_cl_1997_0.jpg"
    
    print(f"\nTesting problematic file: {problematic_path}")
    
    if not Path(problematic_path).exists():
        print("File does not exist!")
        return
        
    try:
        with Image.open(problematic_path) as img:
            print(f"✓ Successfully opened image")
            print(f"  Format: {img.format}")
            print(f"  Mode: {img.mode}")
            print(f"  Size: {img.size}")
            
            # Try to convert to RGB
            rgb_img = img.convert('RGB')
            print(f"✓ Successfully converted to RGB")
            
    except Exception as e:
        print(f"✗ Error: {e}")
        print(f"  Error type: {type(e).__name__}")
        
        # Try to read the file as binary
        try:
            with open(problematic_path, 'rb') as f:
                header = f.read(50)
                print(f"  File header (first 50 bytes): {header}")
        except Exception as read_error:
            print(f"  Cannot read file: {read_error}")

if __name__ == '__main__':
    test_vmmrdb_images()
    test_specific_problematic_file()
