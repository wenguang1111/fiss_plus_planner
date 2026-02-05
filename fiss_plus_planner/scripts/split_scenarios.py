"""
Script to split scenarios into Train, Verification, and Test folders.

Rules:
- Every 1st to 7th file -> Train
- Every 8th and 9th file -> Verification  
- Every 10th file -> Test
"""

import os
import shutil
from pathlib import Path


def split_scenarios(source_dir: str, target_dir: str = None):
    """Split scenarios into Train, Verification, and Test folders.
    
    Args:
        source_dir: Source directory containing scenario files
        target_dir: Target directory for output folders (default: same as source_dir)
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir) if target_dir else source_path
    
    # Create output directories
    train_dir = target_path / "Train"
    verification_dir = target_path / "Verification"
    test_dir = target_path / "Test"
    
    train_dir.mkdir(parents=True, exist_ok=True)
    verification_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all scenario files (exclude directories and other non-scenario files)
    scenario_files = sorted([
        f for f in source_path.iterdir() 
        if f.is_file() and f.suffix == ".xml"
    ])
    
    print(f"Found {len(scenario_files)} scenario files")
    
    train_count = 0
    verification_count = 0
    test_count = 0
    
    for i, file_path in enumerate(scenario_files):
        # Calculate position in cycle (1-10)
        position = (i % 10) + 1
        
        if position <= 7:
            # 1st to 7th -> Train
            dest_dir = train_dir
            train_count += 1
        elif position <= 9:
            # 8th and 9th -> Verification
            dest_dir = verification_dir
            verification_count += 1
        else:
            # 10th -> Test
            dest_dir = test_dir
            test_count += 1
        
        # Copy file to destination
        dest_path = dest_dir / file_path.name
        shutil.copy2(file_path, dest_path)
    
    print(f"\nSplit complete:")
    print(f"  Train: {train_count} files")
    print(f"  Verification: {verification_count} files")
    print(f"  Test: {test_count} files")
    print(f"  Total: {train_count + verification_count + test_count} files")


if __name__ == "__main__":
    source_directory = "/home/wenguang/Desktop/IROS2026/FOP_Scenarios_transformed"
    split_scenarios(source_directory)
