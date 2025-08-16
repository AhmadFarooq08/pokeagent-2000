#!/usr/bin/env python3
"""
Merge real and synthetic Pokemon datasets
"""

import os
import json
import shutil
import argparse
import logging
from pathlib import Path
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def merge_datasets(real_dir: str, synthetic_dir: str, output_dir: str, ratio: float = 0.5):
    """
    Merge real and synthetic datasets
    
    Args:
        real_dir: Directory with real Pokemon data
        synthetic_dir: Directory with synthetic data
        output_dir: Output directory for merged data
        ratio: Proportion of synthetic data (0.5 = 50/50 mix)
    """
    real_path = Path(real_dir)
    synthetic_path = Path(synthetic_dir)
    output_path = Path(output_dir)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all data files
    real_files = list(real_path.glob("*.json"))
    synthetic_files = list(synthetic_path.glob("*.json"))
    
    # Filter out metadata files
    real_files = [f for f in real_files if 'metadata' not in f.name]
    synthetic_files = [f for f in synthetic_files if 'metadata' not in f.name]
    
    logger.info(f"Found {len(real_files)} real files")
    logger.info(f"Found {len(synthetic_files)} synthetic files")
    
    if len(real_files) == 0:
        logger.error("No real data files found!")
        return False
    
    # Calculate how many synthetic files to use
    n_synthetic = int(len(real_files) * ratio / (1 - ratio)) if ratio < 1.0 else len(synthetic_files)
    n_synthetic = min(n_synthetic, len(synthetic_files))
    
    logger.info(f"Using {len(real_files)} real + {n_synthetic} synthetic files")
    
    # Copy real files
    for i, file in enumerate(real_files):
        if i % 1000 == 0:
            logger.info(f"Copying real files: {i}/{len(real_files)}")
        target = output_path / f"real_{file.name}"
        shutil.copy2(file, target)
    
    # Copy synthetic files
    for i, file in enumerate(synthetic_files[:n_synthetic]):
        if i % 100 == 0:
            logger.info(f"Copying synthetic files: {i}/{n_synthetic}")
        target = output_path / f"synthetic_{file.name}"
        shutil.copy2(file, target)
    
    # Create metadata
    metadata = {
        'total_files': len(real_files) + n_synthetic,
        'real_files': len(real_files),
        'synthetic_files': n_synthetic,
        'synthetic_ratio': n_synthetic / (len(real_files) + n_synthetic) if (len(real_files) + n_synthetic) > 0 else 0,
        'real_dir': str(real_path),
        'synthetic_dir': str(synthetic_path),
        'created_timestamp': str(Path().cwd() / 'timestamp')
    }
    
    with open(output_path / 'merged_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Merged dataset created in {output_dir}")
    logger.info(f"Total: {metadata['total_files']} files ({metadata['synthetic_ratio']:.1%} synthetic)")
    
    return True

def validate_datasets(real_dir: str, synthetic_dir: str):
    """Validate that directories contain valid data"""
    real_path = Path(real_dir)
    synthetic_path = Path(synthetic_dir)
    
    if not real_path.exists():
        logger.error(f"Real data directory does not exist: {real_dir}")
        return False
    
    if not synthetic_path.exists():
        logger.error(f"Synthetic data directory does not exist: {synthetic_dir}")
        return False
    
    real_files = list(real_path.glob("*.json"))
    synthetic_files = list(synthetic_path.glob("*.json"))
    
    if len(real_files) == 0:
        logger.error(f"No JSON files found in real data directory: {real_dir}")
        return False
    
    if len(synthetic_files) == 0:
        logger.error(f"No JSON files found in synthetic data directory: {synthetic_dir}")
        return False
    
    # Test read one file from each
    try:
        with open(real_files[0], 'r') as f:
            json.load(f)
        logger.info(f"✅ Real data format validated")
    except Exception as e:
        logger.error(f"Failed to read real data file {real_files[0]}: {e}")
        return False
    
    try:
        with open(synthetic_files[0], 'r') as f:
            json.load(f)
        logger.info(f"✅ Synthetic data format validated")
    except Exception as e:
        logger.error(f"Failed to read synthetic data file {synthetic_files[0]}: {e}")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Merge real and synthetic Pokemon datasets")
    parser.add_argument('--real-dir', required=True, help='Real data directory')
    parser.add_argument('--synthetic-dir', required=True, help='Synthetic data directory')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--ratio', type=float, default=0.3, 
                       help='Synthetic data ratio (default: 0.3 = 30% synthetic)')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate datasets without merging')
    
    args = parser.parse_args()
    
    # Validate ratio
    if not 0.0 <= args.ratio <= 1.0:
        logger.error(f"Ratio must be between 0.0 and 1.0, got {args.ratio}")
        return 1
    
    # Validate input directories
    if not validate_datasets(args.real_dir, args.synthetic_dir):
        return 1
    
    if args.validate_only:
        logger.info("✅ Validation complete")
        return 0
    
    # Merge datasets
    success = merge_datasets(args.real_dir, args.synthetic_dir, args.output_dir, args.ratio)
    
    return 0 if success else 1

if __name__ == "__main__":
    import sys
    sys.exit(main())