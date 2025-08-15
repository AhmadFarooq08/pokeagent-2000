#!/usr/bin/env python3
"""
Download the 3.5M PokeAgent replay dataset
This is the critical foundation for training competitive agents
"""

import os
import sys
import pickle
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from datasets import load_dataset
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_pokechamp_dataset(output_dir: str = "data/pokechamp_raw"):
    """
    Download the main PokeChamp dataset (3.5M battles)
    This is the core dataset used by all SOTA agents
    """
    logger.info("Downloading PokeChamp dataset (3.5M battles)...")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        # Download the full dataset
        logger.info("Loading training split...")
        train_dataset = load_dataset("milkkarten/pokechamp", split="train")
        
        logger.info("Loading test split...")
        test_dataset = load_dataset("milkkarten/pokechamp", split="test") 
        
        # Save to disk for processing
        logger.info(f"Saving train dataset ({len(train_dataset)} samples)...")
        train_dataset.save_to_disk(f"{output_dir}/train")
        
        logger.info(f"Saving test dataset ({len(test_dataset)} samples)...")
        test_dataset.save_to_disk(f"{output_dir}/test")
        
        # Save metadata
        metadata = {
            "train_size": len(train_dataset),
            "test_size": len(test_dataset),
            "total_size": len(train_dataset) + len(test_dataset),
            "features": list(train_dataset.features.keys()),
            "dataset_name": "milkkarten/pokechamp"
        }
        
        with open(f"{output_dir}/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Dataset download complete!")
        logger.info(f"Total battles: {metadata['total_size']:,}")
        logger.info(f"Saved to: {output_dir}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        return False

def download_additional_datasets(output_dir: str = "data/additional"):
    """
    Download additional Pokemon datasets for augmentation
    """
    logger.info("Downloading additional datasets...")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # List of additional datasets to try
    additional_sources = [
        # Add any other Pokemon battle datasets here
        # "username/pokemon-battles-v2",
        # "pokemon-community/vgc-battles", 
    ]
    
    downloaded = []
    for source in additional_sources:
        try:
            logger.info(f"Trying to download {source}...")
            dataset = load_dataset(source)
            dataset.save_to_disk(f"{output_dir}/{source.replace('/', '_')}")
            downloaded.append(source)
            logger.info(f"Successfully downloaded {source}")
        except Exception as e:
            logger.warning(f"Could not download {source}: {e}")
    
    return downloaded

def validate_dataset(data_dir: str) -> bool:
    """
    Validate the downloaded dataset
    """
    logger.info("Validating dataset...")
    
    # Check if files exist
    required_files = [
        f"{data_dir}/train",
        f"{data_dir}/test", 
        f"{data_dir}/metadata.json"
    ]
    
    for file_path in required_files:
        if not Path(file_path).exists():
            logger.error(f"Missing required file: {file_path}")
            return False
    
    # Load metadata
    with open(f"{data_dir}/metadata.json", 'r') as f:
        metadata = json.load(f)
    
    # Check dataset sizes
    if metadata["total_size"] < 1000000:  # Should be ~3.5M
        logger.warning(f"Dataset seems small: {metadata['total_size']} battles")
    
    # Try loading a sample
    try:
        from datasets import Dataset
        train_dataset = Dataset.load_from_disk(f"{data_dir}/train")
        sample = train_dataset[0]
        
        # Check required fields
        required_fields = ['log', 'winner', 'format']
        for field in required_fields:
            if field not in sample:
                logger.error(f"Missing required field: {field}")
                return False
        
        logger.info("Dataset validation passed!")
        logger.info(f"Sample battle format: {sample.get('format', 'unknown')}")
        
        return True
        
    except Exception as e:
        logger.error(f"Dataset validation failed: {e}")
        return False

def filter_gen9ou_battles(input_dir: str, output_dir: str):
    """
    Filter dataset to only Gen9OU battles (our target format)
    """
    logger.info("Filtering for Gen9OU battles...")
    
    from datasets import Dataset
    
    # Load full dataset
    train_dataset = Dataset.load_from_disk(f"{input_dir}/train")
    test_dataset = Dataset.load_from_disk(f"{input_dir}/test")
    
    # Filter for Gen9OU
    gen9ou_formats = ['gen9ou', 'gen9overused']
    
    train_filtered = train_dataset.filter(
        lambda x: x['format'].lower() in gen9ou_formats
    )
    test_filtered = test_dataset.filter(
        lambda x: x['format'].lower() in gen9ou_formats
    )
    
    logger.info(f"Filtered train: {len(train_dataset)} -> {len(train_filtered)}")
    logger.info(f"Filtered test: {len(test_dataset)} -> {len(test_filtered)}")
    
    # Save filtered datasets
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    train_filtered.save_to_disk(f"{output_dir}/train")
    test_filtered.save_to_disk(f"{output_dir}/test")
    
    # Save metadata
    metadata = {
        "train_size": len(train_filtered),
        "test_size": len(test_filtered),
        "format": "gen9ou",
        "filtered_from": input_dir
    }
    
    with open(f"{output_dir}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Gen9OU filtering complete: {len(train_filtered) + len(test_filtered):,} battles")

def analyze_dataset(data_dir: str):
    """
    Analyze the dataset to understand its structure
    """
    logger.info("Analyzing dataset structure...")
    
    from datasets import Dataset
    
    # Load dataset
    train_dataset = Dataset.load_from_disk(f"{data_dir}/train")
    
    # Sample analysis
    sample = train_dataset[0]
    logger.info(f"Sample keys: {list(sample.keys())}")
    
    # Format distribution
    formats = {}
    for i in tqdm(range(min(10000, len(train_dataset))), desc="Analyzing formats"):
        format_name = train_dataset[i]['format']
        formats[format_name] = formats.get(format_name, 0) + 1
    
    logger.info("Format distribution:")
    for format_name, count in sorted(formats.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {format_name}: {count:,}")
    
    # Battle length analysis  
    lengths = []
    for i in tqdm(range(min(1000, len(train_dataset))), desc="Analyzing lengths"):
        log = train_dataset[i]['log']
        turn_count = log.count('|turn|')
        lengths.append(turn_count)
    
    if lengths:
        import numpy as np
        logger.info(f"Battle length stats:")
        logger.info(f"  Mean: {np.mean(lengths):.1f} turns")
        logger.info(f"  Median: {np.median(lengths):.1f} turns") 
        logger.info(f"  Min: {np.min(lengths)} turns")
        logger.info(f"  Max: {np.max(lengths)} turns")

def main():
    parser = argparse.ArgumentParser(description="Download PokeAgent datasets")
    parser.add_argument("--output-dir", default="data", help="Output directory")
    parser.add_argument("--filter-gen9ou", action="store_true", help="Filter for Gen9OU only")
    parser.add_argument("--analyze", action="store_true", help="Analyze dataset structure")
    parser.add_argument("--validate", action="store_true", help="Validate dataset")
    args = parser.parse_args()
    
    # Download main dataset
    raw_dir = f"{args.output_dir}/pokechamp_raw"
    success = download_pokechamp_dataset(raw_dir)
    
    if not success:
        logger.error("Dataset download failed!")
        sys.exit(1)
    
    # Validate if requested
    if args.validate:
        if not validate_dataset(raw_dir):
            logger.error("Dataset validation failed!")
            sys.exit(1)
    
    # Filter for Gen9OU if requested
    if args.filter_gen9ou:
        gen9ou_dir = f"{args.output_dir}/gen9ou_only"
        filter_gen9ou_battles(raw_dir, gen9ou_dir)
    
    # Analyze if requested
    if args.analyze:
        analyze_dataset(raw_dir)
    
    logger.info("Dataset download and processing complete!")

if __name__ == "__main__":
    main()