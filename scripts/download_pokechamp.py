#!/usr/bin/env python3
"""
Download and Process PokéChamp Dataset
Robust downloader with resume capability and extensive error handling
"""

import os
import sys
import json
import gzip
import time
import argparse
import hashlib
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from dataclasses import dataclass
from datasets import load_dataset
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class DownloadConfig:
    """Configuration for dataset download"""
    dataset_name: str = "milkkarten/pokechamp"
    output_dir: str = "data/pokechamp"
    num_workers: int = 16
    max_games: Optional[int] = None
    min_elo: int = 1000
    chunk_size: int = 1000
    retry_attempts: int = 3
    retry_delay: float = 1.0
    checksum_verify: bool = True
    dry_run: bool = False
    use_mock_data: bool = False

class PokechampDownloader:
    """Robust downloader for PokéChamp dataset"""
    
    def __init__(self, config: DownloadConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.raw_dir = self.output_dir / "raw"
        self.processed_dir = self.output_dir / "processed"
        self.metadata_file = self.output_dir / "download_metadata.json"
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Download state
        self.download_state = self._load_state()
    
    def _load_state(self) -> Dict:
        """Load previous download state for resume capability"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load previous state: {e}")
        
        return {
            'version': '1.0',
            'start_time': time.time(),
            'downloaded_files': {},
            'failed_files': {},
            'total_games': 0,
            'processed_games': 0,
            'last_update': time.time()
        }
    
    def _save_state(self):
        """Save current download state"""
        self.download_state['last_update'] = time.time()
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.download_state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def _verify_file(self, file_path: Path, expected_size: Optional[int] = None) -> bool:
        """Verify downloaded file integrity"""
        if not file_path.exists():
            return False
        
        # Check size if provided
        if expected_size is not None:
            actual_size = file_path.stat().st_size
            if actual_size != expected_size:
                logger.warning(f"Size mismatch for {file_path}: {actual_size} != {expected_size}")
                return False
        
        # Basic file validation (can be extended with checksums)
        try:
            if file_path.suffix == '.gz':
                with gzip.open(file_path, 'rt') as f:
                    f.read(100)  # Try to read first 100 chars
            elif file_path.suffix == '.json':
                with open(file_path, 'r') as f:
                    json.load(f)
            return True
        except Exception as e:
            logger.warning(f"File validation failed for {file_path}: {e}")
            return False
    
    def _download_file_with_retry(self, url: str, output_path: Path, 
                                expected_size: Optional[int] = None) -> bool:
        """Download single file with retry logic"""
        for attempt in range(self.config.retry_attempts):
            try:
                logger.info(f"Downloading {url} (attempt {attempt + 1})")
                
                if self.config.dry_run:
                    logger.info(f"DRY RUN: Would download {url} to {output_path}")
                    return True
                
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()
                
                # Download with progress
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                
                # Verify download
                if self._verify_file(output_path, expected_size):
                    logger.info(f"Successfully downloaded {output_path}")
                    return True
                else:
                    logger.warning(f"File verification failed for {output_path}")
                    output_path.unlink(missing_ok=True)
                
            except Exception as e:
                logger.warning(f"Download attempt {attempt + 1} failed: {e}")
                if output_path.exists():
                    output_path.unlink()
                
                if attempt < self.config.retry_attempts - 1:
                    time.sleep(self.config.retry_delay * (2 ** attempt))
        
        logger.error(f"Failed to download {url} after {self.config.retry_attempts} attempts")
        return False
    
    def _create_mock_data(self) -> bool:
        """Create mock data for testing"""
        try:
            # Import mock data generator
            sys.path.append(str(Path(__file__).parent.parent))
            from tests.mock_data import MockDataGenerator
            
            logger.info("Creating mock data for testing...")
            generator = MockDataGenerator()
            
            # Create mock replay files
            num_files = self.config.max_games or 1000
            replay_dir = generator.create_mock_replay_files(num_files, self.raw_dir)
            
            # Update state
            self.download_state['total_games'] = num_files
            self.download_state['mock_data'] = True
            self._save_state()
            
            logger.info(f"Created {num_files} mock replay files")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create mock data: {e}")
            return False
    
    def download_dataset(self) -> bool:
        """Download the complete PokéChamp dataset"""
        logger.info("Starting PokéChamp dataset download...")
        
        if self.config.use_mock_data:
            return self._create_mock_data()
        
        if self.config.dry_run:
            logger.info("DRY RUN MODE: Simulating dataset download")
            return True
        
        try:
            # Load dataset from Hugging Face
            logger.info(f"Loading dataset: {self.config.dataset_name}")
            dataset = load_dataset(self.config.dataset_name, streaming=False)
            
            # Process dataset
            total_games = 0
            processed_games = 0
            
            for split_name, split_data in dataset.items():
                logger.info(f"Processing split: {split_name}")
                
                # Convert to pandas for easier processing
                df = split_data.to_pandas()
                
                # Filter by ELO if specified
                if self.config.min_elo > 0:
                    # Assuming there's a rating column - adjust as needed
                    if 'rating' in df.columns:
                        df = df[df['rating'] >= self.config.min_elo]
                        logger.info(f"Filtered to {len(df)} games with ELO >= {self.config.min_elo}")
                
                # Limit number of games if specified
                if self.config.max_games:
                    df = df.head(self.config.max_games)
                    logger.info(f"Limited to {len(df)} games")
                
                total_games += len(df)
                
                # Save games in chunks
                chunk_size = self.config.chunk_size
                for i in range(0, len(df), chunk_size):
                    chunk = df.iloc[i:i + chunk_size]
                    chunk_file = self.raw_dir / f"{split_name}_chunk_{i//chunk_size:04d}.json.gz"
                    
                    # Check if already processed
                    if str(chunk_file) in self.download_state['downloaded_files']:
                        logger.info(f"Skipping already processed chunk: {chunk_file}")
                        processed_games += len(chunk)
                        continue
                    
                    try:
                        # Convert chunk to our format
                        games = []
                        for _, row in chunk.iterrows():
                            game_data = {
                                'id': row.get('id', f"game_{i}_{_}"),
                                'log': row.get('log', ''),
                                'winner': row.get('winner', ''),
                                'players': row.get('players', []),
                                'format': row.get('format', 'gen9ou'),
                                'endType': row.get('endType', 'normal')
                            }
                            games.append(game_data)
                        
                        # Save compressed
                        with gzip.open(chunk_file, 'wt') as f:
                            json.dump(games, f)
                        
                        # Verify and update state
                        if self._verify_file(chunk_file):
                            self.download_state['downloaded_files'][str(chunk_file)] = {
                                'games': len(chunk),
                                'timestamp': time.time(),
                                'checksum': self._calculate_checksum(chunk_file) if self.config.checksum_verify else None
                            }
                            processed_games += len(chunk)
                            logger.info(f"Processed chunk {i//chunk_size + 1}: {len(chunk)} games")
                        else:
                            logger.error(f"Failed to verify chunk: {chunk_file}")
                            chunk_file.unlink(missing_ok=True)
                    
                    except Exception as e:
                        logger.error(f"Failed to process chunk {i//chunk_size}: {e}")
                        self.download_state['failed_files'][str(chunk_file)] = str(e)
                    
                    # Save state periodically
                    if processed_games % (chunk_size * 5) == 0:
                        self.download_state['total_games'] = total_games
                        self.download_state['processed_games'] = processed_games
                        self._save_state()
            
            # Final state update
            self.download_state['total_games'] = total_games
            self.download_state['processed_games'] = processed_games
            self.download_state['completed'] = True
            self._save_state()
            
            logger.info(f"Download completed: {processed_games}/{total_games} games")
            return processed_games > 0
            
        except Exception as e:
            logger.error(f"Dataset download failed: {e}")
            self.download_state['error'] = str(e)
            self._save_state()
            return False
    
    def validate_download(self) -> Tuple[bool, Dict]:
        """Validate downloaded data"""
        logger.info("Validating downloaded data...")
        
        validation_result = {
            'total_files': 0,
            'valid_files': 0,
            'corrupted_files': [],
            'total_games': 0,
            'validation_passed': False
        }
        
        try:
            # Check all downloaded files
            for file_path_str, file_info in self.download_state['downloaded_files'].items():
                file_path = Path(file_path_str)
                validation_result['total_files'] += 1
                
                if self._verify_file(file_path):
                    validation_result['valid_files'] += 1
                    validation_result['total_games'] += file_info.get('games', 0)
                else:
                    validation_result['corrupted_files'].append(str(file_path))
            
            # Check completion
            validation_result['validation_passed'] = (
                validation_result['valid_files'] == validation_result['total_files'] and
                validation_result['total_files'] > 0
            )
            
            logger.info(f"Validation: {validation_result['valid_files']}/{validation_result['total_files']} files valid")
            logger.info(f"Total games: {validation_result['total_games']}")
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            validation_result['error'] = str(e)
        
        return validation_result['validation_passed'], validation_result
    
    def get_download_stats(self) -> Dict:
        """Get comprehensive download statistics"""
        stats = {
            'total_games': self.download_state.get('total_games', 0),
            'processed_games': self.download_state.get('processed_games', 0),
            'downloaded_files': len(self.download_state.get('downloaded_files', {})),
            'failed_files': len(self.download_state.get('failed_files', {})),
            'completion_rate': 0.0,
            'elapsed_time': time.time() - self.download_state.get('start_time', time.time()),
            'estimated_size_gb': 0.0
        }
        
        if stats['total_games'] > 0:
            stats['completion_rate'] = stats['processed_games'] / stats['total_games']
        
        # Estimate total size
        total_size = 0
        for file_path_str in self.download_state.get('downloaded_files', {}):
            file_path = Path(file_path_str)
            if file_path.exists():
                total_size += file_path.stat().st_size
        
        stats['estimated_size_gb'] = total_size / (1024**3)
        
        return stats

def main():
    parser = argparse.ArgumentParser(description="Download PokéChamp Dataset")
    parser.add_argument("--output-dir", default="data/pokechamp", 
                       help="Output directory for dataset")
    parser.add_argument("--num-workers", type=int, default=16,
                       help="Number of worker processes")
    parser.add_argument("--max-games", type=int, default=None,
                       help="Maximum number of games to download")
    parser.add_argument("--min-elo", type=int, default=1000,
                       help="Minimum ELO rating to include")
    parser.add_argument("--chunk-size", type=int, default=1000,
                       help="Number of games per file chunk")
    parser.add_argument("--retry-attempts", type=int, default=3,
                       help="Number of retry attempts for failed downloads")
    parser.add_argument("--dry-run", action="store_true",
                       help="Simulate download without actually downloading")
    parser.add_argument("--use-mock-data", action="store_true",
                       help="Create mock data for testing")
    parser.add_argument("--validate-only", action="store_true",
                       help="Only validate existing download")
    parser.add_argument("--stats", action="store_true",
                       help="Show download statistics")
    
    args = parser.parse_args()
    
    # Create config
    config = DownloadConfig(
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        max_games=args.max_games,
        min_elo=args.min_elo,
        chunk_size=args.chunk_size,
        retry_attempts=args.retry_attempts,
        dry_run=args.dry_run,
        use_mock_data=args.use_mock_data
    )
    
    # Create downloader
    downloader = PokechampDownloader(config)
    
    if args.stats:
        stats = downloader.get_download_stats()
        print("\n📊 Download Statistics:")
        print(f"  Total games: {stats['total_games']:,}")
        print(f"  Processed games: {stats['processed_games']:,}")
        print(f"  Downloaded files: {stats['downloaded_files']}")
        print(f"  Failed files: {stats['failed_files']}")
        print(f"  Completion rate: {stats['completion_rate']*100:.1f}%")
        print(f"  Elapsed time: {stats['elapsed_time']/3600:.1f} hours")
        print(f"  Estimated size: {stats['estimated_size_gb']:.2f} GB")
        return
    
    if args.validate_only:
        success, validation_result = downloader.validate_download()
        print(f"\n✅ Validation {'PASSED' if success else 'FAILED'}")
        print(f"  Valid files: {validation_result['valid_files']}/{validation_result['total_files']}")
        print(f"  Total games: {validation_result['total_games']:,}")
        if validation_result['corrupted_files']:
            print(f"  Corrupted files: {len(validation_result['corrupted_files'])}")
        return
    
    # Run download
    logger.info("🚀 Starting PokéChamp dataset download")
    start_time = time.time()
    
    try:
        success = downloader.download_dataset()
        
        if success:
            # Validate download
            validation_passed, validation_result = downloader.validate_download()
            
            # Show final statistics
            stats = downloader.get_download_stats()
            elapsed_time = time.time() - start_time
            
            print(f"\n🎉 Download {'completed successfully' if validation_passed else 'completed with issues'}!")
            print(f"  Downloaded: {stats['processed_games']:,} games")
            print(f"  Files: {stats['downloaded_files']}")
            print(f"  Size: {stats['estimated_size_gb']:.2f} GB")
            print(f"  Time: {elapsed_time/3600:.1f} hours")
            print(f"  Validation: {'PASSED' if validation_passed else 'FAILED'}")
            
        else:
            print("❌ Download failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Download interrupted by user")
        print("\n⏸️ Download interrupted - progress saved for resume")
    except Exception as e:
        logger.error(f"Download failed with error: {e}")
        print(f"❌ Download failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()