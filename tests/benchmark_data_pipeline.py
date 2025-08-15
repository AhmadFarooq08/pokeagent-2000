#!/usr/bin/env python3
"""
Data Pipeline Benchmark Script
Performance testing for data loading and processing
"""

import os
import sys
import time
import json
import argparse
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import psutil
import gc

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

class DataPipelineBenchmark:
    """Benchmark data pipeline performance"""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
    
    def benchmark_mock_data_generation(self, num_samples: int = 10000) -> Dict:
        """Benchmark mock data generation"""
        logger.info(f"Benchmarking mock data generation ({num_samples} samples)...")
        
        from tests.mock_data import MockDataGenerator
        
        generator = MockDataGenerator()
        
        # Benchmark team generation
        start_time = time.time()
        teams = [generator.generate_mock_team() for _ in range(100)]
        team_time = time.time() - start_time
        
        # Benchmark replay generation
        start_time = time.time()
        replays = [generator.generate_mock_replay() for _ in range(100)]
        replay_time = time.time() - start_time
        
        # Benchmark file creation
        start_time = time.time()
        temp_dir = generator.create_temp_dir()
        replay_dir = generator.create_mock_replay_files(100, temp_dir / "replays")
        file_creation_time = time.time() - start_time
        
        # Benchmark dataset creation
        start_time = time.time()
        dataset_dir = generator.create_mock_processed_dataset(num_samples, temp_dir / "dataset")
        dataset_creation_time = time.time() - start_time
        
        generator.cleanup()
        
        results = {
            'team_generation': {
                'time_per_team_ms': round(team_time / 100 * 1000, 3),
                'teams_per_second': round(100 / team_time, 1)
            },
            'replay_generation': {
                'time_per_replay_ms': round(replay_time / 100 * 1000, 3),
                'replays_per_second': round(100 / replay_time, 1)
            },
            'file_creation': {
                'time_for_100_files_sec': round(file_creation_time, 3),
                'files_per_second': round(100 / file_creation_time, 1)
            },
            'dataset_creation': {
                'time_sec': round(dataset_creation_time, 3),
                'samples_per_second': round(num_samples / dataset_creation_time, 1)
            }
        }
        
        logger.info(f"Mock data generation: {results['dataset_creation']['samples_per_second']} samples/sec")
        return results
    
    def benchmark_replay_parsing(self, num_replays: int = 1000) -> Dict:
        """Benchmark replay parsing performance"""
        logger.info(f"Benchmarking replay parsing ({num_replays} replays)...")
        
        from tests.mock_data import MockDataGenerator
        from data.replay_parser import ReplayParser
        
        # Generate mock replays
        generator = MockDataGenerator()
        replays = [generator.generate_mock_replay() for _ in range(num_replays)]
        
        # Benchmark parsing
        parser = ReplayParser()
        
        start_time = time.time()
        total_samples = 0
        
        for replay in replays:
            samples = parser.parse_replay(replay)
            if samples:
                total_samples += len(samples)
            parser.reset_state()
        
        parsing_time = time.time() - start_time
        
        generator.cleanup()
        
        results = {
            'replays_processed': num_replays,
            'total_samples': total_samples,
            'parsing_time_sec': round(parsing_time, 3),
            'replays_per_second': round(num_replays / parsing_time, 1),
            'samples_per_second': round(total_samples / parsing_time, 1),
            'avg_samples_per_replay': round(total_samples / num_replays, 1)
        }
        
        logger.info(f"Replay parsing: {results['replays_per_second']} replays/sec, {results['samples_per_second']} samples/sec")
        return results
    
    def benchmark_dataset_loading(self, num_samples: int = 10000, batch_sizes: List[int] = None) -> Dict:
        """Benchmark dataset loading with different configurations"""
        logger.info(f"Benchmarking dataset loading ({num_samples} samples)...")
        
        from tests.mock_data import MockDataGenerator
        from data.real_pokemon_dataset import RealPokemonDataset, DatasetConfig, create_real_dataloader
        
        if batch_sizes is None:
            batch_sizes = [16, 32, 64, 128, 256]
        
        # Create mock dataset
        generator = MockDataGenerator()
        temp_dir = generator.create_temp_dir()
        dataset_dir = generator.create_mock_processed_dataset(num_samples, temp_dir / "dataset")
        
        config = DatasetConfig(
            data_dir=str(dataset_dir),
            max_samples=num_samples,
            cache_size=1000
        )
        
        results = {
            'dataset_creation': {},
            'batch_loading': {}
        }
        
        # Benchmark dataset creation
        start_time = time.time()
        dataset = RealPokemonDataset(config, split='train')
        creation_time = time.time() - start_time
        
        results['dataset_creation'] = {
            'time_sec': round(creation_time, 3),
            'samples': len(dataset),
            'samples_per_second': round(len(dataset) / creation_time, 1)
        }
        
        # Benchmark different batch sizes and worker counts
        for batch_size in batch_sizes:
            for num_workers in [0, 2, 4]:
                logger.info(f"  Testing batch_size={batch_size}, num_workers={num_workers}")
                
                try:
                    # Create dataloader
                    dataloader = create_real_dataloader(
                        config,
                        split='train',
                        batch_size=batch_size,
                        num_workers=num_workers,
                        shuffle=False
                    )
                    
                    # Benchmark loading
                    start_time = time.time()
                    samples_loaded = 0
                    batches_loaded = 0
                    
                    for batch in dataloader:
                        batch_size_actual = len(batch['labels'])
                        samples_loaded += batch_size_actual
                        batches_loaded += 1
                        
                        # Load first 20 batches for timing
                        if batches_loaded >= 20:
                            break
                    
                    loading_time = time.time() - start_time
                    
                    key = f"batch_{batch_size}_workers_{num_workers}"
                    results['batch_loading'][key] = {
                        'batch_size': batch_size,
                        'num_workers': num_workers,
                        'samples_loaded': samples_loaded,
                        'batches_loaded': batches_loaded,
                        'loading_time_sec': round(loading_time, 3),
                        'samples_per_second': round(samples_loaded / loading_time, 1) if loading_time > 0 else 0,
                        'batches_per_second': round(batches_loaded / loading_time, 1) if loading_time > 0 else 0
                    }
                    
                except Exception as e:
                    logger.warning(f"Failed to benchmark batch_size={batch_size}, workers={num_workers}: {e}")
        
        generator.cleanup()
        
        # Find best configuration
        best_config = max(
            results['batch_loading'].values(),
            key=lambda x: x['samples_per_second']
        )
        
        results['best_config'] = best_config
        logger.info(f"Best config: batch_size={best_config['batch_size']}, workers={best_config['num_workers']}, {best_config['samples_per_second']} samples/sec")
        
        return results
    
    def benchmark_model_throughput(self, batch_sizes: List[int] = None) -> Dict:
        """Benchmark model throughput"""
        logger.info("Benchmarking model throughput...")
        
        try:
            import torch
            from models.metamon_transformer import MetamonTransformer, ModelConfig, StateEncoder
            
            if not torch.cuda.is_available():
                logger.warning("CUDA not available, skipping model benchmark")
                return {'error': 'CUDA not available'}
            
            if batch_sizes is None:
                batch_sizes = [1, 4, 8, 16, 32, 64]
            
            # Setup model
            config = ModelConfig()
            model = MetamonTransformer(config)
            device = torch.device('cuda:0')
            model = model.to(device)
            model.eval()
            
            encoder = StateEncoder()
            
            results = {
                'model_info': {
                    'parameters': sum(p.numel() for p in model.parameters()),
                    'hidden_size': config.hidden_size,
                    'num_layers': config.num_layers
                },
                'batch_performance': {}
            }
            
            for batch_size in batch_sizes:
                logger.info(f"  Testing batch_size={batch_size}")
                
                try:
                    # Create batch input
                    dummy_input = encoder.encode_battle_state(None)
                    batch_input = {}
                    
                    for key, value in dummy_input.items():
                        tensor = torch.tensor(value).unsqueeze(0).repeat(batch_size, *[1]*len(value.shape))
                        batch_input[key] = tensor.to(device)
                    
                    # Warmup
                    with torch.no_grad():
                        for _ in range(10):
                            _ = model(batch_input)
                    
                    # Benchmark
                    torch.cuda.synchronize()
                    start_time = time.time()
                    
                    iterations = 100
                    with torch.no_grad():
                        for _ in range(iterations):
                            output = model(batch_input)
                    
                    torch.cuda.synchronize()
                    end_time = time.time()
                    
                    total_time = end_time - start_time
                    time_per_batch = total_time / iterations
                    samples_per_second = batch_size * iterations / total_time
                    
                    # Memory usage
                    memory_mb = torch.cuda.max_memory_allocated() / 1024**2
                    
                    results['batch_performance'][f'batch_{batch_size}'] = {
                        'batch_size': batch_size,
                        'time_per_batch_ms': round(time_per_batch * 1000, 2),
                        'samples_per_second': round(samples_per_second, 1),
                        'memory_mb': round(memory_mb, 1)
                    }
                    
                    torch.cuda.reset_peak_memory_stats()
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        results['batch_performance'][f'batch_{batch_size}'] = {
                            'batch_size': batch_size,
                            'error': 'Out of memory'
                        }
                        logger.warning(f"OOM at batch_size={batch_size}")
                        break
                    else:
                        raise e
            
            # Find optimal batch size
            valid_results = {k: v for k, v in results['batch_performance'].items() 
                           if 'error' not in v}
            
            if valid_results:
                best_batch = max(valid_results.values(), key=lambda x: x['samples_per_second'])
                results['optimal_batch'] = best_batch
                logger.info(f"Optimal batch size: {best_batch['batch_size']}, {best_batch['samples_per_second']} samples/sec")
            
            return results
            
        except ImportError:
            return {'error': 'PyTorch not available'}
        except Exception as e:
            return {'error': f'Model benchmark failed: {e}'}
    
    def benchmark_memory_usage(self) -> Dict:
        """Benchmark memory usage patterns"""
        logger.info("Benchmarking memory usage...")
        
        import gc
        
        # Get initial memory
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024**2  # MB
        
        results = {
            'initial_memory_mb': round(initial_memory, 1),
            'peak_memory_mb': round(initial_memory, 1),
            'memory_stages': {}
        }
        
        def record_memory(stage: str):
            gc.collect()  # Force garbage collection
            current_memory = process.memory_info().rss / 1024**2
            results['memory_stages'][stage] = round(current_memory, 1)
            results['peak_memory_mb'] = max(results['peak_memory_mb'], current_memory)
            return current_memory
        
        # Stage 1: Mock data generation
        from tests.mock_data import MockDataGenerator
        generator = MockDataGenerator()
        record_memory('after_imports')
        
        temp_dir = generator.create_temp_dir()
        dataset_dir = generator.create_mock_processed_dataset(5000, temp_dir / "memory_test")
        record_memory('after_mock_data_creation')
        
        # Stage 2: Dataset loading
        from data.real_pokemon_dataset import RealPokemonDataset, DatasetConfig
        config = DatasetConfig(data_dir=str(dataset_dir), max_samples=5000, cache_size=1000)
        dataset = RealPokemonDataset(config, split='train')
        record_memory('after_dataset_creation')
        
        # Stage 3: Sample loading
        samples = [dataset[i] for i in range(min(100, len(dataset)))]
        record_memory('after_sample_loading')
        
        # Stage 4: Model creation (if available)
        try:
            import torch
            from models.metamon_transformer import MetamonTransformer, ModelConfig
            
            model_config = ModelConfig()
            model = MetamonTransformer(model_config)
            record_memory('after_model_creation')
            
            if torch.cuda.is_available():
                model = model.cuda()
                record_memory('after_model_to_gpu')
            
        except ImportError:
            logger.info("PyTorch not available for model memory test")
        
        # Cleanup
        generator.cleanup()
        del dataset, samples
        gc.collect()
        record_memory('after_cleanup')
        
        # Calculate memory growth
        memory_growth = results['peak_memory_mb'] - results['initial_memory_mb']
        results['memory_growth_mb'] = round(memory_growth, 1)
        
        logger.info(f"Memory usage: {results['initial_memory_mb']}MB -> {results['peak_memory_mb']}MB (+{memory_growth:.1f}MB)")
        
        return results
    
    def run_all_benchmarks(self, comprehensive: bool = False) -> Dict:
        """Run all benchmarks"""
        logger.info("🚀 Starting data pipeline benchmarks...")
        
        all_results = {
            'timestamp': time.time(),
            'comprehensive': comprehensive,
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'memory_gb': round(psutil.virtual_memory().total / 1024**3, 1),
                'python_version': sys.version
            }
        }
        
        # Core benchmarks
        benchmarks = [
            ('mock_data_generation', lambda: self.benchmark_mock_data_generation(1000 if not comprehensive else 10000)),
            ('replay_parsing', lambda: self.benchmark_replay_parsing(100 if not comprehensive else 1000)),
            ('dataset_loading', lambda: self.benchmark_dataset_loading(1000 if not comprehensive else 10000)),
            ('memory_usage', self.benchmark_memory_usage)
        ]
        
        # Add comprehensive benchmarks
        if comprehensive:
            benchmarks.append(('model_throughput', self.benchmark_model_throughput))
        
        # Run benchmarks
        for name, benchmark_func in benchmarks:
            logger.info(f"\n📊 Running {name} benchmark...")
            try:
                start_time = time.time()
                result = benchmark_func()
                end_time = time.time()
                
                result['benchmark_time_sec'] = round(end_time - start_time, 3)
                all_results[name] = result
                
                logger.info(f"✅ {name} completed in {result['benchmark_time_sec']}s")
                
            except Exception as e:
                logger.error(f"❌ {name} failed: {e}")
                all_results[name] = {'error': str(e)}
        
        # Save results
        results_file = self.output_dir / f"benchmark_results_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        logger.info(f"📁 Results saved to: {results_file}")
        
        return all_results

def main():
    parser = argparse.ArgumentParser(description="Benchmark data pipeline performance")
    parser.add_argument("--comprehensive", action="store_true", 
                       help="Run comprehensive benchmarks (slower but more thorough)")
    parser.add_argument("--output-dir", default="benchmark_results",
                       help="Output directory for results")
    parser.add_argument("--mock-samples", type=int, default=1000,
                       help="Number of mock samples for testing")
    
    args = parser.parse_args()
    
    # Run benchmarks
    benchmark = DataPipelineBenchmark(args.output_dir)
    results = benchmark.run_all_benchmarks(comprehensive=args.comprehensive)
    
    # Print summary
    print("\n" + "="*60)
    print("📊 BENCHMARK SUMMARY")
    print("="*60)
    
    if 'mock_data_generation' in results:
        mock_results = results['mock_data_generation']
        if 'dataset_creation' in mock_results:
            print(f"Mock data generation: {mock_results['dataset_creation']['samples_per_second']} samples/sec")
    
    if 'replay_parsing' in results:
        parse_results = results['replay_parsing']
        if 'replays_per_second' in parse_results:
            print(f"Replay parsing: {parse_results['replays_per_second']} replays/sec")
    
    if 'dataset_loading' in results:
        load_results = results['dataset_loading']
        if 'best_config' in load_results:
            best = load_results['best_config']
            print(f"Data loading: {best['samples_per_second']} samples/sec (batch={best['batch_size']}, workers={best['num_workers']})")
    
    if 'model_throughput' in results:
        model_results = results['model_throughput']
        if 'optimal_batch' in model_results:
            optimal = model_results['optimal_batch']
            print(f"Model throughput: {optimal['samples_per_second']} samples/sec (batch={optimal['batch_size']})")
    
    if 'memory_usage' in results:
        memory_results = results['memory_usage']
        if 'memory_growth_mb' in memory_results:
            print(f"Memory usage: {memory_results['peak_memory_mb']}MB peak (+{memory_results['memory_growth_mb']}MB)")
    
    # Check if performance is acceptable
    performance_issues = []
    
    if 'dataset_loading' in results and 'best_config' in results['dataset_loading']:
        best_throughput = results['dataset_loading']['best_config']['samples_per_second']
        if best_throughput < 100:
            performance_issues.append(f"Data loading too slow: {best_throughput} < 100 samples/sec")
    
    if 'model_throughput' in results and 'optimal_batch' in results['model_throughput']:
        model_throughput = results['model_throughput']['optimal_batch']['samples_per_second']
        if model_throughput < 50:
            performance_issues.append(f"Model throughput too slow: {model_throughput} < 50 samples/sec")
    
    if performance_issues:
        print(f"\n⚠️  Performance Issues:")
        for issue in performance_issues:
            print(f"  - {issue}")
        print("Consider optimizing before cluster deployment.")
    else:
        print(f"\n🎉 All benchmarks show acceptable performance!")

if __name__ == "__main__":
    main()