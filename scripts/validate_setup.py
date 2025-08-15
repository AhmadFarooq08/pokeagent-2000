#!/usr/bin/env python3
"""
Pre-flight Validation Script
Comprehensive validation before cluster deployment
"""

import os
import sys
import time
import json
import psutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
import argparse
from dataclasses import dataclass

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of a validation check"""
    name: str
    passed: bool
    message: str
    details: Optional[Dict] = None
    benchmark_score: Optional[float] = None
    
    def __str__(self):
        status = "✅ PASS" if self.passed else "❌ FAIL"
        score_str = f" ({self.benchmark_score:.2f})" if self.benchmark_score else ""
        return f"{status} {self.name}{score_str}: {self.message}"

class SystemValidator:
    """Comprehensive system validation"""
    
    def __init__(self, comprehensive: bool = False):
        self.comprehensive = comprehensive
        self.results: List[ValidationResult] = []
        
        # Add project root to path
        sys.path.append(str(Path(__file__).parent.parent))
    
    def add_result(self, result: ValidationResult):
        """Add validation result"""
        self.results.append(result)
        logger.info(str(result))
    
    def validate_python_environment(self) -> ValidationResult:
        """Validate Python environment"""
        try:
            # Check Python version
            python_version = sys.version_info
            if python_version < (3, 8):
                return ValidationResult(
                    "Python Version",
                    False,
                    f"Python {python_version.major}.{python_version.minor} < 3.8 (required)"
                )
            
            # Check available memory
            memory_gb = psutil.virtual_memory().total / (1024**3)
            
            details = {
                'python_version': f"{python_version.major}.{python_version.minor}.{python_version.micro}",
                'memory_gb': round(memory_gb, 1),
                'cpu_count': psutil.cpu_count(),
                'platform': sys.platform
            }
            
            return ValidationResult(
                "Python Environment",
                True,
                f"Python {details['python_version']}, {details['memory_gb']}GB RAM, {details['cpu_count']} CPUs",
                details
            )
            
        except Exception as e:
            return ValidationResult(
                "Python Environment",
                False,
                f"Environment check failed: {e}"
            )
    
    def validate_dependencies(self) -> ValidationResult:
        """Validate required dependencies"""
        required_packages = {
            'torch': 'PyTorch',
            'transformers': 'Hugging Face Transformers',
            'datasets': 'Hugging Face Datasets',
            'numpy': 'NumPy',
            'pandas': 'Pandas',
            'requests': 'Requests',
            'wandb': 'Weights & Biases'
        }
        
        missing_packages = []
        installed_versions = {}
        
        for package, description in required_packages.items():
            try:
                module = __import__(package)
                version = getattr(module, '__version__', 'unknown')
                installed_versions[package] = version
            except ImportError:
                missing_packages.append(f"{package} ({description})")
        
        if missing_packages:
            return ValidationResult(
                "Dependencies",
                False,
                f"Missing packages: {', '.join(missing_packages)}",
                {'missing': missing_packages, 'installed': installed_versions}
            )
        
        return ValidationResult(
            "Dependencies",
            True,
            f"All {len(required_packages)} required packages installed",
            installed_versions
        )
    
    def validate_cuda_setup(self) -> ValidationResult:
        """Validate CUDA setup"""
        try:
            import torch
            
            if not torch.cuda.is_available():
                return ValidationResult(
                    "CUDA Setup",
                    False,
                    "CUDA not available - GPU training will not work"
                )
            
            gpu_count = torch.cuda.device_count()
            gpu_info = []
            
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / (1024**3)
                gpu_info.append({
                    'device_id': i,
                    'name': props.name,
                    'memory_gb': round(memory_gb, 1),
                    'compute_capability': f"{props.major}.{props.minor}"
                })
            
            details = {
                'cuda_version': torch.version.cuda,
                'pytorch_version': torch.__version__,
                'gpu_count': gpu_count,
                'gpus': gpu_info
            }
            
            total_memory = sum(gpu['memory_gb'] for gpu in gpu_info)
            
            return ValidationResult(
                "CUDA Setup",
                True,
                f"{gpu_count} GPUs, {total_memory:.1f}GB total VRAM",
                details
            )
            
        except ImportError:
            return ValidationResult(
                "CUDA Setup",
                False,
                "PyTorch not available for CUDA validation"
            )
        except Exception as e:
            return ValidationResult(
                "CUDA Setup",
                False,
                f"CUDA validation failed: {e}"
            )
    
    def validate_model_creation(self) -> ValidationResult:
        """Validate model can be created"""
        try:
            from models.metamon_transformer import MetamonTransformer, ModelConfig
            
            # Create model
            config = ModelConfig()
            model = MetamonTransformer(config)
            
            param_count = sum(p.numel() for p in model.parameters())
            model_size_mb = param_count * 4 / (1024**2)  # Assuming float32
            
            details = {
                'parameter_count': param_count,
                'model_size_mb': round(model_size_mb, 1),
                'hidden_size': config.hidden_size,
                'num_layers': config.num_layers,
                'num_heads': config.num_heads
            }
            
            return ValidationResult(
                "Model Creation",
                True,
                f"{param_count:,} parameters, {model_size_mb:.1f}MB",
                details
            )
            
        except Exception as e:
            return ValidationResult(
                "Model Creation",
                False,
                f"Model creation failed: {e}"
            )
    
    def validate_data_pipeline(self) -> ValidationResult:
        """Validate data pipeline"""
        try:
            from tests.mock_data import MockDataGenerator
            from data.real_pokemon_dataset import RealPokemonDataset, DatasetConfig
            
            # Create mock data
            generator = MockDataGenerator()
            temp_dir = generator.create_temp_dir()
            dataset_dir = generator.create_mock_processed_dataset(100, temp_dir / "test_data")
            
            # Create dataset
            config = DatasetConfig(
                data_dir=str(dataset_dir),
                max_samples=50,
                cache_size=10
            )
            
            dataset = RealPokemonDataset(config, split='train')
            
            # Test sample loading
            if len(dataset) > 0:
                sample = dataset[0]
                
                # Validate sample structure
                required_keys = ['input_ids', 'labels', 'rewards']
                for key in required_keys:
                    if key not in sample:
                        raise ValueError(f"Missing key in sample: {key}")
            
            generator.cleanup()
            
            details = {
                'dataset_size': len(dataset),
                'sample_keys': list(sample.keys()) if len(dataset) > 0 else [],
                'config': config.__dict__
            }
            
            return ValidationResult(
                "Data Pipeline",
                True,
                f"Dataset created with {len(dataset)} samples",
                details
            )
            
        except Exception as e:
            return ValidationResult(
                "Data Pipeline",
                False,
                f"Data pipeline validation failed: {e}"
            )
    
    def benchmark_data_loading(self) -> ValidationResult:
        """Benchmark data loading performance"""
        try:
            from tests.mock_data import MockDataGenerator
            from data.real_pokemon_dataset import RealPokemonDataset, DatasetConfig, create_real_dataloader
            
            # Create larger mock dataset
            generator = MockDataGenerator()
            temp_dir = generator.create_temp_dir()
            dataset_dir = generator.create_mock_processed_dataset(1000, temp_dir / "benchmark_data")
            
            config = DatasetConfig(
                data_dir=str(dataset_dir),
                max_samples=1000,
                cache_size=100
            )
            
            # Benchmark dataset creation
            start_time = time.time()
            dataset = RealPokemonDataset(config, split='train')
            creation_time = time.time() - start_time
            
            # Benchmark data loading
            dataloader = create_real_dataloader(
                config, 
                split='train', 
                batch_size=16, 
                num_workers=0  # Single threaded for fair comparison
            )
            
            start_time = time.time()
            samples_loaded = 0
            
            for batch_idx, batch in enumerate(dataloader):
                samples_loaded += batch['labels'].shape[0] if hasattr(batch['labels'], 'shape') else len(batch['labels'])
                if batch_idx >= 10:  # Load 10 batches
                    break
            
            loading_time = time.time() - start_time
            samples_per_second = samples_loaded / loading_time if loading_time > 0 else 0
            
            generator.cleanup()
            
            details = {
                'dataset_creation_time': round(creation_time, 3),
                'samples_loaded': samples_loaded,
                'loading_time': round(loading_time, 3),
                'samples_per_second': round(samples_per_second, 1)
            }
            
            # Score based on samples per second (target: >100)
            score = min(100, samples_per_second) / 100 * 100
            
            return ValidationResult(
                "Data Loading Benchmark",
                samples_per_second > 50,  # Minimum acceptable rate
                f"{samples_per_second:.1f} samples/sec",
                details,
                score
            )
            
        except Exception as e:
            return ValidationResult(
                "Data Loading Benchmark",
                False,
                f"Benchmark failed: {e}"
            )
    
    def benchmark_model_forward(self) -> ValidationResult:
        """Benchmark model forward pass"""
        try:
            import torch
            from models.metamon_transformer import MetamonTransformer, ModelConfig, StateEncoder
            
            if not torch.cuda.is_available():
                return ValidationResult(
                    "Model Forward Benchmark",
                    False,
                    "CUDA not available for GPU benchmark"
                )
            
            # Create model and move to GPU
            config = ModelConfig()
            model = MetamonTransformer(config)
            device = torch.device('cuda:0')
            model = model.to(device)
            model.eval()
            
            # Create encoder and sample input
            encoder = StateEncoder()
            dummy_input = encoder.encode_battle_state(None)
            
            # Move input to GPU
            gpu_input = {}
            for key, value in dummy_input.items():
                gpu_input[key] = torch.tensor(value).to(device)
            
            # Warmup
            with torch.no_grad():
                for _ in range(10):
                    _ = model(gpu_input)
            
            # Benchmark
            torch.cuda.synchronize()
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(100):
                    output = model(gpu_input)
            
            torch.cuda.synchronize()
            end_time = time.time()
            
            total_time = end_time - start_time
            time_per_forward = total_time / 100 * 1000  # milliseconds
            
            details = {
                'forward_passes': 100,
                'total_time_sec': round(total_time, 3),
                'time_per_forward_ms': round(time_per_forward, 2),
                'throughput_fps': round(100 / total_time, 1),
                'model_parameters': sum(p.numel() for p in model.parameters()),
                'gpu_memory_mb': round(torch.cuda.max_memory_allocated() / 1024**2, 1)
            }
            
            # Score based on time per forward (target: <50ms)
            score = max(0, min(100, (50 - time_per_forward) / 50 * 100))
            
            return ValidationResult(
                "Model Forward Benchmark",
                time_per_forward < 100,  # Should be under 100ms
                f"{time_per_forward:.1f}ms per forward pass",
                details,
                score
            )
            
        except Exception as e:
            return ValidationResult(
                "Model Forward Benchmark",
                False,
                f"Benchmark failed: {e}"
            )
    
    def validate_distributed_setup(self) -> ValidationResult:
        """Validate distributed training setup"""
        try:
            # Check if we can import distributed components
            from scripts.train_multinode import DistributedConfig, GracefulShutdown
            
            # Test config creation
            config = DistributedConfig(
                nodes=2,
                gpus_per_node=4,
                batch_size_per_gpu=128,
                dry_run=True
            )
            
            # Test shutdown handler
            shutdown = GracefulShutdown()
            
            details = {
                'world_size': config.world_size,
                'total_batch_size': config.total_batch_size,
                'scaled_lr': config.scale_learning_rate(),
                'shutdown_handler': True
            }
            
            return ValidationResult(
                "Distributed Setup",
                True,
                f"Config supports {config.world_size} GPUs",
                details
            )
            
        except Exception as e:
            return ValidationResult(
                "Distributed Setup",
                False,
                f"Distributed setup validation failed: {e}"
            )
    
    def validate_slurm_script(self) -> ValidationResult:
        """Validate SLURM launch script"""
        try:
            script_path = Path(__file__).parent / "launch_training.sh"
            
            if not script_path.exists():
                return ValidationResult(
                    "SLURM Script",
                    False,
                    f"Launch script not found: {script_path}"
                )
            
            if not os.access(script_path, os.X_OK):
                return ValidationResult(
                    "SLURM Script",
                    False,
                    "Launch script is not executable"
                )
            
            # Test script help
            try:
                result = subprocess.run(
                    [str(script_path), "--help"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if result.returncode != 0:
                    return ValidationResult(
                        "SLURM Script",
                        False,
                        f"Script help failed with return code {result.returncode}"
                    )
                
            except subprocess.TimeoutExpired:
                return ValidationResult(
                    "SLURM Script",
                    False,
                    "Script help timed out"
                )
            
            details = {
                'script_path': str(script_path),
                'executable': True,
                'help_works': True
            }
            
            return ValidationResult(
                "SLURM Script",
                True,
                "Launch script is ready",
                details
            )
            
        except Exception as e:
            return ValidationResult(
                "SLURM Script",
                False,
                f"SLURM script validation failed: {e}"
            )
    
    def run_all_validations(self) -> Dict[str, Any]:
        """Run all validations"""
        logger.info("🔍 Starting comprehensive validation...")
        
        # Core validations (always run)
        validations = [
            self.validate_python_environment,
            self.validate_dependencies,
            self.validate_cuda_setup,
            self.validate_model_creation,
            self.validate_data_pipeline,
            self.validate_distributed_setup,
            self.validate_slurm_script
        ]
        
        # Comprehensive validations (optional)
        if self.comprehensive:
            validations.extend([
                self.benchmark_data_loading,
                self.benchmark_model_forward
            ])
        
        # Run validations
        for validation in validations:
            try:
                result = validation()
                self.add_result(result)
            except Exception as e:
                error_result = ValidationResult(
                    validation.__name__.replace('validate_', '').replace('benchmark_', ''),
                    False,
                    f"Validation crashed: {e}"
                )
                self.add_result(error_result)
        
        # Calculate summary
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        failed_tests = total_tests - passed_tests
        
        benchmarks = [r for r in self.results if r.benchmark_score is not None]
        avg_benchmark = sum(r.benchmark_score for r in benchmarks) / len(benchmarks) if benchmarks else None
        
        summary = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            'all_passed': failed_tests == 0,
            'benchmark_count': len(benchmarks),
            'average_benchmark_score': avg_benchmark
        }
        
        return summary

def main():
    parser = argparse.ArgumentParser(description="Validate system setup for Pokemon training")
    parser.add_argument("--comprehensive", action="store_true", 
                       help="Run comprehensive validation including benchmarks")
    parser.add_argument("--output", help="Save results to JSON file")
    parser.add_argument("--quiet", action="store_true", help="Suppress detailed output")
    
    args = parser.parse_args()
    
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    # Run validation
    validator = SystemValidator(comprehensive=args.comprehensive)
    summary = validator.run_all_validations()
    
    # Print summary
    print("\n" + "="*60)
    print("🎯 VALIDATION SUMMARY")
    print("="*60)
    print(f"Tests run: {summary['total_tests']}")
    print(f"Passed: {summary['passed_tests']}")
    print(f"Failed: {summary['failed_tests']}")
    print(f"Success rate: {summary['success_rate']:.1f}%")
    
    if summary['benchmark_count'] > 0:
        print(f"Benchmarks: {summary['benchmark_count']}")
        print(f"Average score: {summary['average_benchmark_score']:.1f}/100")
    
    # Show failed tests
    failed_results = [r for r in validator.results if not r.passed]
    if failed_results:
        print(f"\n❌ FAILED TESTS:")
        for result in failed_results:
            print(f"  - {result.name}: {result.message}")
    
    # Save results if requested
    if args.output:
        output_data = {
            'summary': summary,
            'results': [
                {
                    'name': r.name,
                    'passed': r.passed,
                    'message': r.message,
                    'details': r.details,
                    'benchmark_score': r.benchmark_score
                }
                for r in validator.results
            ],
            'timestamp': time.time(),
            'comprehensive': args.comprehensive
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n📊 Results saved to: {args.output}")
    
    # Final verdict
    if summary['all_passed']:
        print("\n🎉 All validations passed! System is ready for cluster deployment.")
        exit(0)
    else:
        print(f"\n⚠️  {summary['failed_tests']} validation(s) failed. Please fix issues before deploying.")
        exit(1)

if __name__ == "__main__":
    main()