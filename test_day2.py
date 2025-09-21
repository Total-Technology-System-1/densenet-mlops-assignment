"""
Day 2 Testing Script - Complete Benchmarking System Test
Tests all Day 2 components together.
"""

import sys
import torch
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from models.densenet_base import DenseNetBase
from benchmarking.benchmark_runner import BenchmarkRunner
from utils.device_utils import detect_environment
from utils.config_parser import create_default_config, validate_config
from utils.results_exporter import ResultsExporter


def test_day2_components():
    """Test all Day 2 components systematically."""
    
    print("=" * 60)
    print("DAY 2 COMPONENT TESTING")
    print("=" * 60)
    
    # Test 1: Environment Detection
    print("\n1. Testing Environment Detection...")
    try:
        env_info = detect_environment()
        print(f"   Device: {env_info['device']}")
        print(f"   GPU Available: {env_info['cuda_available']}")
        if env_info['cuda_available']:
            print(f"   GPU: {env_info.get('current_gpu_name', 'Unknown')}")
        print("   ✅ Environment detection passed")
    except Exception as e:
        print(f"   ❌ Environment detection failed: {e}")
        return False
    
    # Test 2: Model Loading
    print("\n2. Testing DenseNet Model...")
    try:
        model = DenseNetBase(device=env_info['device'])
        model_info = model.get_model_info()
        print(f"   Model: {model_info['architecture']}")
        print(f"   Parameters: {model_info['total_parameters']:,}")
        print(f"   Size: {model_info['model_size_mb']} MB")
        print("   ✅ Model loading passed")
    except Exception as e:
        print(f"   ❌ Model loading failed: {e}")
        return False
    
    # Test 3: Basic Inference
    print("\n3. Testing Basic Inference...")
    try:
        dummy_input = torch.randn(2, 3, 224, 224)
        predictions, latency = model.single_inference(dummy_input)
        print(f"   Input shape: {dummy_input.shape}")
        print(f"   Output shape: {predictions.shape}")
        print(f"   Latency: {latency:.3f} ms")
        print("   ✅ Basic inference passed")
    except Exception as e:
        print(f"   ❌ Basic inference failed: {e}")
        return False
    
    # Test 4: Configuration System
    print("\n4. Testing Configuration System...")
    try:
        config = create_default_config()
        config_dict = validate_config({})
        print(f"   Batch sizes: {config_dict['benchmark']['batch_sizes']}")
        print(f"   Optimizations: {config_dict['optimization']['techniques']}")
        print("   ✅ Configuration system passed")
    except Exception as e:
        print(f"   ❌ Configuration system failed: {e}")
        return False
    
    # Test 5: Quick Benchmark Run
    print("\n5. Testing Quick Benchmark...")
    try:
        # Override config for quick test
        test_config = config_dict.copy()
        test_config['benchmark']['batch_sizes'] = [1, 2]  # Small batch sizes
        test_config['benchmark']['num_iterations'] = 5    # Few iterations
        test_config['output_dir'] = './test_results'
        
        # Run benchmark
        runner = BenchmarkRunner(test_config)
        results = runner.run_baseline_benchmark()
        
        print(f"   Results collected: {len(results)}")
        if results:
            sample = results[0]
            if 'error' not in sample:
                print(f"   Sample latency: {sample.get('mean_latency_ms', 0):.3f} ms")
                print(f"   Sample throughput: {sample.get('throughput_samples_sec', 0):.2f} samples/sec")
                print("   ✅ Benchmark run passed")
            else:
                print(f"   ⚠️  Benchmark had errors: {sample['error']}")
        else:
            print("   ⚠️  No results collected")
    except Exception as e:
        print(f"   ❌ Benchmark run failed: {e}")
        return False
    
    # Test 6: Results Export
    print("\n6. Testing Results Export...")
    try:
        exporter = ResultsExporter(test_config)
        csv_path = exporter.export_to_csv(results, './test_results/test_results.csv')
        print(f"   CSV exported: {csv_path}")
        
        # Check if file exists and has content
        csv_file = Path(csv_path)
        if csv_file.exists() and csv_file.stat().st_size > 0:
            print("   ✅ Results export passed")
        else:
            print("   ⚠️  CSV file empty or missing")
    except Exception as e:
        print(f"   ❌ Results export failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("DAY 2 TESTING COMPLETE!")
    print("=" * 60)
    print("✅ All core components are working")
    print("🚀 Ready for Day 3: Advanced Profiling & TensorBoard")
    
    return True


def run_mini_benchmark():
    """Run a complete mini benchmark to demonstrate Day 2 capabilities."""
    
    print("\n" + "=" * 60)
    print("MINI BENCHMARK DEMONSTRATION")
    print("=" * 60)
    
    try:
        # Setup configuration
        config = {
            'benchmark': {
                'batch_sizes': [1, 4, 8],
                'num_iterations': 10,
                'warmup_iterations': 3,
                'device': 'auto'
            },
            'output_dir': './demo_results',
            'directories': {
                'logs': './demo_results/logs'
            }
        }
        
        # Run benchmark
        runner = BenchmarkRunner(config)
        results = runner.run_all_benchmarks()
        
        # Export results
        exporter = ResultsExporter(config)
        exported_files = exporter.export_all_formats(
            results, 
            './demo_results',
            'mini_benchmark'
        )
        
        # Display summary
        print(f"\n📊 BENCHMARK SUMMARY:")
        print(f"   Total benchmarks: {len(results)}")
        successful = [r for r in results if 'error' not in r]
        print(f"   Successful: {len(successful)}")
        
        if successful:
            latencies = [r['mean_latency_ms'] for r in successful]
            throughputs = [r['throughput_samples_sec'] for r in successful]
            print(f"   Best latency: {min(latencies):.3f} ms")
            print(f"   Best throughput: {max(throughputs):.2f} samples/sec")
        
        print(f"\n📁 FILES CREATED:")
        for format_name, filepath in exported_files.items():
            print(f"   {format_name.upper()}: {filepath}")
        
        print(f"\n🎉 Mini benchmark completed successfully!")
        
    except Exception as e:
        print(f"❌ Mini benchmark failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Run tests
    if test_day2_components():
        # If basic tests pass, run mini benchmark
        run_mini_benchmark()
    else:
        print("Basic tests failed - fix issues before proceeding")
        sys.exit(1)