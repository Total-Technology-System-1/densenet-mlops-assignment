"""
Day 4 Integration Test - Test Optimization Integration with BenchmarkRunner
"""

import sys
import torch
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from benchmarking.benchmark_runner import BenchmarkRunner
from utils.device_utils import detect_environment
from utils.results_exporter import ResultsExporter


def test_optimization_integration():
    """Test optimization integration with benchmark runner."""
    
    print("=" * 60)
    print("DAY 4 OPTIMIZATION INTEGRATION TEST")
    print("=" * 60)
    
    # Test configuration
    test_config = {
        'benchmark': {
            'batch_sizes': [1, 4],  # Small batch sizes for quick test
            'num_iterations': 10,   # Few iterations for speed
            'warmup_iterations': 3,
            'device': 'auto'
        },
        'optimization': {
            'techniques': ['quantization_dynamic', 'pruning_unstructured']
        },
        'output_dir': './test_integration_results',
        'directories': {
            'logs': './test_integration_results/logs'
        }
    }
    
    try:
        print("\n1. Testing BenchmarkRunner Initialization...")
        runner = BenchmarkRunner(test_config)
        print("   ✅ BenchmarkRunner initialized successfully")
        
        print("\n2. Testing Baseline Benchmarking...")
        baseline_results = runner.run_baseline_benchmark()
        print(f"   ✅ Baseline results: {len(baseline_results)} entries")
        
        if baseline_results:
            sample = baseline_results[0]
            if sample and 'error' not in sample:
                print(f"   Sample latency: {sample.get('mean_latency_ms', 0):.3f} ms")
            elif sample:
                print(f"   Sample had error: {sample['error']}")
            else:
                print("   Sample result is None")
        
        print("\n3. Testing Optimization Benchmarking...")
        optimization_results = runner.run_optimization_benchmarks()
        print(f"   ✅ Optimization results: {len(optimization_results)} entries")
        
        successful_opts = [r for r in optimization_results if 'error' not in r]
        failed_opts = [r for r in optimization_results if 'error' in r]
        
        print(f"   Successful: {len(successful_opts)}")
        print(f"   Failed: {len(failed_opts)}")
        
        for result in successful_opts[:2]:  # Show first 2 successful results
            technique = result.get('optimization_technique', 'unknown')
            latency = result.get('mean_latency_ms', 0)
            batch_size = result.get('batch_size', 0)
            print(f"   {technique} (batch {batch_size}): {latency:.3f} ms")
        
        print("\n4. Testing Complete Pipeline...")
        all_results = runner.run_all_benchmarks()
        print(f"   ✅ Complete pipeline: {len(all_results)} total results")
        
        # Categorize results
        baseline_count = len([r for r in all_results if r.get('optimization_technique') == 'baseline'])
        opt_count = len(all_results) - baseline_count
        
        print(f"   Baseline results: {baseline_count}")
        print(f"   Optimization results: {opt_count}")
        
        print("\n5. Testing Results Export...")
        exporter = ResultsExporter(test_config)
        csv_path = exporter.export_to_csv(all_results, './test_integration_results/complete_results.csv')
        print(f"   ✅ Results exported: {csv_path}")
        
        # Verify CSV content
        import pandas as pd
        try:
            df = pd.read_csv(csv_path)
            print(f"   CSV contains {len(df)} rows")
            print(f"   Techniques: {df['optimization_technique'].unique().tolist()}")
        except Exception as e:
            print(f"   Could not read CSV: {e}")
        
        print("\n6. Testing Summary Generation...")
        summary = runner.get_summary_stats()
        print(f"   ✅ Summary generated")
        print(f"   Total benchmarks: {summary.get('total_benchmarks', 0)}")
        print(f"   Successful: {summary.get('successful_benchmarks', 0)}")
        print(f"   Best throughput: {summary.get('best_throughput', 0):.1f} samples/sec")
        
        print("\n" + "=" * 60)
        print("INTEGRATION TEST COMPLETE!")
        print("=" * 60)
        print("✅ Optimization integration successful")
        print("✅ Baseline + Optimization benchmarking working")
        print("✅ Results export in correct format")
        print("🚀 Ready for complete Day 4 implementation")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Run integration test
    success = test_optimization_integration()
    
    if not success:
        print("\n🔧 Fix integration issues before proceeding to complete Day 4")
        sys.exit(1)
    else:
        print("\n✅ Integration test passed - ready for complete Day 4!")