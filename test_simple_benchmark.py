"""Test simplified benchmark runner."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from benchmarking.simple_benchmark_runner import SimpleBenchmarkRunner
from utils.results_exporter import ResultsExporter

def test_simple_benchmark():
    config = {
        'benchmark': {
            'batch_sizes': [1, 4],
            'num_iterations': 10
        },
        'optimization': {
            'techniques': ['quantization_dynamic', 'pruning_unstructured']
        }
    }
    
    runner = SimpleBenchmarkRunner(config)
    results = runner.run_complete_benchmark()
    
    print(f"Total results: {len(results)}")
    
    # Export results
    exporter = ResultsExporter()
    csv_path = exporter.export_to_csv(results, './simple_benchmark_results.csv')
    print(f"Results exported to: {csv_path}")
    
    return results

if __name__ == "__main__":
    test_simple_benchmark()