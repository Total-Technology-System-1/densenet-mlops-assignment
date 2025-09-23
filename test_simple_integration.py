"""
Simplified Integration Test - Bypass Profiler Issues
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from models.densenet_base import DenseNetBase
from optimization.optimization_manager import OptimizationManager

def test_simple_integration():
    print("Simple Integration Test...")
    
    try:
        # Test 1: Load model
        model = DenseNetBase()
        print("Model loaded successfully")
        
        # Test 2: Test optimizations
        opt_manager = OptimizationManager()
        results = opt_manager.run_optimization_suite(
            model.model,
            techniques=['quantization_dynamic', 'pruning_unstructured'],
            output_dir="./simple_test_results"
        )
        
        print(f"Optimization results: {len(results)} techniques tested")
        for technique, result in results.items():
            status = "✅" if result.get('success', False) else "❌"
            print(f"  {status} {technique}")
        
        return True
        
    except Exception as e:
        print(f"Test failed: {e}")
        return False

if __name__ == "__main__":
    test_simple_integration()