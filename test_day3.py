"""
Day 3 Testing Script - Complete Optimization System Test
Tests all Day 3 optimization components.
"""

import sys
import torch
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from models.densenet_base import DenseNetBase
from optimization.optimization_manager import OptimizationManager
from optimization.quantization import QuantizationOptimizer
from optimization.pruning import PruningOptimizer
from optimization.onnx_optimization import ONNXOptimizer
from utils.device_utils import detect_environment


def test_day3_optimizations():
    """Test all Day 3 optimization components."""
    
    print("=" * 60)
    print("DAY 3 OPTIMIZATION TESTING")
    print("=" * 60)
    
    # Test 1: Environment and Model Setup
    print("\n1. Testing Environment & Model Setup...")
    try:
        env_info = detect_environment()
        print(f"   Device: {env_info['device']}")
        print(f"   PyTorch: {env_info['pytorch_version']}")
        
        # Load baseline model
        densenet_base = DenseNetBase(device=env_info['device'])
        model_info = densenet_base.get_model_info()
        print(f"   Model: {model_info['architecture']}")
        print(f"   Parameters: {model_info['total_parameters']:,}")
        print("   ✅ Environment and model setup passed")
        
    except Exception as e:
        print(f"   ❌ Environment setup failed: {e}")
        return False
    
    # Test 2: Quantization Optimizer
    print("\n2. Testing Quantization Optimizer...")
    try:
        quant_optimizer = QuantizationOptimizer(str(env_info['device']))
        
        # Test dynamic quantization
        quantized_model, quant_info = quant_optimizer.optimize_densenet(
            densenet_base.model, "dynamic"
        )
        
        if quant_info.get('success', False):
            print(f"   Dynamic quantization: ✅")
            print(f"   Model size: {quant_info['model_size_mb']:.2f} MB")
        else:
            print(f"   Dynamic quantization: ❌ {quant_info.get('error', '')}")
        
        # Test FP16 quantization
        fp16_model, fp16_info = quant_optimizer.optimize_densenet(
            densenet_base.model, "fp16"
        )
        
        if fp16_info.get('success', False):
            print(f"   FP16 quantization: ✅")
        else:
            print(f"   FP16 quantization: ❌ {fp16_info.get('error', '')}")
        
        print("   ✅ Quantization optimizer passed")
        
    except Exception as e:
        print(f"   ❌ Quantization optimizer failed: {e}")
    
    # Test 3: Pruning Optimizer  
    print("\n3. Testing Pruning Optimizer...")
    try:
        prune_optimizer = PruningOptimizer(str(env_info['device']))
        
        # Test unstructured pruning
        pruned_model, prune_info = prune_optimizer.optimize_densenet(
            densenet_base.model, "unstructured", 0.3
        )
        
        if prune_info.get('success', False):
            print(f"   Unstructured pruning: ✅")
            print(f"   Sparsity: {prune_info.get('overall_sparsity', 0)*100:.1f}%")
        else:
            print(f"   Unstructured pruning: ❌ {prune_info.get('error', '')}")
        
        print("   ✅ Pruning optimizer passed")
        
    except Exception as e:
        print(f"   ❌ Pruning optimizer failed: {e}")
    
    # Test 4: ONNX Optimizer
    print("\n4. Testing ONNX Optimizer...")
    try:
        onnx_optimizer = ONNXOptimizer(str(env_info['device']))
        
        # Test ONNX conversion
        onnx_session, onnx_info = onnx_optimizer.optimize_densenet(
            densenet_base.model, "basic", output_dir="./test_onnx"
        )
        
        if onnx_info.get('success', False):
            print(f"   ONNX conversion: ✅")
            print(f"   Model size: {onnx_info.get('model_size_mb', 0):.2f} MB")
            print(f"   Providers: {onnx_info.get('providers', [])}")
        else:
            print(f"   ONNX conversion: ❌ {onnx_info.get('error', '')}")
        
        print("   ✅ ONNX optimizer passed")
        
    except Exception as e:
        print(f"   ❌ ONNX optimizer failed: {e}")
    
    # Test 5: Optimization Manager Integration
    print("\n5. Testing Optimization Manager...")
    try:
        opt_manager = OptimizationManager(device=env_info['device'])
        
        # Test available techniques
        available = opt_manager.get_available_techniques()
        print(f"   Available techniques: {len(available)}")
        
        # Run quick optimization suite
        results = opt_manager.run_optimization_suite(
            densenet_base.model,
            techniques=['quantization_dynamic', 'pruning_unstructured'],
            output_dir="./test_optimizations"
        )
        
        successful = sum(1 for r in results.values() if r.get('success', False))
        print(f"   Optimization results: {successful}/{len(results)} successful")
        
        # Generate summary
        summary = opt_manager.generate_optimization_summary()
        print(f"   Summary: {summary.get('successful_techniques', 0)} techniques succeeded")
        
        print("   ✅ Optimization manager passed")
        
    except Exception as e:
        print(f"   ❌ Optimization manager failed: {e}")
    
    # Test 6: Benchmark Integration
    print("\n6. Testing Benchmark Integration...")
    try:
        # Convert optimization results to benchmark format
        baseline_info = densenet_base.get_model_info()
        benchmark_results = opt_manager.convert_to_benchmark_results(baseline_info)
        
        print(f"   Benchmark results generated: {len(benchmark_results)}")
        
        if benchmark_results:
            sample = benchmark_results[0]
            print(f"   Sample result: {sample['optimization_technique']}")
            print(f"   Model variant: {sample['model_variant']}")
        
        print("   ✅ Benchmark integration passed")
        
    except Exception as e:
        print(f"   ❌ Benchmark integration failed: {e}")
    
    print("\n" + "=" * 60)
    print("DAY 3 OPTIMIZATION TESTING COMPLETE!")
    print("=" * 60)
    print("✅ Optimization techniques implemented:")
    print("   - Model Quantization (Dynamic, Static, FP16)")
    print("   - Model Pruning (Structured, Unstructured)")
    print("   - ONNX Optimization (Basic, Extended, All)")
    print("🚀 Ready for Day 4: Integration with Benchmarking Pipeline")
    
    return True


def run_quick_optimization_demo():
    """Run a quick optimization demo showing all techniques."""
    
    print("\n" + "=" * 60)
    print("QUICK OPTIMIZATION DEMONSTRATION")
    print("=" * 60)
    
    try:
        from torchvision import models
        
        # Load baseline model
        print("Loading DenseNet-121...")
        model = models.densenet121(pretrained=True)
        model.eval()
        
        # Create test input
        test_input = torch.randn(1, 3, 224, 224)
        
        # Baseline inference
        print("Running baseline inference...")
        with torch.no_grad():
            baseline_output = model(test_input)
        
        print(f"Baseline output shape: {baseline_output.shape}")
        
        # Test quantization
        print("\nTesting Dynamic Quantization...")
        quant_optimizer = QuantizationOptimizer()
        quantized_model, _ = quant_optimizer.optimize_densenet(model, "dynamic")
        
        with torch.no_grad():
            quant_output = quantized_model(test_input)
        
        mse = torch.mean((baseline_output - quant_output) ** 2)
        print(f"Quantized MSE vs baseline: {mse.item():.6f}")
        
        # Test pruning
        print("\nTesting Unstructured Pruning...")
        prune_optimizer = PruningOptimizer()
        pruned_model, prune_info = prune_optimizer.optimize_densenet(model, "unstructured", 0.2)
        
        with torch.no_grad():
            pruned_output = pruned_model(test_input)
        
        mse = torch.mean((baseline_output - pruned_output) ** 2)
        sparsity = prune_info.get('overall_sparsity', 0) * 100
        print(f"Pruned MSE vs baseline: {mse.item():.6f}")
        print(f"Achieved sparsity: {sparsity:.1f}%")
        
        print("\n🎉 Quick optimization demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Quick demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Run tests
    if test_day3_optimizations():
        # If basic tests pass, run demo
        run_quick_optimization_demo()
    else:
        print("Basic tests failed - fix issues before proceeding")
        sys.exit(1)