"""
Optimization Manager - Day 3
Integrates all optimization techniques with the benchmarking pipeline.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple, Optional
import logging
from datetime import datetime

from optimization.quantization import QuantizationOptimizer
from optimization.pruning import PruningOptimizer
from optimization.onnx_optimization import ONNXOptimizer
from models.densenet_base import DenseNetBase
from utils.logger import setup_logger


class OptimizationManager:
    """
    Manager for all DenseNet optimization techniques.
    Coordinates quantization, pruning, and ONNX optimizations.
    """
    
    def __init__(self, device: str = "auto", config: Optional[Dict[str, Any]] = None):
        """
        Initialize optimization manager.
        
        Args:
            device: Device to use ('cuda', 'cpu', or 'auto')
            config: Configuration dictionary
        """
        self.logger = setup_logger("OptimizationManager", level=logging.INFO)
        
        # Device setup
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        
        # Configuration
        self.config = config or {}
        
        # Initialize optimizers
        self.quantization_optimizer = QuantizationOptimizer(str(self.device))
        self.pruning_optimizer = PruningOptimizer(str(self.device))
        self.onnx_optimizer = ONNXOptimizer(str(self.device))
        
        self.logger.info(f"Optimization manager initialized for {self.device}")
        
        # Track optimization results
        self.optimization_results = {}
    
    def get_available_techniques(self) -> List[str]:
        """Get list of available optimization techniques."""
        return [
            'quantization_dynamic',
            'quantization_static', 
            'quantization_fp16',
            'pruning_unstructured',
            'pruning_structured',
            'onnx_basic',
            'onnx_extended',
            'onnx_all'
        ]
    
    def apply_quantization(
        self,
        model: nn.Module,
        technique: str = "dynamic",
        calibration_data: Optional[torch.Tensor] = None
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Apply quantization optimization.
        
        Args:
            model: Original model
            technique: 'dynamic', 'static', or 'fp16'
            calibration_data: Calibration data for static quantization
            
        Returns:
            Tuple of (optimized_model, optimization_info)
        """
        self.logger.info(f"Applying {technique} quantization...")
        
        try:
            optimized_model, info = self.quantization_optimizer.optimize_densenet(
                model, technique, calibration_data
            )
            
            # Add benchmark results if successful
            if info.get('success', False):
                test_input = torch.randn(1, 3, 224, 224)
                benchmark_results = self.quantization_optimizer.benchmark_quantized_model(
                    optimized_model, model, test_input,
                    num_iterations=50, technique=f"quantization_{technique}"
                )
                info.update(benchmark_results)
            
            return optimized_model, info
            
        except Exception as e:
            self.logger.error(f"Quantization failed: {e}")
            return model, {
                'technique': f'quantization_{technique}',
                'success': False,
                'error': str(e)
            }
    
    def apply_pruning(
        self,
        model: nn.Module,
        technique: str = "unstructured",
        pruning_amount: float = 0.3
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Apply pruning optimization.
        
        Args:
            model: Original model
            technique: 'unstructured' or 'structured'
            pruning_amount: Fraction of weights to prune
            
        Returns:
            Tuple of (optimized_model, optimization_info)
        """
        self.logger.info(f"Applying {technique} pruning ({pruning_amount*100}% sparsity)...")
        
        try:
            optimized_model, info = self.pruning_optimizer.optimize_densenet(
                model, technique, pruning_amount
            )
            
            # Add benchmark results if successful
            if info.get('success', False):
                test_input = torch.randn(1, 3, 224, 224)
                benchmark_results = self.pruning_optimizer.benchmark_pruned_model(
                    optimized_model, model, test_input,
                    num_iterations=50, technique=f"pruning_{technique}"
                )
                info.update(benchmark_results)
            
            return optimized_model, info
            
        except Exception as e:
            self.logger.error(f"Pruning failed: {e}")
            return model, {
                'technique': f'pruning_{technique}',
                'success': False,
                'error': str(e)
            }
    
    def apply_onnx_optimization(
        self,
        model: nn.Module,
        optimization_level: str = "basic",
        output_dir: Optional[str] = None
    ) -> Tuple[Optional[object], Dict[str, Any]]:
        """
        Apply ONNX optimization.
        
        Args:
            model: Original model
            optimization_level: 'basic', 'extended', or 'all'
            output_dir: Directory to save ONNX models
            
        Returns:
            Tuple of (onnx_session, optimization_info)
        """
        self.logger.info(f"Applying ONNX optimization (level: {optimization_level})...")
        
        try:
            onnx_session, info = self.onnx_optimizer.optimize_densenet(
                model, optimization_level, output_dir=output_dir
            )
            
            # Add benchmark results if successful
            if info.get('success', False) and onnx_session:
                test_input = torch.randn(1, 3, 224, 224)
                benchmark_results = self.onnx_optimizer.benchmark_onnx_model(
                    onnx_session, model, test_input,
                    num_iterations=50, technique=f"onnx_{optimization_level}"
                )
                info.update(benchmark_results)
            
            return onnx_session, info
            
        except Exception as e:
            self.logger.error(f"ONNX optimization failed: {e}")
            return None, {
                'technique': f'onnx_{optimization_level}',
                'success': False,
                'error': str(e)
            }
    
    def run_optimization_suite(
        self,
        model: nn.Module,
        techniques: List[str] = None,
        output_dir: str = "./results/optimizations"
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run a comprehensive optimization suite.
        
        Args:
            model: Original DenseNet model
            techniques: List of techniques to apply
            output_dir: Output directory for models and results
            
        Returns:
            Dictionary mapping technique names to results
        """
        if techniques is None:
            techniques = [
                'quantization_dynamic',
                'pruning_unstructured', 
                'onnx_basic'
            ]
        
        self.logger.info(f"Running optimization suite with {len(techniques)} techniques...")
        
        results = {}
        
        for technique in techniques:
            self.logger.info(f"Processing technique: {technique}")
            
            try:
                if technique.startswith('quantization_'):
                    quant_type = technique.split('_')[1]
                    _, result = self.apply_quantization(model, quant_type)
                    
                elif technique.startswith('pruning_'):
                    prune_type = technique.split('_')[1]
                    _, result = self.apply_pruning(model, prune_type)
                    
                elif technique.startswith('onnx_'):
                    opt_level = technique.split('_')[1]
                    _, result = self.apply_onnx_optimization(model, opt_level, output_dir)
                    
                else:
                    self.logger.warning(f"Unknown technique: {technique}")
                    continue
                
                results[technique] = result
                
                # Log results
                if result.get('success', False):
                    self.logger.info(f"  ✅ {technique} completed successfully")
                    if 'speedup_ratio' in result:
                        self.logger.info(f"     Speedup: {result['speedup_ratio']:.2f}x")
                    if 'size_reduction_percent' in result:
                        self.logger.info(f"     Size reduction: {result['size_reduction_percent']:.1f}%")
                else:
                    self.logger.error(f"  ❌ {technique} failed: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                self.logger.error(f"Failed to process {technique}: {e}")
                results[technique] = {
                    'technique': technique,
                    'success': False,
                    'error': str(e)
                }
        
        self.optimization_results = results
        self.logger.info(f"Optimization suite completed: {len(results)} techniques processed")
        
        return results
    
    def generate_optimization_summary(self) -> Dict[str, Any]:
        """Generate summary of optimization results."""
        if not self.optimization_results:
            return {}
        
        successful = [r for r in self.optimization_results.values() if r.get('success', False)]
        failed = [r for r in self.optimization_results.values() if not r.get('success', False)]
        
        summary = {
            'total_techniques': len(self.optimization_results),
            'successful_techniques': len(successful),
            'failed_techniques': len(failed),
            'techniques_tested': list(self.optimization_results.keys())
        }
        
        if successful:
            # Find best performing optimizations
            speedups = [(r['technique'], r.get('speedup_ratio', 1.0)) for r in successful if 'speedup_ratio' in r]
            size_reductions = [(r['technique'], r.get('size_reduction_percent', 0)) for r in successful if 'size_reduction_percent' in r]
            
            if speedups:
                best_speedup = max(speedups, key=lambda x: x[1])
                summary['best_speedup'] = {
                    'technique': best_speedup[0],
                    'speedup': best_speedup[1]
                }
            
            if size_reductions:
                best_size_reduction = max(size_reductions, key=lambda x: x[1])
                summary['best_size_reduction'] = {
                    'technique': best_size_reduction[0],
                    'reduction_percent': best_size_reduction[1]
                }
        
        return summary
    
    def convert_to_benchmark_results(
        self,
        baseline_model_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Convert optimization results to benchmark results format.
        
        Args:
            baseline_model_info: Information about baseline model
            
        Returns:
            List of benchmark results in standard format
        """
        benchmark_results = []
        
        for technique, result in self.optimization_results.items():
            if not result.get('success', False):
                continue
            
            # Map optimization result to benchmark format
            benchmark_result = {
                'model_variant': f"densenet121_{technique}",
                'device': str(self.device),
                'optimization_technique': technique,
                'model_size_mb': result.get('model_size_mb', baseline_model_info.get('model_size_mb', 0)),
                'timestamp': datetime.now().isoformat()
            }
            
            # Add performance metrics if available
            if 'speedup_ratio' in result:
                # Calculate latency from speedup
                baseline_latency = result.get('original_latency_ms') or result.get('pytorch_latency_ms', 100)
                optimized_latency = baseline_latency / result['speedup_ratio']
                
                benchmark_result.update({
                    'mean_latency_ms': optimized_latency,
                    'throughput_samples_sec': 1000 / optimized_latency
                })
            
            # Add size reduction info
            if 'size_reduction_percent' in result:
                benchmark_result['size_reduction_percent'] = result['size_reduction_percent']
            
            # Add sparsity info for pruning
            if 'sparsity_percent' in result:
                benchmark_result['sparsity_percent'] = result['sparsity_percent']
            
            # Placeholder for accuracy (to be filled by validation)
            benchmark_result.update({
                'accuracy_top1': None,
                'accuracy_top5': None
            })
            
            benchmark_results.append(benchmark_result)
        
        return benchmark_results


def test_optimization_manager():
    """Test the optimization manager with a simple example."""
    from torchvision import models
    
    # Load baseline model
    model = models.densenet121(pretrained=True)
    model.eval()
    
    # Create optimization manager
    manager = OptimizationManager()
    
    # Run optimization suite
    results = manager.run_optimization_suite(
        model,
        techniques=['quantization_dynamic', 'pruning_unstructured'],
        output_dir="./test_optimizations"
    )
    
    # Generate summary
    summary = manager.generate_optimization_summary()
    
    print("Optimization Results:")
    for technique, result in results.items():
        status = "✅" if result.get('success', False) else "❌"
        print(f"  {status} {technique}")
        if result.get('success', False) and 'speedup_ratio' in result:
            print(f"      Speedup: {result['speedup_ratio']:.2f}x")
    
    print(f"\nSummary: {summary}")


if __name__ == "__main__":
    test_optimization_manager()