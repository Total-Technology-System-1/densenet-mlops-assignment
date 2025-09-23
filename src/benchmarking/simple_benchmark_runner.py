"""
Simplified Benchmark Runner - Day 4
Bypasses profiler issues while integrating optimizations.
"""

import time
import logging
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

import torch
from models.densenet_base import DenseNetBase
from optimization.optimization_manager import OptimizationManager
from utils.device_utils import detect_environment, get_memory_info
from utils.logger import setup_logger


class SimpleBenchmarkRunner:
    """Simplified benchmark runner focusing on optimization integration."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = setup_logger("SimpleBenchmarkRunner", level=logging.INFO)
        
        # Setup environment
        self.environment = detect_environment()
        self.device = torch.device(self.environment['device'])
        
        # Benchmarking parameters
        self.batch_sizes = config.get('benchmark', {}).get('batch_sizes', [1, 4, 8])
        self.num_iterations = config.get('benchmark', {}).get('num_iterations', 20)
        
        # Results storage
        self.all_results = []
        
        self.logger.info(f"SimpleBenchmarkRunner initialized for {self.device}")
    
    def run_baseline_benchmark(self) -> List[Dict[str, Any]]:
        """Run baseline DenseNet benchmarking."""
        self.logger.info("Running baseline benchmarking...")
        
        baseline_model = DenseNetBase(device=str(self.device))
        results = []
        
        for batch_size in self.batch_sizes:
            try:
                result = self._benchmark_model(
                    baseline_model.model, batch_size, "baseline"
                )
                results.append(result)
                self.logger.info(f"  Baseline batch {batch_size}: {result['mean_latency_ms']:.3f}ms")
                
            except Exception as e:
                self.logger.error(f"  Baseline batch {batch_size} failed: {e}")
        
        return results
    
    def run_optimization_benchmarks(self) -> List[Dict[str, Any]]:
        """Run optimization benchmarking."""
        self.logger.info("Running optimization benchmarking...")
        
        # Initialize optimization manager
        opt_manager = OptimizationManager(device=str(self.device))
        baseline_model = DenseNetBase(device=str(self.device))
        
        # Get techniques from config
        techniques = self.config.get('optimization', {}).get('techniques', 
                                                           ['quantization_dynamic', 'pruning_unstructured'])
        
        results = []
        
        for technique in techniques:
            self.logger.info(f"  Processing {technique}...")
            
            try:
                # Apply optimization
                if technique.startswith('quantization_'):
                    quant_type = technique.split('_')[1]
                    optimized_model, opt_info = opt_manager.apply_quantization(
                        baseline_model.model, quant_type
                    )
                elif technique.startswith('pruning_'):
                    prune_type = technique.split('_')[1]
                    optimized_model, opt_info = opt_manager.apply_pruning(
                        baseline_model.model, prune_type
                    )
                else:
                    continue
                
                # Benchmark optimized model
                if opt_info.get('success', False):
                    for batch_size in self.batch_sizes:
                        try:
                            result = self._benchmark_model(
                                optimized_model, batch_size, technique
                            )
                            
                            # Add optimization metrics
                            if 'size_reduction_percent' in opt_info:
                                result['size_reduction_percent'] = opt_info['size_reduction_percent']
                            if 'speedup_ratio' in opt_info:
                                result['speedup_ratio'] = opt_info['speedup_ratio']
                            
                            results.append(result)
                            self.logger.info(f"    {technique} batch {batch_size}: {result['mean_latency_ms']:.3f}ms")
                            
                        except Exception as e:
                            self.logger.error(f"    {technique} batch {batch_size} failed: {e}")
                
            except Exception as e:
                self.logger.error(f"  {technique} failed: {e}")
        
        return results
    
    def _benchmark_model(self, model, batch_size: int, technique: str) -> Dict[str, Any]:
        """Benchmark a single model configuration."""
        # Prepare model and input
        model = model.to(self.device)
        model.eval()
        test_input = torch.randn(batch_size, 3, 224, 224, device=self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = model(test_input)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
        
        # Benchmark
        latencies = []
        with torch.no_grad():
            for _ in range(self.num_iterations):
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                start_time = time.perf_counter()
                _ = model(test_input)
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                latencies.append((time.perf_counter() - start_time) * 1000)
        
        # Calculate metrics
        mean_latency = sum(latencies) / len(latencies)
        throughput = batch_size * 1000 / mean_latency
        memory_info = get_memory_info()
        
        # Calculate model size
        model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
        
        return {
            'model_variant': f'densenet121_{technique}',
            'batch_size': batch_size,
            'device': str(self.device),
            'optimization_technique': technique,
            'mean_latency_ms': round(mean_latency, 3),
            'throughput_samples_sec': round(throughput, 2),
            'model_size_mb': round(model_size_mb, 2),
            'ram_usage_mb': memory_info.get('process_ram_mb', 0),
            'vram_usage_mb': memory_info.get('gpu_allocated_mb', 0),
            'cpu_utilization_pct': memory_info.get('cpu_percent', 0),
            'gpu_utilization_pct': 0,  # Placeholder
            'accuracy_top1': None,     # Placeholder
            'accuracy_top5': None,     # Placeholder
            'iterations': len(latencies),
            'timestamp': datetime.now().isoformat()
        }
    
    def run_complete_benchmark(self) -> List[Dict[str, Any]]:
        """Run complete benchmarking suite."""
        self.logger.info("Starting complete benchmarking suite...")
        
        # Run baseline
        baseline_results = self.run_baseline_benchmark()
        
        # Run optimizations
        optimization_results = self.run_optimization_benchmarks()
        
        # Combine results
        all_results = baseline_results + optimization_results
        self.all_results = all_results
        
        self.logger.info(f"Complete benchmark finished:")
        self.logger.info(f"  Baseline results: {len(baseline_results)}")
        self.logger.info(f"  Optimization results: {len(optimization_results)}")
        self.logger.info(f"  Total results: {len(all_results)}")
        
        return all_results