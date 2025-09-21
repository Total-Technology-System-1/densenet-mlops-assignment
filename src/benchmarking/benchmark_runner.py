"""
Core Benchmarking Runner - Day 2
Orchestrates all benchmarking operations for DenseNet optimization.
"""

import os
import sys
import time
import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.densenet_base import DenseNetBase
from benchmarking.profiler import PyTorchProfilerWrapper
from benchmarking.metrics_collector import MetricsCollector
from benchmarking.tensorboard_logger import TensorBoardLogger
from utils.device_utils import detect_environment, get_memory_info
from utils.logger import setup_logger


class BenchmarkRunner:
    """
    Main benchmarking orchestrator for DenseNet models.
    
    This class coordinates:
    - Model initialization and warmup
    - PyTorch profiling across different batch sizes
    - Memory and performance metrics collection
    - TensorBoard logging and visualization
    - Results aggregation and export
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the benchmark runner.
        
        Args:
            config: Configuration dictionary with benchmarking parameters
        """
        self.config = config
        self.logger = setup_logger("BenchmarkRunner", level=logging.INFO)
        
        # Initialize components
        self.environment = detect_environment()
        self.device = torch.device(self.environment['device'])
        
        # Setup paths
        self.results_dir = Path(config.get('output_dir', './results'))
        self.logs_dir = Path(config.get('directories', {}).get('logs', './logs/tensorboard'))
        
        # Create directories
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging and profiling components
        self.tensorboard_logger = TensorBoardLogger(str(self.logs_dir))
        self.metrics_collector = MetricsCollector()
        
        # Benchmarking parameters
        self.batch_sizes = config.get('batch_sizes', [1, 4, 8, 16, 32])
        self.num_iterations = config.get('num_iterations', 50)
        self.warmup_iterations = config.get('warmup_iterations', 10)
        
        # Results storage
        self.all_results = []
        
        self.logger.info(f"🚀 BenchmarkRunner initialized")
        self.logger.info(f"   Device: {self.device}")
        self.logger.info(f"   Batch sizes: {self.batch_sizes}")
        self.logger.info(f"   Iterations: {self.num_iterations}")
    
    def _log_system_info(self) -> None:
        """Log comprehensive system information."""
        self.logger.info("="*60)
        self.logger.info("SYSTEM INFORMATION")
        self.logger.info("="*60)
        
        env = self.environment
        self.logger.info(f"Platform: {env.get('platform', 'Unknown')}")
        self.logger.info(f"Python: {env.get('python_version', 'Unknown')}")
        self.logger.info(f"PyTorch: {env.get('pytorch_version', 'Unknown')}")
        self.logger.info(f"Device: {env['device']} ({env.get('device_name', 'Unknown')})")
        
        if env.get('cuda_available'):
            self.logger.info(f"CUDA Version: {env.get('cuda_version', 'Unknown')}")
            self.logger.info(f"GPU Count: {env.get('gpu_count', 0)}")
            for i, name in enumerate(env.get('gpu_names', [])):
                memory = env.get('gpu_memory_gb', [])[i] if i < len(env.get('gpu_memory_gb', [])) else 'Unknown'
                self.logger.info(f"  GPU {i}: {name} ({memory} GB)")
        
        memory_info = get_memory_info()
        self.logger.info(f"RAM: {memory_info.get('system_available_gb', 0):.1f} GB available")
        
        self.logger.info("="*60)
    
    def run_baseline_benchmark(self) -> List[Dict[str, Any]]:
        """
        Run baseline DenseNet-121 benchmarking across all batch sizes.
        
        Returns:
            List of benchmark results for each batch size
        """
        self.logger.info("🔍 Starting Baseline DenseNet-121 Benchmarking")
        
        # Initialize model
        model = DenseNetBase(device=str(self.device))
        model_info = model.get_model_info()
        
        baseline_results = []
        
        for batch_size in self.batch_sizes:
            self.logger.info(f"📊 Benchmarking batch size: {batch_size}")
            
            try:
                # Run detailed profiling for this batch size
                result = self._benchmark_single_batch_size(
                    model, batch_size, "baseline", model_info
                )
                baseline_results.append(result)
                
                # Log to TensorBoard
                self._log_to_tensorboard(result, "baseline")
                
                self.logger.info(f"   ✅ Completed: {result['mean_latency_ms']:.3f}ms avg latency")
                
            except Exception as e:
                self.logger.error(f"   ❌ Failed batch size {batch_size}: {e}")
                # Create error result
                error_result = self._create_error_result(batch_size, "baseline", str(e), model_info)
                baseline_results.append(error_result)
        
        self.logger.info(f"✅ Baseline benchmarking complete: {len(baseline_results)} results")
        return baseline_results
    
    def _benchmark_single_batch_size(
        self, 
        model: DenseNetBase, 
        batch_size: int, 
        optimization_technique: str,
        model_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Benchmark a single batch size with comprehensive profiling.
        
        Args:
            model: DenseNet model to benchmark
            batch_size: Batch size to test
            optimization_technique: Name of optimization technique
            model_info: Model information dictionary
            
        Returns:
            Comprehensive benchmark results
        """
        # Prepare input data
        dummy_input = torch.randn(batch_size, 3, 224, 224, device=self.device)
        
        # Warmup
        model.warm_up(batch_size, self.warmup_iterations)
        
        # Clear memory and prepare for profiling
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        # Initialize metrics collection
        self.metrics_collector.start_monitoring()
        
        # Profile with PyTorch Profiler
        profiler_wrapper = PyTorchProfilerWrapper(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if self.device.type == 'cuda' else [ProfilerActivity.CPU],
            record_shapes=True,
            with_stack=True
        )
        
        latencies = []
        memory_snapshots = []
        
        with profiler_wrapper as prof:
            for i in range(self.num_iterations):
                # Memory snapshot before
                memory_before = get_memory_info()
                
                # Timed inference
                with record_function(f"inference_batch_{batch_size}_iter_{i}"):
                    predictions, latency = model.single_inference(dummy_input)
                    latencies.append(latency)
                
                # Memory snapshot after
                memory_after = get_memory_info()
                memory_snapshots.append({
                    'before': memory_before,
                    'after': memory_after,
                    'iteration': i
                })
                
                # Step profiler
                prof.step()
        
        # Stop monitoring
        final_metrics = self.metrics_collector.stop_monitoring()
        
        # Calculate statistics
        import statistics
        
        mean_latency = statistics.mean(latencies)
        throughput = batch_size * 1000 / mean_latency  # samples/second
        
        # Get memory statistics
        memory_stats = self._analyze_memory_usage(memory_snapshots)
        
        # Create comprehensive result
        result = {
            # Basic info
            'model_variant': f"densenet121_{optimization_technique}",
            'batch_size': batch_size,
            'device': str(self.device),
            'optimization_technique': optimization_technique,
            
            # Performance metrics
            'mean_latency_ms': round(mean_latency, 3),
            'median_latency_ms': round(statistics.median(latencies), 3),
            'min_latency_ms': round(min(latencies), 3),
            'max_latency_ms': round(max(latencies), 3),
            'std_latency_ms': round(statistics.stdev(latencies) if len(latencies) > 1 else 0.0, 3),
            'throughput_samples_sec': round(throughput, 2),
            
            # Memory metrics
            'ram_usage_mb': memory_stats['peak_ram_mb'],
            'vram_usage_mb': memory_stats.get('peak_vram_mb', 0),
            
            # System metrics
            'cpu_utilization_pct': final_metrics.get('avg_cpu_percent', 0),
            'gpu_utilization_pct': final_metrics.get('avg_gpu_percent', 0),
            
            # Model info
            'model_size_mb': model_info['model_size_mb'],
            'total_parameters': model_info['total_parameters'],
            
            # Profiling info
            'iterations': self.num_iterations,
            'warmup_iterations': self.warmup_iterations,
            'timestamp': datetime.now().isoformat(),
            
            # Accuracy placeholders (to be filled by validation)
            'accuracy_top1': None,
            'accuracy_top5': None
        }
        
        # Save detailed profiling results
        self._save_profiler_results(prof, batch_size, optimization_technique)
        
        return result
    
    def _analyze_memory_usage(self, memory_snapshots: List[Dict]) -> Dict[str, float]:
        """Analyze memory usage patterns from snapshots."""
        if not memory_snapshots:
            return {'peak_ram_mb': 0, 'peak_vram_mb': 0}
        
        ram_usage = [snap['after']['process_ram_mb'] for snap in memory_snapshots]
        vram_usage = [snap['after'].get('gpu_allocated_mb', 0) for snap in memory_snapshots]
        
        return {
            'peak_ram_mb': round(max(ram_usage), 2),
            'peak_vram_mb': round(max(vram_usage), 2),
            'avg_ram_mb': round(sum(ram_usage) / len(ram_usage), 2),
            'avg_vram_mb': round(sum(vram_usage) / len(vram_usage), 2)
        }
    
    def _save_profiler_results(
        self, 
        prof: torch.profiler.profile, 
        batch_size: int, 
        optimization_technique: str
    ) -> None:
        """Save PyTorch profiler results to files."""
        profile_dir = self.results_dir / "profiles" / optimization_technique
        profile_dir.mkdir(parents=True, exist_ok=True)
        
        # Save trace for Chrome tracing
        trace_path = profile_dir / f"trace_batch_{batch_size}.json"
        prof.export_chrome_trace(str(trace_path))
        
        # Save stacks for analysis
        stacks_path = profile_dir / f"stacks_batch_{batch_size}.txt"
        with open(stacks_path, 'w') as f:
            f.write(prof.key_averages(group_by_stack_n=5).table(
                sort_by="self_cuda_time_total" if self.device.type == 'cuda' else "self_cpu_time_total",
                row_limit=50
            ))
        
        self.logger.info(f"   📁 Profiler results saved: {profile_dir}")
    
    def _log_to_tensorboard(self, result: Dict[str, Any], optimization_technique: str) -> None:
        """Log results to TensorBoard."""
        step = result['batch_size']  # Use batch size as step
        
        # Performance metrics
        self.tensorboard_logger.log_scalar(
            f"{optimization_technique}/latency_ms", result['mean_latency_ms'], step
        )
        self.tensorboard_logger.log_scalar(
            f"{optimization_technique}/throughput_samples_sec", result['throughput_samples_sec'], step
        )
        
        # Memory metrics
        self.tensorboard_logger.log_scalar(
            f"{optimization_technique}/ram_usage_mb", result['ram_usage_mb'], step
        )
        if result['vram_usage_mb'] > 0:
            self.tensorboard_logger.log_scalar(
                f"{optimization_technique}/vram_usage_mb", result['vram_usage_mb'], step
            )
        
        # System metrics
        self.tensorboard_logger.log_scalar(
            f"{optimization_technique}/cpu_utilization_pct", result['cpu_utilization_pct'], step
        )
        if result['gpu_utilization_pct'] > 0:
            self.tensorboard_logger.log_scalar(
                f"{optimization_technique}/gpu_utilization_pct", result['gpu_utilization_pct'], step
            )
    
    def _create_error_result(
        self, 
        batch_size: int, 
        optimization_technique: str, 
        error_msg: str,
        model_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create error result entry."""
        return {
            'model_variant': f"densenet121_{optimization_technique}",
            'batch_size': batch_size,
            'device': str(self.device),
            'optimization_technique': optimization_technique,
            'error': error_msg,
            'model_size_mb': model_info.get('model_size_mb', 0),
            'timestamp': datetime.now().isoformat()
        }
    
    def run_all_benchmarks(self) -> List[Dict[str, Any]]:
        """
        Run complete benchmarking suite.
        
        Returns:
            List of all benchmark results
        """
        self.logger.info("🚀 Starting Complete Benchmarking Suite")
        self._log_system_info()
        
        all_results = []
        
        try:
            # Run baseline benchmarking
            baseline_results = self.run_baseline_benchmark()
            all_results.extend(baseline_results)
            
            # TODO: Day 5-6 will add optimization benchmarks here
            # For now, we only have baseline
            
            # Store results
            self.all_results = all_results
            
            self.logger.info(f"🎉 Benchmarking suite completed!")
            self.logger.info(f"   Total results: {len(all_results)}")
            
            return all_results
            
        except Exception as e:
            self.logger.error(f"❌ Benchmarking suite failed: {e}")
            raise
        
        finally:
            # Cleanup
            self.tensorboard_logger.close()
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics from all benchmarking results."""
        if not self.all_results:
            return {}
        
        # Filter successful results
        successful_results = [r for r in self.all_results if 'error' not in r]
        
        if not successful_results:
            return {'error': 'No successful benchmark results'}
        
        # Calculate summary statistics
        latencies = [r['mean_latency_ms'] for r in successful_results]
        throughputs = [r['throughput_samples_sec'] for r in successful_results]
        batch_sizes = [r['batch_size'] for r in successful_results]
        
        import statistics
        
        return {
            'total_benchmarks': len(self.all_results),
            'successful_benchmarks': len(successful_results),
            'batch_sizes_tested': sorted(set(batch_sizes)),
            'latency_stats': {
                'min': min(latencies),
                'max': max(latencies),
                'mean': statistics.mean(latencies),
                'median': statistics.median(latencies)
            },
            'throughput_stats': {
                'min': min(throughputs),
                'max': max(throughputs),
                'mean': statistics.mean(throughputs),
                'median': statistics.median(throughputs)
            },
            'best_latency': min(latencies),
            'best_throughput': max(throughputs),
            'optimal_batch_size': successful_results[throughputs.index(max(throughputs))]['batch_size']
        }