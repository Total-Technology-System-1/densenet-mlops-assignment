"""
TensorBoard Logger - Day 2
Comprehensive TensorBoard integration for benchmarking visualization.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    """
    TensorBoard integration for MLOps benchmarking visualization.
    
    Provides:
    - Scalar metrics logging (latency, throughput, memory usage)
    - Histogram logging for performance distributions
    - Model graph visualization
    - Custom dashboard layouts
    - Profiler integration
    """
    
    def __init__(
        self,
        log_dir: str = "./logs/tensorboard",
        experiment_name: Optional[str] = None,
        comment: str = "",
        flush_secs: int = 10
    ):
        """
        Initialize TensorBoard logger.
        
        Args:
            log_dir: Base directory for TensorBoard logs
            experiment_name: Name of the experiment (auto-generated if None)
            comment: Additional comment for the run
            flush_secs: How often to flush data to disk
        """
        self.logger = logging.getLogger(__name__)
        
        # Generate experiment name if not provided
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"densenet_benchmark_{timestamp}"
        
        # Setup log directory
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize SummaryWriter
        self.writer = SummaryWriter(
            log_dir=str(self.log_dir),
            comment=comment,
            flush_secs=flush_secs
        )
        
        # Track logging state
        self.is_active = True
        self.global_step = 0
        
        self.logger.info(f"📊 TensorBoard logger initialized")
        self.logger.info(f"   Log directory: {self.log_dir}")
        self.logger.info(f"   Experiment: {experiment_name}")
        
        # Log system information
        self._log_system_info()
    
    def _log_system_info(self) -> None:
        """Log system information as text."""
        try:
            import platform
            import psutil
            from utils.device_utils import detect_environment
            
            # Get system info
            env_info = detect_environment()
            
            system_info = f"""
# System Information

## Hardware
- **Platform**: {env_info.get('platform', 'Unknown')}
- **CPU Cores**: {env_info.get('cpu_count', 'Unknown')}
- **Total RAM**: {env_info.get('total_ram_gb', 'Unknown')} GB
- **GPU**: {env_info.get('current_gpu_name', 'None')} ({env_info.get('current_gpu_memory_gb', 0)} GB)

## Software
- **Python**: {env_info.get('python_version', 'Unknown')}
- **PyTorch**: {env_info.get('pytorch_version', 'Unknown')}
- **CUDA**: {env_info.get('cuda_version', 'Not available')}

## Environment
- **Execution**: {', '.join([k for k, v in env_info.items() if k.startswith('is_') and v])}
- **Device**: {env_info.get('device', 'Unknown')}

---
*Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
            """
            
            self.writer.add_text("System/Info", system_info, 0)
            
        except Exception as e:
            self.logger.warning(f"Failed to log system info: {e}")
    
    def log_scalar(
        self, 
        tag: str, 
        value: Union[float, int, torch.Tensor], 
        step: Optional[int] = None
    ) -> None:
        """
        Log a scalar value.
        
        Args:
            tag: Data identifier (e.g., 'baseline/latency_ms')
            value: Scalar value to log
            step: Global step value (auto-incremented if None)
        """
        if not self.is_active:
            return
        
        if step is None:
            step = self.global_step
            self.global_step += 1
        
        try:
            # Convert tensor to scalar if needed
            if isinstance(value, torch.Tensor):
                value = value.item()
            
            self.writer.add_scalar(tag, value, step)
            
        except Exception as e:
            self.logger.error(f"Failed to log scalar {tag}: {e}")
    
    def log_scalars(
        self, 
        main_tag: str, 
        tag_scalar_dict: Dict[str, Union[float, int]], 
        step: Optional[int] = None
    ) -> None:
        """
        Log multiple related scalars.
        
        Args:
            main_tag: Main tag for the scalar group
            tag_scalar_dict: Dictionary of tag-value pairs
            step: Global step value
        """
        if not self.is_active:
            return
        
        if step is None:
            step = self.global_step
            self.global_step += 1
        
        try:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)
        except Exception as e:
            self.logger.error(f"Failed to log scalars {main_tag}: {e}")
    
    def log_histogram(
        self, 
        tag: str, 
        values: Union[torch.Tensor, np.ndarray, List[float]], 
        step: Optional[int] = None,
        bins: str = 'tensorflow'
    ) -> None:
        """
        Log a histogram of values.
        
        Args:
            tag: Data identifier
            values: Array of values for histogram
            step: Global step value
            bins: Binning method ('tensorflow', 'auto', 'fd', etc.)
        """
        if not self.is_active:
            return
        
        if step is None:
            step = self.global_step
            self.global_step += 1
        
        try:
            # Convert to numpy array if needed
            if isinstance(values, torch.Tensor):
                values = values.detach().cpu().numpy()
            elif isinstance(values, list):
                values = np.array(values)
            
            self.writer.add_histogram(tag, values, step, bins=bins)
            
        except Exception as e:
            self.logger.error(f"Failed to log histogram {tag}: {e}")
    
    def log_benchmark_results(
        self, 
        results: Dict[str, Any], 
        optimization_technique: str = "baseline"
    ) -> None:
        """
        Log comprehensive benchmark results.
        
        Args:
            results: Dictionary with benchmark results
            optimization_technique: Name of the optimization technique
        """
        if not self.is_active:
            return
        
        try:
            batch_size = results.get('batch_size', 0)
            
            # Performance metrics
            if 'mean_latency_ms' in results:
                self.log_scalar(f"{optimization_technique}/latency_ms", 
                              results['mean_latency_ms'], batch_size)
            
            if 'throughput_samples_sec' in results:
                self.log_scalar(f"{optimization_technique}/throughput_samples_sec", 
                              results['throughput_samples_sec'], batch_size)
            
            # Memory metrics
            if 'ram_usage_mb' in results:
                self.log_scalar(f"{optimization_technique}/memory/ram_usage_mb", 
                              results['ram_usage_mb'], batch_size)
            
            if 'vram_usage_mb' in results and results['vram_usage_mb'] > 0:
                self.log_scalar(f"{optimization_technique}/memory/vram_usage_mb", 
                              results['vram_usage_mb'], batch_size)
            
            # System utilization
            if 'cpu_utilization_pct' in results:
                self.log_scalar(f"{optimization_technique}/utilization/cpu_percent", 
                              results['cpu_utilization_pct'], batch_size)
            
            if 'gpu_utilization_pct' in results and results['gpu_utilization_pct'] > 0:
                self.log_scalar(f"{optimization_technique}/utilization/gpu_percent", 
                              results['gpu_utilization_pct'], batch_size)
            
            # Model metrics
            if 'model_size_mb' in results:
                self.log_scalar(f"{optimization_technique}/model/size_mb", 
                              results['model_size_mb'], batch_size)
            
            # Accuracy metrics (if available)
            if 'accuracy_top1' in results and results['accuracy_top1'] is not None:
                self.log_scalar(f"{optimization_technique}/accuracy/top1_percent", 
                              results['accuracy_top1'], batch_size)
            
            if 'accuracy_top5' in results and results['accuracy_top5'] is not None:
                self.log_scalar(f"{optimization_technique}/accuracy/top5_percent", 
                              results['accuracy_top5'], batch_size)
            
            # Log as grouped scalars for comparison
            performance_metrics = {}
            if 'mean_latency_ms' in results:
                performance_metrics['latency_ms'] = results['mean_latency_ms']
            if 'throughput_samples_sec' in results:
                performance_metrics['throughput'] = results['throughput_samples_sec']
            
            if performance_metrics:
                self.log_scalars(f"Performance/{optimization_technique}", 
                               performance_metrics, batch_size)
            
            # Memory metrics group
            memory_metrics = {}
            if 'ram_usage_mb' in results:
                memory_metrics['RAM_MB'] = results['ram_usage_mb']
            if 'vram_usage_mb' in results and results['vram_usage_mb'] > 0:
                memory_metrics['VRAM_MB'] = results['vram_usage_mb']
            
            if memory_metrics:
                self.log_scalars(f"Memory/{optimization_technique}", 
                               memory_metrics, batch_size)
            
        except Exception as e:
            self.logger.error(f"Failed to log benchmark results: {e}")
    
    def log_model_graph(
        self, 
        model: torch.nn.Module, 
        input_size: tuple = (1, 3, 224, 224),
        device: str = "cpu"
    ) -> None:
        """
        Log model computational graph.
        
        Args:
            model: PyTorch model
            input_size: Input tensor size
            device: Device for dummy input
        """
        if not self.is_active:
            return
        
        try:
            # Create dummy input
            dummy_input = torch.randn(input_size).to(device)
            
            # Log model graph
            self.writer.add_graph(model, dummy_input)
            
            self.logger.info(f"📊 Model graph logged to TensorBoard")
            
        except Exception as e:
            self.logger.error(f"Failed to log model graph: {e}")
    
    def log_latency_distribution(
        self, 
        latencies: List[float], 
        optimization_technique: str,
        batch_size: int
    ) -> None:
        """
        Log distribution of latency measurements.
        
        Args:
            latencies: List of latency measurements in milliseconds
            optimization_technique: Name of optimization technique
            batch_size: Batch size for this measurement
        """
        if not self.is_active or not latencies:
            return
        
        try:
            # Log histogram
            self.log_histogram(
                f"{optimization_technique}/latency_distribution_batch_{batch_size}",
                latencies,
                step=batch_size
            )
            
            # Log statistics as scalars
            import statistics
            stats = {
                'min': min(latencies),
                'max': max(latencies),
                'mean': statistics.mean(latencies),
                'median': statistics.median(latencies),
                'std': statistics.stdev(latencies) if len(latencies) > 1 else 0.0
            }
            
            for stat_name, value in stats.items():
                self.log_scalar(
                    f"{optimization_technique}/latency_stats/{stat_name}",
                    value,
                    batch_size
                )
            
        except Exception as e:
            self.logger.error(f"Failed to log latency distribution: {e}")
    
    def log_comparison_chart(
        self,
        optimization_results: Dict[str, Dict[str, Any]],
        metric_name: str = "mean_latency_ms",
        chart_title: str = "Performance Comparison"
    ) -> None:
        """
        Log comparison chart across optimization techniques.
        
        Args:
            optimization_results: Dict mapping technique names to results
            metric_name: Metric to compare
            chart_title: Title for the chart
        """
        if not self.is_active:
            return
        
        try:
            # Group results by batch size
            batch_sizes = set()
            for results in optimization_results.values():
                batch_sizes.update([r.get('batch_size', 0) for r in results if isinstance(results, list)])
            
            # Create comparison for each batch size
            for batch_size in sorted(batch_sizes):
                comparison_data = {}
                
                for technique, results_list in optimization_results.items():
                    if isinstance(results_list, list):
                        # Find result for this batch size
                        batch_result = next(
                            (r for r in results_list if r.get('batch_size') == batch_size),
                            None
                        )
                        if batch_result and metric_name in batch_result:
                            comparison_data[technique] = batch_result[metric_name]
                
                if comparison_data:
                    self.log_scalars(
                        f"{chart_title}/batch_{batch_size}",
                        comparison_data,
                        batch_size
                    )
            
        except Exception as e:
            self.logger.error(f"Failed to log comparison chart: {e}")
    
    def log_system_metrics(
        self, 
        metrics: Dict[str, Any],
        step: Optional[int] = None
    ) -> None:
        """
        Log system metrics from MetricsCollector.
        
        Args:
            metrics: Dictionary with system metrics
            step: Global step value
        """
        if not self.is_active:
            return
        
        if step is None:
            step = self.global_step
            self.global_step += 1
        
        try:
            # CPU metrics
            if 'cpu' in metrics:
                cpu_metrics = metrics['cpu']
                for key, value in cpu_metrics.items():
                    if isinstance(value, (int, float)):
                        self.log_scalar(f"System/CPU/{key}", value, step)
            
            # Memory metrics
            if 'memory' in metrics:
                memory_metrics = metrics['memory']
                for key, value in memory_metrics.items():
                    if isinstance(value, (int, float)):
                        self.log_scalar(f"System/Memory/{key}", value, step)
            
            # GPU metrics
            if 'gpu' in metrics:
                for gpu_id, gpu_metrics in metrics['gpu'].items():
                    for key, value in gpu_metrics.items():
                        if isinstance(value, (int, float)):
                            self.log_scalar(f"System/GPU_{gpu_id}/{key}", value, step)
            
            # Process metrics
            if 'process' in metrics:
                process_metrics = metrics['process']
                for key, value in process_metrics.items():
                    if isinstance(value, (int, float)):
                        self.log_scalar(f"System/Process/{key}", value, step)
            
        except Exception as e:
            self.logger.error(f"Failed to log system metrics: {e}")
    
    def log_profiler_summary(
        self, 
        profiler_table: str,
        optimization_technique: str
    ) -> None:
        """
        Log PyTorch profiler summary table.
        
        Args:
            profiler_table: Formatted profiler table string
            optimization_technique: Name of optimization technique
        """
        if not self.is_active:
            return
        
        try:
            # Log as text
            self.writer.add_text(
                f"Profiler/{optimization_technique}/Summary",
                f"```\n{profiler_table}\n```",
                self.global_step
            )
            
        except Exception as e:
            self.logger.error(f"Failed to log profiler summary: {e}")
    
    def log_configuration(self, config: Dict[str, Any]) -> None:
        """
        Log experiment configuration.
        
        Args:
            config: Configuration dictionary
        """
        if not self.is_active:
            return
        
        try:
            # Convert config to readable format
            config_text = "# Experiment Configuration\n\n"
            
            def format_dict(d, indent=0):
                text = ""
                for key, value in d.items():
                    if isinstance(value, dict):
                        text += "  " * indent + f"**{key}:**\n"
                        text += format_dict(value, indent + 1)
                    else:
                        text += "  " * indent + f"- **{key}**: {value}\n"
                return text
            
            config_text += format_dict(config)
            
            self.writer.add_text("Configuration", config_text, 0)
            
        except Exception as e:
            self.logger.error(f"Failed to log configuration: {e}")
    
    def create_custom_layout(self) -> None:
        """Create custom TensorBoard layout for better visualization."""
        if not self.is_active:
            return
        
        try:
            from tensorboard.plugins.custom_scalar import layout_pb2
            
            # Create custom layout
            layout_summary = layout_pb2.Layout(
                category=[
                    layout_pb2.Category(
                        title='Performance Metrics',
                        chart=[
                            layout_pb2.Chart(
                                title='Latency Comparison',
                                multiline=layout_pb2.MultilineChartContent(
                                    tag=[r'Performance/.*/latency_ms']
                                )
                            ),
                            layout_pb2.Chart(
                                title='Throughput Comparison',
                                multiline=layout_pb2.MultilineChartContent(
                                    tag=[r'Performance/.*/throughput']
                                )
                            )
                        ]
                    ),
                    layout_pb2.Category(
                        title='Memory Usage',
                        chart=[
                            layout_pb2.Chart(
                                title='RAM Usage',
                                multiline=layout_pb2.MultilineChartContent(
                                    tag=[r'Memory/.*/RAM_MB']
                                )
                            ),
                            layout_pb2.Chart(
                                title='VRAM Usage',
                                multiline=layout_pb2.MultilineChartContent(
                                    tag=[r'Memory/.*/VRAM_MB']
                                )
                            )
                        ]
                    ),
                    layout_pb2.Category(
                        title='System Resources',
                        chart=[
                            layout_pb2.Chart(
                                title='CPU Utilization',
                                multiline=layout_pb2.MultilineChartContent(
                                    tag=[r'System/CPU/.*']
                                )
                            ),
                            layout_pb2.Chart(
                                title='GPU Utilization',
                                multiline=layout_pb2.MultilineChartContent(
                                    tag=[r'System/GPU_.*/.*']
                                )
                            )
                        ]
                    )
                ]
            )
            
            self.writer.add_custom_scalar_layout(layout_summary)
            
        except ImportError:
            self.logger.info("Custom scalar layout not available")
        except Exception as e:
            self.logger.error(f"Failed to create custom layout: {e}")
    
    def flush(self) -> None:
        """Flush pending data to disk."""
        if self.is_active:
            try:
                self.writer.flush()
            except Exception as e:
                self.logger.error(f"Failed to flush TensorBoard data: {e}")
    
    def close(self) -> None:
        """Close the TensorBoard logger."""
        if self.is_active:
            try:
                self.writer.close()
                self.is_active = False
                self.logger.info("📊 TensorBoard logger closed")
            except Exception as e:
                self.logger.error(f"Failed to close TensorBoard logger: {e}")
    
    def get_log_dir(self) -> str:
        """Get the log directory path."""
        return str(self.log_dir)


# Utility functions
def create_tensorboard_logger(
    experiment_name: str,
    log_dir: str = "./logs/tensorboard",
    config: Optional[Dict[str, Any]] = None
) -> TensorBoardLogger:
    """
    Create and configure a TensorBoard logger.
    
    Args:
        experiment_name: Name of the experiment
        log_dir: Base log directory
        config: Configuration to log
        
    Returns:
        Configured TensorBoard logger
    """
    logger = TensorBoardLogger(
        log_dir=log_dir,
        experiment_name=experiment_name
    )
    
    # Log configuration if provided
    if config:
        logger.log_configuration(config)
    
    # Create custom layout
    logger.create_custom_layout()
    
    return logger


def start_tensorboard_server(
    logdir: str,
    port: int = 6006,
    host: str = "localhost"
) -> None:
    """
    Start TensorBoard server programmatically.
    
    Args:
        logdir: Directory containing TensorBoard logs
        port: Port to serve on
        host: Host to bind to
    """
    import subprocess
    import sys
    
    try:
        cmd = [
            sys.executable, "-m", "tensorboard.main",
            "--logdir", logdir,
            "--port", str(port),
            "--host", host,
            "--reload_interval", "1"
        ]
        
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        print(f"🚀 TensorBoard started at http://{host}:{port}")
        print(f"📊 Serving logs from: {logdir}")
        print("Press Ctrl+C to stop")
        
        # Wait for process to complete or be interrupted
        try:
            process.wait()
        except KeyboardInterrupt:
            process.terminate()
            print("\n⏹️  TensorBoard stopped")
        
    except Exception as e:
        print(f"❌ Failed to start TensorBoard: {e}")


if __name__ == "__main__":
    # Example usage
    logger = TensorBoardLogger(experiment_name="test_experiment")
    
    # Log some test data
    for i in range(10):
        logger.log_scalar("test/loss", 1.0 / (i + 1), i)
        logger.log_scalar("test/accuracy", i * 10, i)
    
    logger.close()
    print("Test logging completed!")