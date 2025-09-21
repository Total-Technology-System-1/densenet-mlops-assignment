"""
PyTorch Profiler Wrapper - Day 2
Comprehensive profiling integration for DenseNet benchmarking.
"""

import torch
from torch.profiler import profile, record_function, ProfilerActivity, schedule
from typing import List, Optional, Dict, Any, Union, Callable
import logging
from pathlib import Path
import json
import os


class PyTorchProfilerWrapper:
    """
    Wrapper for PyTorch Profiler with enhanced functionality for MLOps benchmarking.
    
    This class provides:
    - Simplified profiler configuration
    - Automatic trace export
    - Memory profiling integration
    - TensorBoard-compatible outputs
    """
    
    def __init__(
        self,
        activities: List[ProfilerActivity] = None,
        schedule_config: Optional[Dict[str, int]] = None,
        record_shapes: bool = True,
        profile_memory: bool = True,
        with_stack: bool = False,
        with_flops: bool = True,
        with_modules: bool = True,
        output_dir: Optional[str] = None
    ):
        """
        Initialize the PyTorch Profiler wrapper.
        
        Args:
            activities: List of profiling activities (CPU, CUDA)
            schedule_config: Profiler schedule configuration
            record_shapes: Whether to record tensor shapes
            profile_memory: Whether to profile memory usage
            with_stack: Whether to record call stack
            with_flops: Whether to record FLOPs
            with_modules: Whether to record module hierarchy
            output_dir: Directory to save profiler outputs
        """
        self.logger = logging.getLogger(__name__)
        
        # Set default activities
        if activities is None:
            activities = [ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(ProfilerActivity.CUDA)
        
        self.activities = activities
        
        # Set default schedule (for iterative profiling)
        if schedule_config is None:
            schedule_config = {
                'wait': 2,
                'warmup': 2, 
                'active': 6,
                'repeat': 1
            }
        
        # Create profiler schedule
        self.schedule_fn = schedule(
            wait=schedule_config['wait'],
            warmup=schedule_config['warmup'],
            active=schedule_config['active'],
            repeat=schedule_config['repeat']
        )
        
        # Profiler configuration
        self.profiler_config = {
            'activities': self.activities,
            'schedule': self.schedule_fn,
            'record_shapes': record_shapes,
            'profile_memory': profile_memory,
            'with_stack': with_stack,
            'with_flops': with_flops,
            'with_modules': with_modules
        }
        
        # Output configuration
        self.output_dir = Path(output_dir) if output_dir else Path('./results/profiles')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Profiler instance
        self.profiler: Optional[profile] = None
        self.is_profiling = False
        
        self.logger.info(f"🔍 PyTorch Profiler initialized")
        self.logger.info(f"   Activities: {[a.name for a in self.activities]}")
        self.logger.info(f"   Output dir: {self.output_dir}")
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
    
    def start(self) -> None:
        """Start profiling."""
        if self.is_profiling:
            self.logger.warning("Profiler already running")
            return
        
        self.logger.info("▶️  Starting PyTorch profiler...")
        
        # Create profiler instance
        self.profiler = profile(**self.profiler_config)
        
        # Start profiling
        self.profiler.start()
        self.is_profiling = True
        
        self.logger.info("✅ Profiler started")
    
    def stop(self) -> 'torch.profiler.profile':
        """
        Stop profiling and return the profiler object.
        
        Returns:
            PyTorch profiler object with collected data
        """
        if not self.is_profiling or self.profiler is None:
            self.logger.warning("Profiler not running")
            return None
        
        self.logger.info("⏹️  Stopping PyTorch profiler...")
        
        # Stop profiling
        self.profiler.stop()
        self.is_profiling = False
        
        self.logger.info("✅ Profiler stopped")
        return self.profiler
    
    def step(self) -> None:
        """Step the profiler (for iterative profiling)."""
        if self.profiler and self.is_profiling:
            self.profiler.step()
    
    def export_traces(
        self, 
        filename_prefix: str = "profile",
        export_chrome_trace: bool = True,
        export_tensorboard: bool = True
    ) -> Dict[str, str]:
        """
        Export profiler traces to various formats.
        
        Args:
            filename_prefix: Prefix for output files
            export_chrome_trace: Whether to export Chrome trace format
            export_tensorboard: Whether to export TensorBoard format
            
        Returns:
            Dictionary mapping format names to file paths
        """
        if self.profiler is None:
            self.logger.error("No profiler data to export")
            return {}
        
        exported_files = {}
        
        try:
            # Export Chrome trace (JSON format for Chrome://tracing)
            if export_chrome_trace:
                chrome_path = self.output_dir / f"{filename_prefix}_chrome_trace.json"
                self.profiler.export_chrome_trace(str(chrome_path))
                exported_files['chrome_trace'] = str(chrome_path)
                self.logger.info(f"📁 Chrome trace exported: {chrome_path}")
            
            # Export for TensorBoard
            if export_tensorboard:
                tb_dir = self.output_dir / f"{filename_prefix}_tensorboard"
                tb_dir.mkdir(exist_ok=True)
                self.profiler.export_stacks(str(tb_dir / "profiler_stacks.txt"), "self_cuda_time_total")
                exported_files['tensorboard'] = str(tb_dir)
                self.logger.info(f"📁 TensorBoard data exported: {tb_dir}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to export traces: {e}")
        
        return exported_files
    
    def get_summary_table(
        self, 
        sort_by: str = "self_cuda_time_total",
        row_limit: int = 20,
        group_by_input_shape: bool = False,
        group_by_stack: Optional[int] = None
    ) -> str:
        """
        Generate a summary table of profiling results.
        
        Args:
            sort_by: Column to sort by
            row_limit: Maximum number of rows
            group_by_input_shape: Whether to group by input shapes
            group_by_stack: Stack depth for grouping (None to disable)
            
        Returns:
            Formatted table string
        """
        if self.profiler is None:
            return "No profiler data available"
        
        try:
            # Get key averages
            if group_by_stack:
                key_averages = self.profiler.key_averages(group_by_stack_n=group_by_stack)
            elif group_by_input_shape:
                key_averages = self.profiler.key_averages(group_by_input_shape=True)
            else:
                key_averages = self.profiler.key_averages()
            
            # Generate table
            table = key_averages.table(
                sort_by=sort_by,
                row_limit=row_limit,
                max_src_column_width=50,
                max_name_column_width=30,
                max_shapes_column_width=30,
                header=f"Top {row_limit} operations (sorted by {sort_by})",
                top_level_events_only=False
            )
            
            return table
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate summary table: {e}")
            return f"Error generating summary: {e}"
    
    def get_memory_profile(self) -> Dict[str, Any]:
        """
        Extract memory profiling information.
        
        Returns:
            Dictionary with memory profiling data
        """
        if self.profiler is None:
            return {}
        
        try:
            # Get memory events
            memory_profile = {}
            
            # CUDA memory profiling
            if torch.cuda.is_available() and ProfilerActivity.CUDA in self.activities:
                memory_profile['cuda'] = {
                    'peak_allocated': torch.cuda.max_memory_allocated() / 1024**2,  # MB
                    'peak_reserved': torch.cuda.max_memory_reserved() / 1024**2,   # MB
                    'current_allocated': torch.cuda.memory_allocated() / 1024**2, # MB
                    'current_reserved': torch.cuda.memory_reserved() / 1024**2    # MB
                }
            
            # CPU memory profiling (basic)
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            memory_profile['cpu'] = {
                'rss_mb': memory_info.rss / 1024**2,  # Resident Set Size
                'vms_mb': memory_info.vms / 1024**2,  # Virtual Memory Size
                'percent': process.memory_percent()
            }
            
            return memory_profile
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get memory profile: {e}")
            return {}
    
    def save_detailed_analysis(
        self, 
        filename: str = "detailed_analysis.json"
    ) -> str:
        """
        Save detailed profiling analysis to JSON file.
        
        Args:
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        if self.profiler is None:
            self.logger.error("No profiler data to analyze")
            return ""
        
        analysis_path = self.output_dir / filename
        
        try:
            analysis = {
                'summary_table': self.get_summary_table(),
                'memory_profile': self.get_memory_profile(),
                'configuration': {
                    'activities': [a.name for a in self.activities],
                    'record_shapes': self.profiler_config['record_shapes'],
                    'profile_memory': self.profiler_config['profile_memory'],
                    'with_stack': self.profiler_config['with_stack'],
                    'with_flops': self.profiler_config['with_flops']
                },
                'system_info': {
                    'cuda_available': torch.cuda.is_available(),
                    'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
                    'pytorch_version': torch.__version__
                }
            }
            
            # Save to JSON
            with open(analysis_path, 'w') as f:
                json.dump(analysis, f, indent=2, default=str)
            
            self.logger.info(f"📁 Detailed analysis saved: {analysis_path}")
            return str(analysis_path)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save detailed analysis: {e}")
            return ""


# Context manager for quick profiling
class QuickProfiler:
    """Simple context manager for quick profiling tasks."""
    
    def __init__(self, name: str = "operation", output_dir: Optional[str] = None):
        self.name = name
        self.profiler_wrapper = PyTorchProfilerWrapper(output_dir=output_dir)
        
    def __enter__(self):
        self.profiler_wrapper.start()
        return self.profiler_wrapper
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        prof = self.profiler_wrapper.stop()
        if prof:
            # Quick export
            self.profiler_wrapper.export_traces(filename_prefix=self.name)
            
            # Print quick summary
            print(f"\n🔍 Profiling Summary for '{self.name}':")
            print(self.profiler_wrapper.get_summary_table(row_limit=10))


# Utility functions
def profile_model_inference(
    model: torch.nn.Module,
    input_data: torch.Tensor,
    num_iterations: int = 10,
    output_dir: str = "./profiles"
) -> Dict[str, Any]:
    """
    Quick utility to profile model inference.
    
    Args:
        model: PyTorch model to profile
        input_data: Input tensor for the model
        num_iterations: Number of inference iterations
        output_dir: Directory to save profiling results
        
    Returns:
        Dictionary with profiling summary
    """
    logger = logging.getLogger(__name__)
    
    # Setup profiler
    profiler_wrapper = PyTorchProfilerWrapper(output_dir=output_dir)
    
    # Profile inference
    with profiler_wrapper:
        model.eval()
        with torch.no_grad():
            for i in range(num_iterations):
                with record_function(f"inference_iteration_{i}"):
                    _ = model(input_data)
                profiler_wrapper.step()
    
    # Export results
    exported = profiler_wrapper.export_traces(filename_prefix="model_inference")
    summary_table = profiler_wrapper.get_summary_table()
    memory_profile = profiler_wrapper.get_memory_profile()
    
    logger.info("🔍 Model inference profiling completed")
    
    return {
        'summary_table': summary_table,
        'memory_profile': memory_profile,
        'exported_files': exported,
        'iterations': num_iterations
    }


def compare_model_profiles(
    models_dict: Dict[str, torch.nn.Module],
    input_data: torch.Tensor,
    output_dir: str = "./profiles/comparison"
) -> Dict[str, Any]:
    """
    Compare profiling results across multiple models.
    
    Args:
        models_dict: Dictionary mapping model names to model instances
        input_data: Input tensor for all models
        output_dir: Directory to save comparison results
        
    Returns:
        Dictionary with comparison results
    """
    comparison_results = {}
    
    for model_name, model in models_dict.items():
        print(f"🔍 Profiling {model_name}...")
        
        profile_dir = f"{output_dir}/{model_name}"
        result = profile_model_inference(model, input_data, output_dir=profile_dir)
        comparison_results[model_name] = result
    
    print("✅ Model comparison profiling completed")
    return comparison_results