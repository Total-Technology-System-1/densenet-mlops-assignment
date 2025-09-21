"""
System Metrics Collector - Day 2
Comprehensive system resource monitoring for benchmarking.
"""

import time
import threading
import psutil
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from collections import defaultdict, deque
import json

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    pynvml = None


class MetricsCollector:
    """
    System metrics collector for comprehensive performance monitoring.
    
    Collects:
    - CPU usage and temperature
    - Memory usage (RAM)
    - GPU usage and memory (if available)
    - Process-specific metrics
    - Network and disk I/O (optional)
    """
    
    def __init__(
        self,
        collection_interval: float = 0.1,  # 100ms
        buffer_size: int = 1000,
        collect_gpu: bool = True,
        collect_process: bool = True,
        collect_system: bool = True
    ):
        """
        Initialize metrics collector.
        
        Args:
            collection_interval: Time between metric collections (seconds)
            buffer_size: Maximum number of samples to keep in memory
            collect_gpu: Whether to collect GPU metrics
            collect_process: Whether to collect process-specific metrics
            collect_system: Whether to collect system-wide metrics
        """
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.collection_interval = collection_interval
        self.buffer_size = buffer_size
        self.collect_gpu = collect_gpu and NVML_AVAILABLE
        self.collect_process = collect_process
        self.collect_system = collect_system
        
        # Initialize NVIDIA ML if available
        if self.collect_gpu and NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.gpu_count = pynvml.nvmlDeviceGetCount()
                self.logger.info(f"📊 NVML initialized - {self.gpu_count} GPU(s) detected")
            except Exception as e:
                self.logger.warning(f"Failed to initialize NVML: {e}")
                self.collect_gpu = False
                self.gpu_count = 0
        else:
            self.gpu_count = 0
        
        # Metrics storage
        self.metrics_buffer = {
            'cpu': deque(maxlen=buffer_size),
            'memory': deque(maxlen=buffer_size),
            'gpu': defaultdict(lambda: deque(maxlen=buffer_size)),
            'process': deque(maxlen=buffer_size),
            'timestamps': deque(maxlen=buffer_size)
        }
        
        # Collection state
        self.is_collecting = False
        self.collection_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # Process handle
        if self.collect_process:
            self.process = psutil.Process()
        
        self.logger.info("📊 MetricsCollector initialized")
        self.logger.info(f"   Collection interval: {collection_interval}s")
        self.logger.info(f"   GPU monitoring: {self.collect_gpu}")
        self.logger.info(f"   Process monitoring: {self.collect_process}")
    
    def _collect_cpu_metrics(self) -> Dict[str, Any]:
        """Collect CPU metrics."""
        try:
            cpu_metrics = {
                'usage_percent': psutil.cpu_percent(interval=None, percpu=False),
                'usage_per_core': psutil.cpu_percent(interval=None, percpu=True),
                'load_avg': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None,
                'core_count': psutil.cpu_count(),
                'logical_count': psutil.cpu_count(logical=True)
            }
            
            # CPU frequency if available
            try:
                freq = psutil.cpu_freq()
                if freq:
                    cpu_metrics.update({
                        'frequency_mhz': freq.current,
                        'frequency_min_mhz': freq.min,
                        'frequency_max_mhz': freq.max
                    })
            except:
                pass
            
            # CPU temperature if available (Linux)
            try:
                temps = psutil.sensors_temperatures()
                if 'coretemp' in temps:
                    cpu_temps = [sensor.current for sensor in temps['coretemp']]
                    cpu_metrics['temperature_c'] = max(cpu_temps) if cpu_temps else None
            except:
                pass
            
            return cpu_metrics
            
        except Exception as e:
            self.logger.warning(f"Failed to collect CPU metrics: {e}")
            return {}
    
    def _collect_memory_metrics(self) -> Dict[str, Any]:
        """Collect memory metrics."""
        try:
            # System memory
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            return {
                # Virtual memory
                'total_gb': memory.total / 1024**3,
                'available_gb': memory.available / 1024**3,
                'used_gb': memory.used / 1024**3,
                'free_gb': memory.free / 1024**3,
                'usage_percent': memory.percent,
                
                # Swap memory
                'swap_total_gb': swap.total / 1024**3,
                'swap_used_gb': swap.used / 1024**3,
                'swap_free_gb': swap.free / 1024**3,
                'swap_usage_percent': swap.percent,
                
                # Memory details (if available)
                'buffers_gb': getattr(memory, 'buffers', 0) / 1024**3,
                'cached_gb': getattr(memory, 'cached', 0) / 1024**3,
                'shared_gb': getattr(memory, 'shared', 0) / 1024**3
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to collect memory metrics: {e}")
            return {}
    
    def _collect_gpu_metrics(self) -> Dict[int, Dict[str, Any]]:
        """Collect GPU metrics for all available GPUs."""
        if not self.collect_gpu or not NVML_AVAILABLE:
            return {}
        
        gpu_metrics = {}
        
        try:
            for gpu_id in range(self.gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                
                # Basic info
                name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                
                # Memory info
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                # Utilization info
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                
                # Temperature
                try:
                    temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                except:
                    temperature = None
                
                # Power usage
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                except:
                    power = None
                
                # Clock speeds
                try:
                    graphics_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
                    memory_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
                except:
                    graphics_clock = memory_clock = None
                
                gpu_metrics[gpu_id] = {
                    'name': name,
                    'memory_used_mb': memory_info.used / 1024**2,
                    'memory_total_mb': memory_info.total / 1024**2,
                    'memory_free_mb': memory_info.free / 1024**2,
                    'memory_usage_percent': (memory_info.used / memory_info.total) * 100,
                    'gpu_utilization_percent': utilization.gpu,
                    'memory_utilization_percent': utilization.memory,
                    'temperature_c': temperature,
                    'power_watts': power,
                    'graphics_clock_mhz': graphics_clock,
                    'memory_clock_mhz': memory_clock
                }
            
        except Exception as e:
            self.logger.warning(f"Failed to collect GPU metrics: {e}")
        
        return gpu_metrics
    
    def _collect_process_metrics(self) -> Dict[str, Any]:
        """Collect process-specific metrics."""
        if not self.collect_process:
            return {}
        
        try:
            # Memory info
            memory_info = self.process.memory_info()
            memory_percent = self.process.memory_percent()
            
            # CPU info
            cpu_percent = self.process.cpu_percent()
            cpu_times = self.process.cpu_times()
            
            # Process info
            num_threads = self.process.num_threads()
            
            # I/O info (if available)
            try:
                io_counters = self.process.io_counters()
                io_info = {
                    'read_bytes': io_counters.read_bytes,
                    'write_bytes': io_counters.write_bytes,
                    'read_count': io_counters.read_count,
                    'write_count': io_counters.write_count
                }
            except:
                io_info = {}
            
            return {
                'pid': self.process.pid,
                'memory_rss_mb': memory_info.rss / 1024**2,
                'memory_vms_mb': memory_info.vms / 1024**2,
                'memory_percent': memory_percent,
                'cpu_percent': cpu_percent,
                'cpu_user_time': cpu_times.user,
                'cpu_system_time': cpu_times.system,
                'num_threads': num_threads,
                **io_info
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to collect process metrics: {e}")
            return {}
    
    def _collect_single_sample(self) -> None:
        """Collect a single sample of all metrics."""
        timestamp = time.time()
        
        # Collect all metrics
        cpu_metrics = self._collect_cpu_metrics() if self.collect_system else {}
        memory_metrics = self._collect_memory_metrics() if self.collect_system else {}
        gpu_metrics = self._collect_gpu_metrics()
        process_metrics = self._collect_process_metrics()
        
        # Store in buffers
        self.metrics_buffer['cpu'].append(cpu_metrics)
        self.metrics_buffer['memory'].append(memory_metrics)
        self.metrics_buffer['process'].append(process_metrics)
        self.metrics_buffer['timestamps'].append(timestamp)
        
        # Store GPU metrics (per GPU)
        for gpu_id, metrics in gpu_metrics.items():
            self.metrics_buffer['gpu'][gpu_id].append(metrics)
    
    def _collection_loop(self) -> None:
        """Main collection loop running in separate thread."""
        self.logger.info("📊 Metrics collection started")
        
        while not self.stop_event.is_set():
            start_time = time.time()
            
            try:
                self._collect_single_sample()
            except Exception as e:
                self.logger.error(f"Error in metrics collection: {e}")
            
            # Sleep for the remaining time
            elapsed = time.time() - start_time
            sleep_time = max(0, self.collection_interval - elapsed)
            
            if self.stop_event.wait(sleep_time):
                break
        
        self.logger.info("📊 Metrics collection stopped")
    
    def start_monitoring(self) -> None:
        """Start metrics collection in background thread."""
        if self.is_collecting:
            self.logger.warning("Metrics collection already running")
            return
        
        self.stop_event.clear()
        self.collection_thread = threading.Thread(
            target=self._collection_loop,
            daemon=True,
            name="MetricsCollector"
        )
        
        self.collection_thread.start()
        self.is_collecting = True
        
        self.logger.info("▶️  Started metrics monitoring")
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """
        Stop metrics collection and return summary statistics.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.is_collecting:
            self.logger.warning("Metrics collection not running")
            return {}
        
        # Stop collection
        self.stop_event.set()
        if self.collection_thread:
            self.collection_thread.join(timeout=2.0)
        
        self.is_collecting = False
        
        # Calculate summary statistics
        summary = self._calculate_summary_stats()
        
        self.logger.info("⏹️  Stopped metrics monitoring")
        return summary
    
    def _calculate_summary_stats(self) -> Dict[str, Any]:
        """Calculate summary statistics from collected metrics."""
        if not self.metrics_buffer['timestamps']:
            return {}
        
        summary = {
            'collection_duration_seconds': len(self.metrics_buffer['timestamps']) * self.collection_interval,
            'samples_collected': len(self.metrics_buffer['timestamps'])
        }
        
        try:
            # CPU summary
            if self.metrics_buffer['cpu']:
                cpu_usage = [m.get('usage_percent', 0) for m in self.metrics_buffer['cpu'] if m]
                if cpu_usage:
                    summary['cpu'] = {
                        'avg_usage_percent': sum(cpu_usage) / len(cpu_usage),
                        'max_usage_percent': max(cpu_usage),
                        'min_usage_percent': min(cpu_usage)
                    }
            
            # Memory summary
            if self.metrics_buffer['memory']:
                memory_usage = [m.get('usage_percent', 0) for m in self.metrics_buffer['memory'] if m]
                if memory_usage:
                    summary['memory'] = {
                        'avg_usage_percent': sum(memory_usage) / len(memory_usage),
                        'max_usage_percent': max(memory_usage),
                        'min_usage_percent': min(memory_usage)
                    }
            
            # GPU summary
            summary['gpu'] = {}
            for gpu_id, gpu_buffer in self.metrics_buffer['gpu'].items():
                if gpu_buffer:
                    gpu_utilization = [m.get('gpu_utilization_percent', 0) for m in gpu_buffer if m]
                    memory_utilization = [m.get('memory_usage_percent', 0) for m in gpu_buffer if m]
                    
                    if gpu_utilization:
                        summary['gpu'][f'gpu_{gpu_id}'] = {
                            'avg_gpu_percent': sum(gpu_utilization) / len(gpu_utilization),
                            'max_gpu_percent': max(gpu_utilization),
                            'avg_memory_percent': sum(memory_utilization) / len(memory_utilization) if memory_utilization else 0,
                            'max_memory_percent': max(memory_utilization) if memory_utilization else 0
                        }
            
            # Process summary
            if self.metrics_buffer['process']:
                process_cpu = [m.get('cpu_percent', 0) for m in self.metrics_buffer['process'] if m]
                process_memory = [m.get('memory_percent', 0) for m in self.metrics_buffer['process'] if m]
                
                if process_cpu:
                    summary['process'] = {
                        'avg_cpu_percent': sum(process_cpu) / len(process_cpu),
                        'max_cpu_percent': max(process_cpu),
                        'avg_memory_percent': sum(process_memory) / len(process_memory) if process_memory else 0,
                        'max_memory_percent': max(process_memory) if process_memory else 0
                    }
        
        except Exception as e:
            self.logger.error(f"Error calculating summary stats: {e}")
        
        return summary
    
    def get_current_snapshot(self) -> Dict[str, Any]:
        """Get a single snapshot of current metrics."""
        self._collect_single_sample()
        
        if not self.metrics_buffer['timestamps']:
            return {}
        
        return {
            'timestamp': self.metrics_buffer['timestamps'][-1],
            'cpu': self.metrics_buffer['cpu'][-1] if self.metrics_buffer['cpu'] else {},
            'memory': self.metrics_buffer['memory'][-1] if self.metrics_buffer['memory'] else {},
            'process': self.metrics_buffer['process'][-1] if self.metrics_buffer['process'] else {},
            'gpu': {f'gpu_{gpu_id}': buffer[-1] for gpu_id, buffer in self.metrics_buffer['gpu'].items() if buffer}
        }
    
    def export_metrics(self, filepath: str) -> None:
        """Export collected metrics to JSON file."""
        try:
            # Convert deques to lists for JSON serialization
            export_data = {
                'timestamps': list(self.metrics_buffer['timestamps']),
                'cpu': list(self.metrics_buffer['cpu']),
                'memory': list(self.metrics_buffer['memory']),
                'process': list(self.metrics_buffer['process']),
                'gpu': {str(gpu_id): list(buffer) for gpu_id, buffer in self.metrics_buffer['gpu'].items()},
                'collection_config': {
                    'interval': self.collection_interval,
                    'buffer_size': self.buffer_size,
                    'gpu_enabled': self.collect_gpu,
                    'process_enabled': self.collect_process,
                    'system_enabled': self.collect_system
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            self.logger.info(f"📁 Metrics exported to: {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to export metrics: {e}")
    
    def clear_buffer(self) -> None:
        """Clear all collected metrics."""
        for buffer in self.metrics_buffer.values():
            if isinstance(buffer, dict):
                for sub_buffer in buffer.values():
                    sub_buffer.clear()
            else:
                buffer.clear()
        
        self.logger.info("🗑️  Metrics buffer cleared")