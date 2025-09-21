"""
Device utilities for GPU/CPU detection and management.
"""

import torch
import psutil
import os
import platform
from typing import Dict, Any, List
import logging


def detect_environment() -> Dict[str, Any]:
    """Detect environment and hardware information."""
    env_info = {}
    
    # Basic system info
    env_info.update({
        "platform": platform.system(),
        "python_version": platform.python_version(),
        "cpu_count": psutil.cpu_count(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None
    })
    
    # Memory info
    memory = psutil.virtual_memory()
    env_info.update({
        "total_ram_gb": round(memory.total / 1024**3, 1),
        "available_ram_gb": round(memory.available / 1024**3, 1)
    })
    
    # GPU info
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        env_info.update({
            "gpu_count": gpu_count,
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(gpu_count)],
            "current_gpu_name": torch.cuda.get_device_name(0) if gpu_count > 0 else None,
            "current_gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1) if gpu_count > 0 else 0
        })
        env_info["device"] = "cuda"
    else:
        env_info.update({"gpu_count": 0, "gpu_names": []})
        env_info["device"] = "cpu"
    
    # Environment detection
    env_info.update({
        "is_colab": 'COLAB_GPU' in os.environ,
        "is_docker": os.path.exists('/.dockerenv'),
        "is_jupyter": 'JUPYTER_SERVER_ROOT' in os.environ
    })
    
    return env_info


def get_memory_info() -> Dict[str, float]:
    """Get current memory usage information."""
    memory_info = {}
    
    # System memory
    system_memory = psutil.virtual_memory()
    process = psutil.Process()
    
    memory_info.update({
        "system_total_gb": round(system_memory.total / 1024**3, 2),
        "system_available_gb": round(system_memory.available / 1024**3, 2),
        "process_ram_mb": round(process.memory_info().rss / 1024**2, 2),
        "cpu_percent": psutil.cpu_percent()
    })
    
    # GPU memory if available
    if torch.cuda.is_available():
        try:
            memory_info.update({
                "gpu_allocated_mb": round(torch.cuda.memory_allocated() / 1024**2, 2),
                "gpu_cached_mb": round(torch.cuda.memory_reserved() / 1024**2, 2)
            })
        except:
            pass
    
    return memory_info