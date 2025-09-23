"""
Model Quantization Implementation - Day 3
INT8 and FP16 quantization for DenseNet optimization.
"""

import torch
import torch.nn as nn
from torch.quantization import quantize_dynamic, QConfig, default_qconfig
import torch.quantization.quantize_fx as quantize_fx
from typing import Dict, Any, Optional, Tuple
import logging
import copy
import time


class QuantizationOptimizer:
    """
    DenseNet quantization optimizer supporting multiple quantization strategies.
    """
    
    def __init__(self, device: str = "cpu"):
        self.logger = logging.getLogger(__name__)
        self.device = torch.device(device)
        
        # Quantization backends
        self.backends = {
            'x86': 'fbgemm',  # Intel x86 CPUs
            'arm': 'qnnpack'  # ARM CPUs, mobile
        }
        
        # Set appropriate backend
        if self.device.type == 'cpu':
            torch.backends.quantized.engine = self.backends['x86']
        
        self.logger.info(f"Quantization optimizer initialized for {self.device}")
    
    def apply_dynamic_quantization(
        self,
        model: nn.Module,
        qconfig_spec: Optional[Dict] = None
    ) -> nn.Module:
        """
        Apply dynamic quantization to the model.
        Weights are quantized, activations are quantized dynamically at runtime.
        """
        self.logger.info("Applying dynamic INT8 quantization...")
        
        # Default layers to quantize
        if qconfig_spec is None:
            qconfig_spec = {
                nn.Linear,
                nn.Conv2d
            }
        
        # Create quantized model
        model_copy = copy.deepcopy(model)
        model_copy.eval()
        
        quantized_model = quantize_dynamic(
            model_copy,
            qconfig_spec,
            dtype=torch.qint8
        )
        
        self.logger.info("Dynamic quantization applied successfully")
        return quantized_model
    
    def apply_static_quantization(
        self,
        model: nn.Module,
        representative_data: torch.Tensor,
        num_calibration_batches: int = 100
    ) -> nn.Module:
        """
        Apply static quantization with calibration data.
        Both weights and activations are quantized.
        """
        self.logger.info("Applying static INT8 quantization...")
        
        model_copy = copy.deepcopy(model)
        model_copy.eval()
        
        # Prepare model for quantization
        model_copy.qconfig = default_qconfig
        model_prepared = torch.quantization.prepare(model_copy)
        
        # Calibration phase
        self.logger.info(f"Running calibration with {num_calibration_batches} batches...")
        with torch.no_grad():
            for i in range(min(num_calibration_batches, representative_data.size(0))):
                batch = representative_data[i:i+1]
                model_prepared(batch)
        
        # Convert to quantized model
        quantized_model = torch.quantization.convert(model_prepared)
        
        self.logger.info("Static quantization applied successfully")
        return quantized_model
    
    def apply_fp16_quantization(self, model: nn.Module) -> nn.Module:
        """
        Apply half precision (FP16) quantization.
        """
        self.logger.info("Applying FP16 quantization...")
        
        model_copy = copy.deepcopy(model)
        model_copy.half()  # Convert to FP16
        
        self.logger.info("FP16 quantization applied successfully")
        return model_copy
    
    def benchmark_quantized_model(
        self,
        quantized_model: nn.Module,
        original_model: nn.Module,
        test_input: torch.Tensor,
        num_iterations: int = 100,
        technique: str = "quantized"
    ) -> Dict[str, Any]:
        """
        Benchmark quantized model performance.
        """
        self.logger.info(f"Benchmarking {technique} model...")
        
        # Ensure models are in eval mode
        quantized_model.eval()
        original_model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = quantized_model(test_input)
        
        # Benchmark quantized model
        quantized_times = []
        with torch.no_grad():
            for _ in range(num_iterations):
                start_time = time.perf_counter()
                _ = quantized_model(test_input)
                end_time = time.perf_counter()
                quantized_times.append((end_time - start_time) * 1000)  # Convert to ms
        
        # Benchmark original model
        original_times = []
        with torch.no_grad():
            for _ in range(num_iterations):
                start_time = time.perf_counter()
                _ = original_model(test_input)
                end_time = time.perf_counter()
                original_times.append((end_time - start_time) * 1000)
        
        # Calculate statistics
        import statistics
        
        avg_quantized_time = statistics.mean(quantized_times)
        avg_original_time = statistics.mean(original_times)
        speedup = avg_original_time / avg_quantized_time
        
        # Model size comparison
        def get_model_size(model):
            param_size = 0
            for param in model.parameters():
                param_size += param.nelement() * param.element_size()
            return param_size / 1024**2  # Convert to MB
        
        original_size = get_model_size(original_model)
        quantized_size = get_model_size(quantized_model)
        size_reduction = (1 - quantized_size / original_size) * 100
        
        results = {
            'technique': technique,
            'quantized_latency_ms': avg_quantized_time,
            'original_latency_ms': avg_original_time,
            'speedup_ratio': speedup,
            'quantized_model_size_mb': quantized_size,
            'original_model_size_mb': original_size,
            'size_reduction_percent': size_reduction,
            'iterations': num_iterations
        }
        
        self.logger.info(f"Quantization results:")
        self.logger.info(f"  Speedup: {speedup:.2f}x")
        self.logger.info(f"  Size reduction: {size_reduction:.1f}%")
        
        return results
    
    def optimize_densenet(
        self,
        model: nn.Module,
        optimization_type: str = "dynamic",
        calibration_data: Optional[torch.Tensor] = None
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Apply quantization optimization to DenseNet model.
        
        Args:
            model: Original DenseNet model
            optimization_type: 'dynamic', 'static', or 'fp16'
            calibration_data: Data for static quantization calibration
            
        Returns:
            Tuple of (optimized_model, optimization_info)
        """
        optimization_info = {
            'technique': f'quantization_{optimization_type}',
            'backend': torch.backends.quantized.engine if optimization_type != 'fp16' else 'native'
        }
        
        try:
            if optimization_type == "dynamic":
                optimized_model = self.apply_dynamic_quantization(model)
                
            elif optimization_type == "static":
                if calibration_data is None:
                    # Generate dummy calibration data if none provided
                    calibration_data = torch.randn(100, 3, 224, 224)
                    self.logger.warning("Using dummy calibration data for static quantization")
                
                optimized_model = self.apply_static_quantization(model, calibration_data)
                
            elif optimization_type == "fp16":
                optimized_model = self.apply_fp16_quantization(model)
                
            else:
                raise ValueError(f"Unknown optimization type: {optimization_type}")
            
            optimization_info['success'] = True
            optimization_info['model_size_mb'] = sum(
                p.numel() * p.element_size() for p in optimized_model.parameters()
            ) / 1024**2
            
            return optimized_model, optimization_info
            
        except Exception as e:
            self.logger.error(f"Quantization failed: {e}")
            optimization_info['success'] = False
            optimization_info['error'] = str(e)
            return model, optimization_info  # Return original model on failure