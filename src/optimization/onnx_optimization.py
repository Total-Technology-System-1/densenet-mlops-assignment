"""
ONNX Model Optimization - Day 3
ONNX conversion and optimization for DenseNet.
"""

import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
from typing import Dict, Any, List, Tuple, Optional
import logging
import copy
import time
import tempfile
import os
from pathlib import Path


class ONNXOptimizer:
    """
    ONNX optimization for DenseNet models.
    Converts PyTorch models to optimized ONNX format for inference.
    """
    
    def __init__(self, device: str = "cpu"):
        self.logger = logging.getLogger(__name__)
        self.device = torch.device(device)
        
        # ONNX Runtime providers
        self.providers = ['CPUExecutionProvider']
        if self.device.type == 'cuda' and 'CUDAExecutionProvider' in ort.get_available_providers():
            self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        self.logger.info(f"ONNX optimizer initialized for {self.device}")
        self.logger.info(f"Available providers: {self.providers}")
    
    def convert_to_onnx(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
        output_path: Optional[str] = None,
        opset_version: int = 11,
        dynamic_axes: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Convert PyTorch model to ONNX format.
        
        Args:
            model: PyTorch model to convert
            input_shape: Input tensor shape
            output_path: Path to save ONNX model (temporary file if None)
            opset_version: ONNX opset version
            dynamic_axes: Dynamic axes specification
            
        Returns:
            Path to the saved ONNX model
        """
        self.logger.info(f"Converting model to ONNX (opset {opset_version})...")
        
        # Prepare model
        model_copy = copy.deepcopy(model)
        model_copy.eval()
        
        # Create dummy input
        dummy_input = torch.randn(*input_shape)
        
        # Setup output path
        if output_path is None:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.onnx')
            output_path = temp_file.name
            temp_file.close()
        
        # Default dynamic axes for batch dimension
        if dynamic_axes is None:
            dynamic_axes = {
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        
        try:
            # Convert to ONNX
            torch.onnx.export(
                model_copy,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes=dynamic_axes,
                verbose=False
            )
            
            # Verify ONNX model
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            
            self.logger.info(f"ONNX model saved: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"ONNX conversion failed: {e}")
            raise
    
    def optimize_onnx_model(
        self,
        onnx_path: str,
        optimization_level: str = "basic"
    ) -> str:
        """
        Apply ONNX Runtime optimizations to the model.
        
        Args:
            onnx_path: Path to ONNX model
            optimization_level: 'basic', 'extended', or 'all'
            
        Returns:
            Path to optimized ONNX model
        """
        self.logger.info(f"Applying ONNX optimizations (level: {optimization_level})...")
        
        # Map optimization levels
        opt_level_map = {
            'basic': ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
            'extended': ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
            'all': ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        }
        
        opt_level = opt_level_map.get(optimization_level, ort.GraphOptimizationLevel.ORT_ENABLE_BASIC)
        
        # Create optimized model path
        optimized_path = onnx_path.replace('.onnx', '_optimized.onnx')
        
        try:
            # Create session options
            sess_options = ort.SessionOptions()
            sess_options.optimized_model_filepath = optimized_path
            sess_options.graph_optimization_level = opt_level
            
            # Create session to trigger optimization
            session = ort.InferenceSession(onnx_path, sess_options, providers=self.providers)
            
            self.logger.info(f"Optimized ONNX model saved: {optimized_path}")
            return optimized_path
            
        except Exception as e:
            self.logger.error(f"ONNX optimization failed: {e}")
            return onnx_path  # Return original path on failure
    
    def create_onnx_session(self, onnx_path: str) -> ort.InferenceSession:
        """
        Create ONNX Runtime inference session.
        
        Args:
            onnx_path: Path to ONNX model
            
        Returns:
            ONNX Runtime session
        """
        try:
            # Session options for performance
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            
            # Enable memory pattern optimization
            sess_options.enable_mem_pattern = True
            sess_options.enable_cpu_mem_arena = True
            
            # Create session
            session = ort.InferenceSession(
                onnx_path, 
                sess_options, 
                providers=self.providers
            )
            
            self.logger.info(f"ONNX session created with providers: {session.get_providers()}")
            return session
            
        except Exception as e:
            self.logger.error(f"Failed to create ONNX session: {e}")
            raise
    
    def benchmark_onnx_model(
        self,
        onnx_session: ort.InferenceSession,
        pytorch_model: nn.Module,
        test_input: torch.Tensor,
        num_iterations: int = 100,
        technique: str = "onnx"
    ) -> Dict[str, Any]:
        """
        Benchmark ONNX model against PyTorch model.
        
        Args:
            onnx_session: ONNX Runtime session
            pytorch_model: Original PyTorch model
            test_input: Test input tensor
            num_iterations: Number of benchmark iterations
            technique: Optimization technique name
            
        Returns:
            Dictionary with benchmark results
        """
        self.logger.info(f"Benchmarking {technique} model...")
        
        # Prepare PyTorch model
        pytorch_model.eval()
        pytorch_model = pytorch_model.to(self.device)
        test_input_torch = test_input.to(self.device)
        
        # Prepare ONNX input
        input_name = onnx_session.get_inputs()[0].name
        test_input_numpy = test_input.cpu().numpy()
        
        # Warmup ONNX
        for _ in range(10):
            _ = onnx_session.run(None, {input_name: test_input_numpy})
        
        # Warmup PyTorch
        with torch.no_grad():
            for _ in range(10):
                _ = pytorch_model(test_input_torch)
        
        # Benchmark ONNX
        onnx_times = []
        for _ in range(num_iterations):
            start_time = time.perf_counter()
            onnx_outputs = onnx_session.run(None, {input_name: test_input_numpy})
            end_time = time.perf_counter()
            onnx_times.append((end_time - start_time) * 1000)  # Convert to ms
        
        # Benchmark PyTorch
        pytorch_times = []
        with torch.no_grad():
            for _ in range(num_iterations):
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                start_time = time.perf_counter()
                pytorch_outputs = pytorch_model(test_input_torch)
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                pytorch_times.append((end_time - start_time) * 1000)
        
        # Calculate statistics
        import statistics
        
        avg_onnx_time = statistics.mean(onnx_times)
        avg_pytorch_time = statistics.mean(pytorch_times)
        speedup = avg_pytorch_time / avg_onnx_time
        
        # Output similarity check
        with torch.no_grad():
            pytorch_output = pytorch_model(test_input_torch).cpu().numpy()
        onnx_output = onnx_session.run(None, {input_name: test_input_numpy})[0]
        
        mse = ((pytorch_output - onnx_output) ** 2).mean()
        max_diff = abs(pytorch_output - onnx_output).max()
        
        results = {
            'technique': technique,
            'onnx_latency_ms': avg_onnx_time,
            'pytorch_latency_ms': avg_pytorch_time,
            'speedup_ratio': speedup,
            'output_mse': float(mse),
            'output_max_diff': float(max_diff),
            'iterations': num_iterations,
            'providers': onnx_session.get_providers()
        }
        
        self.logger.info(f"ONNX results:")
        self.logger.info(f"  Speedup: {speedup:.2f}x")
        self.logger.info(f"  Output MSE: {mse:.6f}")
        self.logger.info(f"  Max difference: {max_diff:.6f}")
        
        return results
    
    def get_onnx_model_info(self, onnx_path: str) -> Dict[str, Any]:
        """
        Get information about ONNX model.
        
        Args:
            onnx_path: Path to ONNX model
            
        Returns:
            Dictionary with model information
        """
        try:
            onnx_model = onnx.load(onnx_path)
            
            # Model size
            model_size = os.path.getsize(onnx_path) / 1024**2  # MB
            
            # Input/output info
            inputs = [(inp.name, inp.type, [d.dim_value for d in inp.type.tensor_type.shape.dim]) 
                     for inp in onnx_model.graph.input]
            outputs = [(out.name, out.type, [d.dim_value for d in out.type.tensor_type.shape.dim]) 
                      for out in onnx_model.graph.output]
            
            # Count operations
            op_count = {}
            for node in onnx_model.graph.node:
                op_count[node.op_type] = op_count.get(node.op_type, 0) + 1
            
            return {
                'model_size_mb': model_size,
                'opset_version': onnx_model.opset_import[0].version,
                'inputs': inputs,
                'outputs': outputs,
                'node_count': len(onnx_model.graph.node),
                'operation_types': op_count,
                'path': onnx_path
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get ONNX model info: {e}")
            return {'error': str(e)}
    
    def optimize_densenet(
        self,
        model: nn.Module,
        optimization_level: str = "basic",
        opset_version: int = 11,
        output_dir: Optional[str] = None
    ) -> Tuple[ort.InferenceSession, Dict[str, Any]]:
        """
        Complete ONNX optimization pipeline for DenseNet.
        
        Args:
            model: Original PyTorch DenseNet model
            optimization_level: ONNX optimization level
            opset_version: ONNX opset version
            output_dir: Directory to save ONNX models
            
        Returns:
            Tuple of (onnx_session, optimization_info)
        """
        optimization_info = {
            'technique': f'onnx_{optimization_level}',
            'opset_version': opset_version,
            'providers': self.providers
        }
        
        try:
            # Setup output directory
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                base_path = os.path.join(output_dir, 'densenet121')
            else:
                temp_dir = tempfile.mkdtemp()
                base_path = os.path.join(temp_dir, 'densenet121')
            
            # Convert to ONNX
            onnx_path = self.convert_to_onnx(
                model,
                output_path=f"{base_path}.onnx",
                opset_version=opset_version
            )
            
            # Optimize ONNX model
            optimized_path = self.optimize_onnx_model(onnx_path, optimization_level)
            
            # Create inference session
            onnx_session = self.create_onnx_session(optimized_path)
            
            # Get model info
            model_info = self.get_onnx_model_info(optimized_path)
            optimization_info.update(model_info)
            
            optimization_info.update({
                'success': True,
                'onnx_path': onnx_path,
                'optimized_path': optimized_path,
                'model_size_mb': model_info.get('model_size_mb', 0)
            })
            
            return onnx_session, optimization_info
            
        except Exception as e:
            self.logger.error(f"ONNX optimization failed: {e}")
            optimization_info['success'] = False
            optimization_info['error'] = str(e)
            
            # Return dummy session on failure
            return None, optimization_info


def test_onnx_optimization():
    """Test ONNX optimization functionality."""
    from torchvision import models
    
    # Load DenseNet
    model = models.densenet121(pretrained=True)
    model.eval()
    
    # Create ONNX optimizer
    optimizer = ONNXOptimizer()
    
    # Apply ONNX optimization
    onnx_session, info = optimizer.optimize_densenet(
        model,
        optimization_level="basic",
        output_dir="./onnx_models"
    )
    
    print(f"ONNX optimization result: {info}")
    
    if onnx_session:
        # Test inference
        test_input = torch.randn(1, 3, 224, 224)
        
        # PyTorch inference
        with torch.no_grad():
            pytorch_output = model(test_input)
        
        # ONNX inference
        input_name = onnx_session.get_inputs()[0].name
        onnx_output = onnx_session.run(None, {input_name: test_input.numpy()})[0]
        
        print(f"Output shapes - PyTorch: {pytorch_output.shape}, ONNX: {onnx_output.shape}")
        
        # Check output similarity
        mse = ((pytorch_output.numpy() - onnx_output) ** 2).mean()
        print(f"MSE between outputs: {mse:.6f}")


if __name__ == "__main__":
    test_onnx_optimization()