"""
Model Pruning Implementation - Day 3
Structured and unstructured pruning for DenseNet optimization.
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from typing import Dict, Any, List, Tuple, Optional
import logging
import copy
import time


class PruningOptimizer:
    """
    DenseNet pruning optimizer supporting structured and unstructured pruning.
    """
    
    def __init__(self, device: str = "cpu"):
        self.logger = logging.getLogger(__name__)
        self.device = torch.device(device)
        
        self.logger.info(f"Pruning optimizer initialized for {self.device}")
    
    def apply_unstructured_pruning(
        self,
        model: nn.Module,
        pruning_amount: float = 0.3,
        pruning_type: str = "magnitude"
    ) -> nn.Module:
        """
        Apply unstructured pruning to the model.
        Removes individual weights based on magnitude or other criteria.
        """
        self.logger.info(f"Applying unstructured {pruning_type} pruning ({pruning_amount*100}% sparsity)...")
        
        model_copy = copy.deepcopy(model)
        
        # Collect layers to prune (Conv2d and Linear layers)
        layers_to_prune = []
        for name, module in model_copy.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                layers_to_prune.append((module, 'weight'))
        
        # Apply pruning based on type
        if pruning_type == "magnitude":
            # L1 magnitude-based pruning
            prune.global_unstructured(
                layers_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=pruning_amount,
            )
        elif pruning_type == "random":
            # Random pruning
            prune.global_unstructured(
                layers_to_prune,
                pruning_method=prune.RandomUnstructured,
                amount=pruning_amount,
            )
        else:
            raise ValueError(f"Unknown pruning type: {pruning_type}")
        
        # Make pruning permanent (remove masks and actually zero out weights)
        for module, param in layers_to_prune:
            prune.remove(module, param)
        
        self.logger.info("Unstructured pruning applied successfully")
        return model_copy
    
    def apply_structured_pruning(
        self,
        model: nn.Module,
        pruning_amount: float = 0.3,
        dim: int = 0
    ) -> nn.Module:
        """
        Apply structured pruning to the model.
        Removes entire channels/filters based on importance.
        """
        self.logger.info(f"Applying structured pruning ({pruning_amount*100}% sparsity, dim={dim})...")
        
        model_copy = copy.deepcopy(model)
        
        # Apply structured pruning to Conv2d layers
        for name, module in model_copy.named_modules():
            if isinstance(module, nn.Conv2d):
                # Prune channels/filters
                prune.ln_structured(
                    module,
                    name='weight',
                    amount=pruning_amount,
                    n=2,  # L2 norm
                    dim=dim  # 0: output channels, 1: input channels
                )
        
        # Make pruning permanent
        for name, module in model_copy.named_modules():
            if isinstance(module, nn.Conv2d) and hasattr(module, 'weight_mask'):
                prune.remove(module, 'weight')
        
        self.logger.info("Structured pruning applied successfully")
        return model_copy
    
    def calculate_sparsity(self, model: nn.Module) -> Dict[str, float]:
        """
        Calculate sparsity statistics for the model.
        """
        total_params = 0
        zero_params = 0
        layer_sparsity = {}
        
        for name, param in model.named_parameters():
            if 'weight' in name:
                layer_total = param.numel()
                layer_zeros = torch.sum(param == 0).item()
                
                total_params += layer_total
                zero_params += layer_zeros
                
                layer_sparsity[name] = layer_zeros / layer_total if layer_total > 0 else 0.0
        
        overall_sparsity = zero_params / total_params if total_params > 0 else 0.0
        
        return {
            'overall_sparsity': overall_sparsity,
            'total_parameters': total_params,
            'zero_parameters': zero_params,
            'layer_sparsity': layer_sparsity
        }
    
    def benchmark_pruned_model(
        self,
        pruned_model: nn.Module,
        original_model: nn.Module,
        test_input: torch.Tensor,
        num_iterations: int = 100,
        technique: str = "pruned"
    ) -> Dict[str, Any]:
        """
        Benchmark pruned model performance against original.
        """
        self.logger.info(f"Benchmarking {technique} model...")
        
        # Ensure models are in eval mode
        pruned_model.eval()
        original_model.eval()
        
        # Move to device
        pruned_model = pruned_model.to(self.device)
        original_model = original_model.to(self.device)
        test_input = test_input.to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = pruned_model(test_input)
        
        # Benchmark pruned model
        pruned_times = []
        with torch.no_grad():
            for _ in range(num_iterations):
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                start_time = time.perf_counter()
                _ = pruned_model(test_input)
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                pruned_times.append((end_time - start_time) * 1000)  # Convert to ms
        
        # Benchmark original model
        original_times = []
        with torch.no_grad():
            for _ in range(num_iterations):
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                start_time = time.perf_counter()
                _ = original_model(test_input)
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                original_times.append((end_time - start_time) * 1000)
        
        # Calculate statistics
        import statistics
        
        avg_pruned_time = statistics.mean(pruned_times)
        avg_original_time = statistics.mean(original_times)
        speedup = avg_original_time / avg_pruned_time
        
        # Calculate sparsity
        sparsity_info = self.calculate_sparsity(pruned_model)
        
        # Model size comparison (effective size considering sparsity)
        def get_effective_model_size(model):
            param_size = 0
            for param in model.parameters():
                non_zero_params = torch.sum(param != 0).item()
                param_size += non_zero_params * param.element_size()
            return param_size / 1024**2  # Convert to MB
        
        original_size = sum(p.numel() * p.element_size() for p in original_model.parameters()) / 1024**2
        effective_pruned_size = get_effective_model_size(pruned_model)
        size_reduction = (1 - effective_pruned_size / original_size) * 100
        
        results = {
            'technique': technique,
            'pruned_latency_ms': avg_pruned_time,
            'original_latency_ms': avg_original_time,
            'speedup_ratio': speedup,
            'original_model_size_mb': original_size,
            'effective_pruned_size_mb': effective_pruned_size,
            'size_reduction_percent': size_reduction,
            'sparsity_percent': sparsity_info['overall_sparsity'] * 100,
            'zero_parameters': sparsity_info['zero_parameters'],
            'total_parameters': sparsity_info['total_parameters'],
            'iterations': num_iterations
        }
        
        self.logger.info(f"Pruning results:")
        self.logger.info(f"  Speedup: {speedup:.2f}x")
        self.logger.info(f"  Sparsity: {sparsity_info['overall_sparsity']*100:.1f}%")
        self.logger.info(f"  Effective size reduction: {size_reduction:.1f}%")
        
        return results
    
    def progressive_pruning(
        self,
        model: nn.Module,
        target_sparsity: float = 0.5,
        steps: int = 5,
        test_input: Optional[torch.Tensor] = None
    ) -> List[Tuple[nn.Module, Dict[str, Any]]]:
        """
        Apply progressive pruning in steps to reach target sparsity.
        """
        self.logger.info(f"Applying progressive pruning to {target_sparsity*100}% sparsity in {steps} steps")
        
        results = []
        current_model = copy.deepcopy(model)
        step_sparsity = target_sparsity / steps
        
        for step in range(steps):
            current_sparsity = (step + 1) * step_sparsity
            
            # Apply pruning step
            pruned_model = self.apply_unstructured_pruning(
                current_model,
                pruning_amount=step_sparsity,
                pruning_type="magnitude"
            )
            
            # Calculate actual sparsity
            sparsity_info = self.calculate_sparsity(pruned_model)
            
            step_info = {
                'step': step + 1,
                'target_sparsity': current_sparsity,
                'actual_sparsity': sparsity_info['overall_sparsity'],
                'technique': f'progressive_pruning_step_{step+1}'
            }
            
            # Benchmark if test input provided
            if test_input is not None:
                benchmark_results = self.benchmark_pruned_model(
                    pruned_model, model, test_input, 
                    num_iterations=50, technique=step_info['technique']
                )
                step_info.update(benchmark_results)
            
            results.append((pruned_model, step_info))
            current_model = pruned_model
            
            self.logger.info(f"Step {step+1}: {sparsity_info['overall_sparsity']*100:.1f}% sparsity achieved")
        
        return results
    
    def optimize_densenet(
        self,
        model: nn.Module,
        pruning_type: str = "unstructured",
        pruning_amount: float = 0.3,
        magnitude_type: str = "magnitude"
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Apply pruning optimization to DenseNet model.
        
        Args:
            model: Original DenseNet model
            pruning_type: 'unstructured' or 'structured'
            pruning_amount: Fraction of weights to prune (0.0 to 1.0)
            magnitude_type: 'magnitude' or 'random' for unstructured pruning
            
        Returns:
            Tuple of (optimized_model, optimization_info)
        """
        optimization_info = {
            'technique': f'pruning_{pruning_type}',
            'pruning_amount': pruning_amount,
            'magnitude_type': magnitude_type
        }
        
        try:
            if pruning_type == "unstructured":
                optimized_model = self.apply_unstructured_pruning(
                    model, pruning_amount, magnitude_type
                )
                
            elif pruning_type == "structured":
                optimized_model = self.apply_structured_pruning(
                    model, pruning_amount
                )
                
            else:
                raise ValueError(f"Unknown pruning type: {pruning_type}")
            
            # Calculate sparsity info
            sparsity_info = self.calculate_sparsity(optimized_model)
            optimization_info.update(sparsity_info)
            
            optimization_info['success'] = True
            optimization_info['model_size_mb'] = sum(
                p.numel() * p.element_size() for p in optimized_model.parameters()
            ) / 1024**2
            
            return optimized_model, optimization_info
            
        except Exception as e:
            self.logger.error(f"Pruning failed: {e}")
            optimization_info['success'] = False
            optimization_info['error'] = str(e)
            return model, optimization_info  # Return original model on failure


def prune_densenet_example():
    """Example usage of pruning optimizer."""
    from torchvision import models
    
    # Load DenseNet
    model = models.densenet121(pretrained=True)
    model.eval()
    
    # Create pruning optimizer
    optimizer = PruningOptimizer()
    
    # Apply unstructured pruning
    pruned_model, info = optimizer.optimize_densenet(
        model, 
        pruning_type="unstructured",
        pruning_amount=0.3
    )
    
    print(f"Pruning applied: {info}")
    
    # Test inference
    test_input = torch.randn(1, 3, 224, 224)
    
    with torch.no_grad():
        original_output = model(test_input)
        pruned_output = pruned_model(test_input)
        
        print(f"Output shapes - Original: {original_output.shape}, Pruned: {pruned_output.shape}")
        
        # Check output similarity
        mse = torch.mean((original_output - pruned_output) ** 2)
        print(f"MSE between outputs: {mse.item():.6f}")


if __name__ == "__main__":
    prune_densenet_example()