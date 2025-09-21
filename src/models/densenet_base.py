"""
Base DenseNet-121 model implementation for benchmarking.
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from typing import Tuple, Optional, Dict, Any
import time
import logging


class DenseNetBase:
    """Base DenseNet-121 model wrapper for benchmarking."""
    
    def __init__(
        self, 
        pretrained: bool = True,
        num_classes: int = 1000,
        device: str = "auto"
    ):
        self.logger = logging.getLogger(__name__)
        
        # Auto-detect device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        
        self.logger.info(f"Initializing DenseNet-121 on device: {self.device}")
        
        # Load model
        self.model = models.densenet121(pretrained=pretrained)
        
        # Modify classifier if needed
        if num_classes != 1000:
            num_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(num_features, num_classes)
        
        # Move to device
        self.model = self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        self.logger.info("DenseNet-121 model initialized successfully")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        # Calculate model size
        param_count = sum(p.numel() for p in self.model.parameters())
        model_size_mb = param_count * 4 / (1024 ** 2)  # Assuming float32
        
        return {
            "architecture": "densenet121",
            "total_parameters": param_count,
            "model_size_mb": model_size_mb,
            "device": str(self.device),
            "pretrained": True
        }
    
    def warm_up(self, batch_size: int = 1, num_warmup: int = 5) -> None:
        """Warm up the model with dummy inputs."""
        self.logger.info(f"Warming up model with batch_size={batch_size}")
        
        dummy_input = torch.randn(
            batch_size, 3, 224, 224, 
            device=self.device, 
            dtype=torch.float32
        )
        
        with torch.no_grad():
            for i in range(num_warmup):
                _ = self.model(dummy_input)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
        
        self.logger.info("Model warmup completed")
    
    def single_inference(self, images: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Run inference and measure latency.
        
        Args:
            images: Input tensor [batch_size, 3, 224, 224]
            
        Returns:
            Tuple of (predictions, latency_ms)
        """
        self.model.eval()
        
        # Ensure input is on correct device
        if images.device != self.device:
            images = images.to(self.device)
        
        # Measure inference time
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        
        with torch.no_grad():
            predictions = self.model(images)
        
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000
        
        return predictions, latency_ms