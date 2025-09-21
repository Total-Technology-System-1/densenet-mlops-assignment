"""
Configuration parser and management - Day 2
Handles all configuration loading and validation.
"""

import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking parameters."""
    batch_sizes: list = None
    num_iterations: int = 50
    warmup_iterations: int = 10
    device: str = "auto"
    
    def __post_init__(self):
        if self.batch_sizes is None:
            self.batch_sizes = [1, 4, 8, 16, 32]


@dataclass  
class ModelConfig:
    """Configuration for model parameters."""
    architecture: str = "densenet121"
    pretrained: bool = True
    num_classes: int = 1000
    input_size: list = None
    
    def __post_init__(self):
        if self.input_size is None:
            self.input_size = [3, 224, 224]


@dataclass
class OptimizationConfig:
    """Configuration for optimization techniques."""
    techniques: list = None
    quantization: dict = None
    pruning: dict = None
    distillation: dict = None
    onnx: dict = None
    
    def __post_init__(self):
        if self.techniques is None:
            self.techniques = ["quantization", "pruning", "onnx"]
        
        if self.quantization is None:
            self.quantization = {
                "backend": "fbgemm",
                "reduce_range": False
            }
        
        if self.pruning is None:
            self.pruning = {
                "amount": 0.3,
                "structured": False
            }
        
        if self.distillation is None:
            self.distillation = {
                "temperature": 4.0,
                "alpha": 0.7
            }
        
        if self.onnx is None:
            self.onnx = {
                "opset_version": 11,
                "optimize": True
            }


@dataclass
class OutputConfig:
    """Configuration for output and logging."""
    results_dir: str = "./results"
    logs_dir: str = "./logs"
    tensorboard_dir: str = "./logs/tensorboard"
    export_formats: list = None
    
    def __post_init__(self):
        if self.export_formats is None:
            self.export_formats = ["csv", "json"]


@dataclass
class MLOpsConfig:
    """Main configuration class combining all settings."""
    benchmark: BenchmarkConfig = None
    model: ModelConfig = None
    optimization: OptimizationConfig = None
    output: OutputConfig = None
    
    def __post_init__(self):
        if self.benchmark is None:
            self.benchmark = BenchmarkConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.optimization is None:
            self.optimization = OptimizationConfig()
        if self.output is None:
            self.output = OutputConfig()


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Load based on file extension
    with open(config_path, 'r') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            config = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")
    
    return config or {}


def save_config(config: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """
    Save configuration to file.
    
    Args:
        config: Configuration dictionary
        output_path: Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        if output_path.suffix.lower() in ['.yaml', '.yml']:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        elif output_path.suffix.lower() == '.json':
            json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unsupported output format: {output_path.suffix}")


def create_default_config() -> MLOpsConfig:
    """Create default configuration."""
    return MLOpsConfig()


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and normalize configuration.
    
    Args:
        config: Raw configuration dictionary
        
    Returns:
        Validated configuration dictionary
    """
    logger = logging.getLogger(__name__)
    
    # Create config object for validation
    try:
        # Extract sections
        benchmark_config = BenchmarkConfig(**config.get('benchmark', {}))
        model_config = ModelConfig(**config.get('model', {}))
        optimization_config = OptimizationConfig(**config.get('optimization', {}))
        output_config = OutputConfig(**config.get('output', {}))
        
        # Create main config
        mlops_config = MLOpsConfig(
            benchmark=benchmark_config,
            model=model_config,
            optimization=optimization_config,
            output=output_config
        )
        
        # Convert back to dictionary
        validated_config = asdict(mlops_config)
        
        logger.info("✅ Configuration validated successfully")
        return validated_config
        
    except Exception as e:
        logger.error(f"❌ Configuration validation failed: {e}")
        logger.info("🔧 Using default configuration")
        return asdict(create_default_config())


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries.
    
    Args:
        base_config: Base configuration
        override_config: Configuration to merge in
        
    Returns:
        Merged configuration
    """
    def merge_dict(base: Dict, override: Dict) -> Dict:
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge_dict(result[key], value)
            else:
                result[key] = value
        return result
    
    return merge_dict(base_config, override_config)


def get_config_from_args(args) -> Dict[str, Any]:
    """
    Extract configuration from command line arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Configuration dictionary
    """
    config = {}
    
    # Benchmark configuration
    if hasattr(args, 'batch_sizes') and args.batch_sizes:
        config['benchmark'] = {'batch_sizes': args.batch_sizes}
    
    if hasattr(args, 'gpu_enabled') and args.gpu_enabled is not None:
        if 'benchmark' not in config:
            config['benchmark'] = {}
        config['benchmark']['device'] = 'cuda' if args.gpu_enabled else 'cpu'
    
    # Optimization configuration
    if hasattr(args, 'optimizations') and args.optimizations:
        config['optimization'] = {'techniques': args.optimizations}
    
    # Output configuration
    if hasattr(args, 'output_dir') and args.output_dir:
        config['output'] = {'results_dir': args.output_dir}
    
    return config


# Default configuration template
DEFAULT_CONFIG_TEMPLATE = """
# DenseNet MLOps Assignment Configuration

benchmark:
  batch_sizes: [1, 4, 8, 16, 32]
  num_iterations: 50
  warmup_iterations: 10
  device: auto  # auto, cpu, cuda

model:
  architecture: densenet121
  pretrained: true
  num_classes: 1000
  input_size: [3, 224, 224]

optimization:
  techniques: [quantization, pruning, onnx]
  
  quantization:
    backend: fbgemm  # fbgemm for CPU, qnnpack for mobile
    reduce_range: false
    
  pruning:
    amount: 0.3  # 30% sparsity
    structured: false
    
  distillation:
    temperature: 4.0
    alpha: 0.7
    
  onnx:
    opset_version: 11
    optimize: true

output:
  results_dir: ./results
  logs_dir: ./logs
  tensorboard_dir: ./logs/tensorboard
  export_formats: [csv, json]
"""


def create_default_config_file(output_path: Union[str, Path] = "configs/benchmark_config.yaml") -> None:
    """
    Create default configuration file.
    
    Args:
        output_path: Path where to save the config file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(DEFAULT_CONFIG_TEMPLATE.strip())
    
    print(f"✅ Default configuration created: {output_path}")


if __name__ == "__main__":
    # Create default config file for testing
    create_default_config_file()
    
    # Test loading
    config = load_config("configs/benchmark_config.yaml")
    validated = validate_config(config)
    
    print("Configuration loaded and validated successfully!")