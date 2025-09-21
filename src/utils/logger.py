"""
Logging utilities for the MLOps assignment - Day 2
Provides consistent logging across all modules.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import json


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    console: bool = True,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with consistent formatting.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional file to log to
        console: Whether to log to console
        format_string: Custom format string
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Default format
    if format_string is None:
        format_string = '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
    
    formatter = logging.Formatter(format_string, datefmt='%Y-%m-%d %H:%M:%S')
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def setup_application_logging(
    log_dir: str = "./logs/application",
    level: int = logging.INFO
) -> Dict[str, logging.Logger]:
    """
    Set up logging for all application components.
    
    Args:
        log_dir: Directory for log files
        level: Logging level
        
    Returns:
        Dictionary of configured loggers
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for log files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    loggers = {}
    
    # Main application logger
    loggers['main'] = setup_logger(
        "DenseNetMLOps",
        level=level,
        log_file=str(log_path / f"main_{timestamp}.log"),
        console=True
    )
    
    # Benchmarking logger
    loggers['benchmark'] = setup_logger(
        "BenchmarkRunner",
        level=level,
        log_file=str(log_path / f"benchmark_{timestamp}.log"),
        console=False
    )
    
    # Model logger
    loggers['model'] = setup_logger(
        "DenseNetModel",
        level=level,
        log_file=str(log_path / f"model_{timestamp}.log"),
        console=False
    )
    
    # Profiler logger
    loggers['profiler'] = setup_logger(
        "Profiler",
        level=level,
        log_file=str(log_path / f"profiler_{timestamp}.log"),
        console=False
    )
    
    return loggers


class JsonLogger:
    """JSON structured logger for machine-readable logs."""
    
    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
    
    def log(self, data: Dict[str, Any]) -> None:
        """Log structured data as JSON."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            **data
        }
        
        with open(self.filepath, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')


def log_function_call(logger: logging.Logger):
    """Decorator to log function calls."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
            try:
                result = func(*args, **kwargs)
                logger.debug(f"{func.__name__} completed successfully")
                return result
            except Exception as e:
                logger.error(f"{func.__name__} failed: {e}")
                raise
        return wrapper
    return decorator