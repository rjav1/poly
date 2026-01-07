"""
Logging configuration for the data collection pipeline.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_logger(name: str, log_dir: str = "logs", level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger with file and console handlers.
    
    Args:
        name: Logger name
        log_dir: Directory to store log files
        level: Logging level
        
    Returns:
        Configured logger
    """
    # Create logs directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # File handler - daily rotating log file
    date_str = datetime.now().strftime('%Y-%m-%d')
    file_handler = logging.FileHandler(
        log_path / f"{name}_{date_str}.log",
        encoding='utf-8'
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get an existing logger or create a new one."""
    return logging.getLogger(name)

