"""
Logging configuration for the application.
Replace print statements gradually with logger calls.
"""
import logging
import logging.handlers
from pathlib import Path

def setup_logging(log_level=logging.INFO):
    """
    Configure application-wide logging.
    
    Parameters
    ----------
    log_level : int
        Logging level (logging.DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Ensure logs directory exists
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Console handler (for terminal output)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    console_handler.setFormatter(console_format)
    
    # File handler (rotating, max 10MB per file, keep 5 files)
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "canteen_system.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(file_format)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# Convenience loggers for each module
def get_logger(name: str):
    """Get a logger for a specific module"""
    return logging.getLogger(name)
