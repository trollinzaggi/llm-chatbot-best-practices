"""
Logging utility module for standardized logging across all libraries
"""
import logging
import os
from typing import Optional
from datetime import datetime


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: str = "INFO"
) -> logging.Logger:
    """
    Set up a logger with both file and console handlers
    
    Args:
        name: Logger name (usually __name__ of the module)
        log_file: Optional log file path
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def log_api_call(logger: logging.Logger, service: str, operation: str, **kwargs):
    """
    Log API call details
    
    Args:
        logger: Logger instance
        service: Service name (e.g., "Azure OpenAI", "LangChain")
        operation: Operation being performed
        **kwargs: Additional details to log
    """
    log_message = f"API Call - Service: {service}, Operation: {operation}"
    if kwargs:
        details = ", ".join([f"{k}: {v}" for k, v in kwargs.items()])
        log_message += f", Details: {details}"
    logger.info(log_message)


def log_error(logger: logging.Logger, error: Exception, context: str = ""):
    """
    Log error with context
    
    Args:
        logger: Logger instance
        error: Exception that occurred
        context: Additional context about where the error occurred
    """
    error_message = f"Error occurred: {type(error).__name__}: {str(error)}"
    if context:
        error_message = f"{context} - {error_message}"
    logger.error(error_message, exc_info=True)


# Create default logger
default_logger = setup_logger(
    "azure-llm-poc",
    log_file=os.getenv("LOG_FILE", "logs/app.log"),
    level=os.getenv("LOG_LEVEL", "INFO")
)
