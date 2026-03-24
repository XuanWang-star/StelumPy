"""
StelumPy Logging Configuration
==============================
Centralized logging setup for the StelumPy package.

Usage
-----
In any StelumPy module::

    import logging
    logger = logging.getLogger(__name__)

    logger.info("Loading model...")
    logger.debug("Debug details")
    logger.warning("Warning message")
    logger.error("Error occurred")

To configure logging level globally::

    import logging
    logging.getLogger('StelumPy').setLevel(logging.DEBUG)
"""

import logging
from typing import Optional


# Default format for StelumPy log messages
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the given module name.

    Parameters
    ----------
    name : str
        Usually __name__ of the calling module.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Only configure handler if not already configured
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        # Create console handler
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)

        # Create formatter
        formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
        handler.setFormatter(formatter)

        logger.addHandler(handler)

    return logger


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> None:
    """
    Configure root StelumPy logger with custom settings.

    Parameters
    ----------
    level : int, optional
        Logging level (default: INFO).
    log_file : str, optional
        Path to log file. If None, only console output.
    """
    root_logger = logging.getLogger('StelumPy')
    root_logger.setLevel(level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
        root_logger.addHandler(file_handler)
