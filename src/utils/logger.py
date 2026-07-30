"""Structured logging utility for the Predictive Maintenance platform."""

import logging
import sys
from pathlib import Path


def get_logger(name: str = "predictive_maintenance", log_file: str | None = None, level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a structured logger instance.

    Args:
        name (str): Logger name identifier.
        log_file (Optional[str]): Path to output log file.
        level (int): Logging level (default: logging.INFO).

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s:%(funcName)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
