"""Unit tests for configuration loading and logging utilities."""

import logging

from src.utils.config import load_config
from src.utils.logger import get_logger


def test_load_config():
    config = load_config()
    assert isinstance(config, dict)
    assert "project" in config
    assert "data" in config
    assert "models" in config
    assert "business" in config


def test_get_logger():
    logger = get_logger("test_logger")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_logger"
