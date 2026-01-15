#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the Equilibrium logging module.
"""

import logging
import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from equilibrium.logger import (
    LOGGER_NAME,
    _get_log_filename,
    _has_handler_type,
    configure_logging,
    get_logger,
)
from equilibrium.settings import LoggingConfig, Paths, Settings


@pytest.fixture
def temp_log_dir():
    """Create a temporary directory for log files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def clean_logger():
    """Ensure the equilibrium logger is clean before each test."""
    logger = logging.getLogger(LOGGER_NAME)
    # Remove all handlers
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    logger.setLevel(logging.NOTSET)
    yield logger
    # Clean up after test
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)


class TestLoggingConfig:
    """Tests for LoggingConfig model."""

    def test_default_values(self):
        """Test that LoggingConfig has correct default values."""
        config = LoggingConfig()
        assert config.enabled is False
        assert config.level == "INFO"
        assert config.console is True
        assert config.file is True
        assert config.filename_pattern == "equilibrium_{date}.log"
        assert config.max_bytes == 10_485_760
        assert config.backup_count == 5
        assert config.rotation_type == "size"
        assert config.time_when == "midnight"
        assert config.time_interval == 1

    def test_custom_values(self):
        """Test that LoggingConfig accepts custom values."""
        config = LoggingConfig(
            enabled=True,
            level="DEBUG",
            console=False,
            file=True,
            filename_pattern="custom_{date}.log",
            max_bytes=5_242_880,
            backup_count=3,
            rotation_type="time",
            time_when="H",
            time_interval=6,
        )
        assert config.enabled is True
        assert config.level == "DEBUG"
        assert config.console is False
        assert config.max_bytes == 5_242_880
        assert config.rotation_type == "time"


class TestSettingsWithLogging:
    """Tests for Settings class with logging configuration."""

    def test_settings_has_logging_config(self):
        """Test that Settings includes logging configuration."""
        settings = Settings()
        assert hasattr(settings, "logging")
        assert isinstance(settings.logging, LoggingConfig)

    def test_settings_logging_from_env(self, temp_log_dir):
        """Test that logging settings can be set via environment variables."""
        with mock.patch.dict(
            os.environ,
            {
                "EQUILIBRIUM_LOGGING__ENABLED": "true",
                "EQUILIBRIUM_LOGGING__LEVEL": "DEBUG",
                "EQUILIBRIUM_LOGGING__CONSOLE": "false",
            },
        ):
            # Create fresh settings to pick up env vars
            settings = Settings()
            assert settings.logging.enabled is True
            assert settings.logging.level == "DEBUG"
            assert settings.logging.console is False


class TestConfigureLogging:
    """Tests for configure_logging function."""

    def test_configure_creates_logger(self, clean_logger, temp_log_dir):
        """Test that configure_logging creates a logger with handlers."""
        settings = Settings()
        # Override paths to use temp directory
        settings.paths = Paths(data_dir=temp_log_dir)
        settings.paths.ensure_exists()

        logger = configure_logging(settings)

        assert logger.name == LOGGER_NAME
        assert len(logger.handlers) >= 1

    def test_configure_with_console_only(self, clean_logger, temp_log_dir):
        """Test configuration with only console handler."""
        import sys

        settings = Settings()
        settings.paths = Paths(data_dir=temp_log_dir)
        settings.paths.ensure_exists()
        settings.logging = LoggingConfig(console=True, file=False)

        logger = configure_logging(settings)

        console_handlers = [
            h
            for h in logger.handlers
            if isinstance(h, logging.StreamHandler)
            and h.stream in (sys.stdout, sys.stderr)
        ]
        assert len(console_handlers) == 1

    def test_configure_with_file_only(self, clean_logger, temp_log_dir):
        """Test configuration with only file handler."""
        settings = Settings()
        settings.paths = Paths(data_dir=temp_log_dir)
        settings.paths.ensure_exists()
        settings.logging = LoggingConfig(console=False, file=True)

        logger = configure_logging(settings)

        # Should have a rotating file handler
        from logging.handlers import RotatingFileHandler

        file_handlers = [
            h for h in logger.handlers if isinstance(h, RotatingFileHandler)
        ]
        assert len(file_handlers) == 1

    def test_log_file_created(self, clean_logger, temp_log_dir):
        """Test that log file is created in log_dir."""
        settings = Settings()
        settings.paths = Paths(data_dir=temp_log_dir)
        settings.paths.ensure_exists()
        settings.logging = LoggingConfig(console=False, file=True)

        logger = configure_logging(settings)
        logger.info("Test message")

        # Force flush
        for handler in logger.handlers:
            handler.flush()

        log_files = list(settings.paths.log_dir.glob("equilibrium_*.log"))
        assert len(log_files) == 1

        # Verify log content
        content = log_files[0].read_text()
        assert "Test message" in content

    def test_log_level_setting(self, clean_logger, temp_log_dir):
        """Test that log level is correctly set."""
        settings = Settings()
        settings.paths = Paths(data_dir=temp_log_dir)
        settings.paths.ensure_exists()
        settings.logging = LoggingConfig(level="DEBUG", console=True, file=False)

        logger = configure_logging(settings)

        assert logger.level == logging.DEBUG

    def test_duplicate_handler_prevention(self, clean_logger, temp_log_dir):
        """Test that calling configure_logging multiple times doesn't add duplicate handlers."""
        settings = Settings()
        settings.paths = Paths(data_dir=temp_log_dir)
        settings.paths.ensure_exists()
        settings.logging = LoggingConfig(console=True, file=True)

        logger1 = configure_logging(settings)
        initial_handler_count = len(logger1.handlers)

        # Call configure_logging again
        logger2 = configure_logging(settings)

        assert logger1 is logger2  # Same logger instance
        assert len(logger2.handlers) == initial_handler_count

    def test_time_based_rotation(self, clean_logger, temp_log_dir):
        """Test time-based rotation configuration."""
        from logging.handlers import TimedRotatingFileHandler

        settings = Settings()
        settings.paths = Paths(data_dir=temp_log_dir)
        settings.paths.ensure_exists()
        settings.logging = LoggingConfig(
            console=False,
            file=True,
            rotation_type="time",
            time_when="H",
            time_interval=1,
        )

        logger = configure_logging(settings)

        timed_handlers = [
            h for h in logger.handlers if isinstance(h, TimedRotatingFileHandler)
        ]
        assert len(timed_handlers) == 1

    def test_size_based_rotation_settings(self, clean_logger, temp_log_dir):
        """Test that size-based rotation uses correct settings."""
        from logging.handlers import RotatingFileHandler

        settings = Settings()
        settings.paths = Paths(data_dir=temp_log_dir)
        settings.paths.ensure_exists()
        settings.logging = LoggingConfig(
            console=False,
            file=True,
            rotation_type="size",
            max_bytes=1_000_000,
            backup_count=3,
        )

        logger = configure_logging(settings)

        rotating_handlers = [
            h for h in logger.handlers if isinstance(h, RotatingFileHandler)
        ]
        assert len(rotating_handlers) == 1

        handler = rotating_handlers[0]
        assert handler.maxBytes == 1_000_000
        assert handler.backupCount == 3


class TestLogRotation:
    """Tests for log rotation functionality."""

    def test_rotation_on_size_exceeded(self, clean_logger, temp_log_dir):
        """Test that log files rotate when size is exceeded."""
        settings = Settings()
        settings.paths = Paths(data_dir=temp_log_dir)
        settings.paths.ensure_exists()
        # Set a very small max_bytes to trigger rotation
        settings.logging = LoggingConfig(
            console=False,
            file=True,
            max_bytes=100,  # 100 bytes
            backup_count=2,
        )

        logger = configure_logging(settings)

        # Write enough data to trigger rotation
        for i in range(50):
            logger.info(f"This is a longer test message number {i} to fill up the log")
            for handler in logger.handlers:
                handler.flush()

        # Check for rotated files
        log_files = list(settings.paths.log_dir.glob("equilibrium_*.log*"))
        # Should have main log file + backup files
        assert len(log_files) >= 2


class TestGetLogFilename:
    """Tests for _get_log_filename helper function."""

    def test_date_replacement(self):
        """Test that {date} is replaced with current date."""
        log_dir = Path("/tmp/logs")
        result = _get_log_filename("test_{date}.log", log_dir)

        # Should contain YYYYMMDD format
        from datetime import datetime

        expected_date = datetime.now().strftime("%Y%m%d")
        assert expected_date in str(result)
        assert result.parent == log_dir

    def test_no_date_placeholder(self):
        """Test pattern without {date} placeholder."""
        log_dir = Path("/tmp/logs")
        result = _get_log_filename("static.log", log_dir)

        assert result == log_dir / "static.log"


class TestHasHandlerType:
    """Tests for _has_handler_type helper function."""

    def test_detects_existing_handler(self):
        """Test detection of existing handler type."""
        logger = logging.getLogger("test_has_handler")
        handler = logging.StreamHandler()
        logger.addHandler(handler)

        assert _has_handler_type(logger, logging.StreamHandler) is True
        assert _has_handler_type(logger, logging.FileHandler) is False

        # Clean up
        logger.removeHandler(handler)

    def test_empty_handlers(self):
        """Test with no handlers."""
        logger = logging.getLogger("test_empty_handlers")
        # Ensure no handlers
        for h in logger.handlers[:]:
            logger.removeHandler(h)

        assert _has_handler_type(logger, logging.StreamHandler) is False


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_root_logger(self):
        """Test getting the root equilibrium logger."""
        logger = get_logger()
        assert logger.name == LOGGER_NAME

    def test_get_child_logger(self):
        """Test getting a child logger."""
        logger = get_logger("mymodule")
        assert logger.name == f"{LOGGER_NAME}.mymodule"

    def test_child_logger_inherits_settings(self, clean_logger, temp_log_dir):
        """Test that child loggers inherit settings from parent."""
        settings = Settings()
        settings.paths = Paths(data_dir=temp_log_dir)
        settings.paths.ensure_exists()
        settings.logging = LoggingConfig(level="DEBUG", console=True, file=False)

        configure_logging(settings)

        child_logger = get_logger("child")

        # Child should inherit effective level when parent is configured
        assert child_logger.getEffectiveLevel() == logging.DEBUG


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
