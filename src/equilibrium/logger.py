"""
Equilibrium Logging Module

Provides automatic file logging with rotating handlers, configurable via settings.

Usage:
    1. Enable logging via environment variables:
       EQUILIBRIUM_LOGGING__ENABLED=true
       EQUILIBRIUM_LOGGING__LEVEL=DEBUG

    2. Or via config.toml:
       [logging]
       enabled = true
       level = "DEBUG"
       console = true
       file = true
       max_bytes = 5242880  # 5 MB
       backup_count = 3

    3. Programmatic configuration:
       from equilibrium.logger import configure_logging
       logger = configure_logging()
       logger.info("Hello from Equilibrium!")

Log files are stored in the log_dir path (default: ~/.local/share/EQUILIBRIUM/logs/).
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .settings import LoggingConfig, Settings

LOGGER_NAME = "equilibrium"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _get_log_filename(pattern: str, log_dir: Path) -> Path:
    """
    Generate the log filename from the pattern.

    Args:
        pattern: The filename pattern with optional {date} placeholder.
        log_dir: The directory where log files should be stored.

    Returns:
        The full path to the log file.
    """
    date_str = datetime.now().strftime("%Y%m%d")
    filename = pattern.replace("{date}", date_str)
    return log_dir / filename


def _has_handler_type(logger: logging.Logger, handler_type: type) -> bool:
    """Check if the logger already has a handler of the specified type."""
    return any(isinstance(h, handler_type) for h in logger.handlers)


def configure_logging(settings: Settings | None = None) -> logging.Logger:
    """
    Configure the Equilibrium logging system based on settings.

    This function sets up the "equilibrium" logger with console and/or file handlers
    according to the logging configuration in settings. It supports both
    size-based and time-based log rotation.

    Args:
        settings: Optional Settings object. If not provided, settings will be
                  obtained via get_settings() (without triggering recursive
                  logging configuration).

    Returns:
        The configured "equilibrium" logger instance.

    Notes:
        - This function is idempotent: calling it multiple times will not add
          duplicate handlers.
        - File handlers use RotatingFileHandler (size-based) or
          TimedRotatingFileHandler (time-based) depending on settings.
        - The {date} placeholder in filename_pattern is replaced with YYYYMMDD.
    """
    # Get settings if not provided, avoiding circular import
    if settings is None:
        from .settings import Settings

        # Create a fresh settings instance to avoid recursion via get_settings()
        settings = Settings()
        settings = settings.ensure_dirs()

    log_config: LoggingConfig = settings.logging
    log_dir: Path = settings.paths.log_dir  # type: ignore

    # Get or create the equilibrium logger
    logger = logging.getLogger(LOGGER_NAME)

    # Set the log level
    level = getattr(logging, log_config.level.upper(), logging.INFO)
    logger.setLevel(level)

    # Create formatter
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # Add console handler if enabled and not already present
    if log_config.console and not _has_handler_type(logger, logging.StreamHandler):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Add file handler if enabled and not already present
    if log_config.file:
        # Check for either type of rotating handler
        has_file_handler = _has_handler_type(
            logger, RotatingFileHandler
        ) or _has_handler_type(logger, TimedRotatingFileHandler)
        if not has_file_handler:
            log_file = _get_log_filename(log_config.filename_pattern, log_dir)

            if log_config.rotation_type == "time":
                file_handler = TimedRotatingFileHandler(
                    log_file,
                    when=log_config.time_when,
                    interval=log_config.time_interval,
                    backupCount=log_config.backup_count,
                )
            else:  # default to size-based rotation
                file_handler = RotatingFileHandler(
                    log_file,
                    maxBytes=log_config.max_bytes,
                    backupCount=log_config.backup_count,
                )

            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False

    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """
    Get a logger instance for the given name.

    Args:
        name: The logger name. If None, returns the root "equilibrium" logger.
              If provided, returns a child logger (e.g., "equilibrium.module").

    Returns:
        A logger instance.
    """
    if name is None:
        return logging.getLogger(LOGGER_NAME)
    return logging.getLogger(f"{LOGGER_NAME}.{name}")
