# src/yourpkg/settings.py
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from platformdirs import user_config_dir, user_data_dir
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

APP_NAME = "EQUILIBRIUM"


class LoggingConfig(BaseModel):
    """
    Configuration for the Equilibrium logging system.

    Attributes:
        enabled: Whether logging is enabled. Opt-in by default (False).
        level: The logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        console: Whether to log to console (stdout).
        file: Whether to log to a file in log_dir.
        filename_pattern: Pattern for log filenames. {date} is replaced with YYYYMMDD.
        max_bytes: Maximum size of a log file before rotation (default 10 MB).
        backup_count: Number of rotated backup files to keep.
        rotation_type: Type of log rotation ('size' or 'time').
        time_when: When to rotate for time-based rotation (see TimedRotatingFileHandler).
        time_interval: Interval for time-based rotation.
    """

    enabled: bool = False
    level: str = "INFO"
    console: bool = True
    file: bool = True
    filename_pattern: str = "equilibrium_{date}.log"
    max_bytes: int = 10_485_760  # 10 MB
    backup_count: int = 5
    rotation_type: Literal["size", "time"] = "size"
    time_when: str = "midnight"
    time_interval: int = 1


class DebugConfig(BaseModel):
    """
    Configuration for debug outputs (e.g., iteration logs).

    Attributes:
        keep_iteration_logs: Max number of iteration log files to keep per model label.
            Use None to disable pruning.
    """

    keep_iteration_logs: Optional[int] = 5


class JaxConfig(BaseModel):
    """
    Configuration for JAX runtime behavior.

    Attributes:
        compilation_cache_enabled: Whether to enable persistent compilation caching.
            When enabled, JAX saves compiled functions to disk, dramatically speeding
            up subsequent runs ("warm runs") by avoiding recompilation.
        compilation_cache_dir: Directory for the compilation cache. If None, uses
            a subdirectory of data_dir.
        min_compile_time_secs: Minimum compilation time (in seconds) before a function
            is cached. Set to 0.0 to cache everything, or higher to only cache
            expensive compilations.
        enable_x64: Whether to enable 64-bit floating point precision (required for
            economic modeling). Default True.
    """

    compilation_cache_enabled: bool = True
    compilation_cache_dir: Optional[Path] = None
    min_compile_time_secs: float = 0.0
    enable_x64: bool = True

    @field_validator("compilation_cache_dir", mode="before")
    @classmethod
    def _expanduser(cls, v):
        if isinstance(v, str):
            return Path(v).expanduser()
        if isinstance(v, Path):
            return v.expanduser()
        return v


def _default_data_dir() -> Path:
    return Path(user_data_dir(APP_NAME)).expanduser()


def _default_config_file() -> Path:
    # user config file path; we read it if present
    return Path(user_config_dir(APP_NAME)) / "config.toml"


class Paths(BaseModel):
    data_dir: Path = Field(default_factory=_default_data_dir)
    save_dir: Optional[Path] = None
    log_dir: Optional[Path] = None
    debug_dir: Optional[Path] = None
    plot_dir: Optional[Path] = None
    jax_cache_dir: Optional[Path] = None

    @field_validator("*", mode="before")
    @classmethod
    def _expanduser(cls, v):
        if isinstance(v, str):
            return Path(v).expanduser()
        if isinstance(v, Path):
            return v.expanduser()
        return v

    @model_validator(mode="after")
    def fill_subdirs(self):
        """
        Ensure subdirectories default to subfolders of data_dir,
        but don't override explicit values.
        """
        base = self.data_dir.expanduser()
        if self.save_dir is None:
            self.save_dir = base / "cache"
        if self.log_dir is None:
            self.log_dir = base / "logs"
        if self.debug_dir is None:
            self.debug_dir = base / "debug"
        if self.plot_dir is None:
            self.plot_dir = base / "plots"
        if self.jax_cache_dir is None:
            self.jax_cache_dir = base / "jax_cache"
        return self

    def ensure_exists(self) -> "Paths":
        for p in [
            self.data_dir,
            self.save_dir,
            self.log_dir,
            self.debug_dir,
            self.plot_dir,
            self.jax_cache_dir,
        ]:
            if isinstance(p, Path):
                p.mkdir(parents=True, exist_ok=True)
        return self


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix=f"{APP_NAME}_",  # e.g., EQUILIBRIUM_PATHS__DATA_DIR overrides paths.data_dir
        env_file=".env",  # optional: project-level .env
        env_nested_delimiter="__",  # EQUILIBRIUM_PATHS__DATA_DIR=...
        extra="ignore",
    )

    # Group directories under one field to keep things tidy
    paths: Paths = Field(default_factory=Paths)

    # Logging configuration
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    # Debug configuration
    debug: DebugConfig = Field(default_factory=DebugConfig)

    # JAX configuration
    jax: JaxConfig = Field(default_factory=JaxConfig)

    # other toggles you might add later, kept minimal for now
    verbose: bool = False
    seed: Optional[int] = None

    # Optional: load a user file if it exists (TOML/JSON/env-like)
    config_file: Path = Field(default_factory=_default_config_file)

    @field_validator("config_file", mode="after")
    @classmethod
    def _normalize_config_file(cls, v: Path) -> Path:
        return v.expanduser()

    def ensure_dirs(self) -> "Settings":
        self.paths.ensure_exists()
        return self


def _configure_jax(settings: Settings) -> None:
    """
    Configure JAX runtime settings based on the Settings object.

    This function should be called once during initialization to set up:
    - Persistent compilation caching (dramatically speeds up subsequent runs)
    - 64-bit precision (required for economic modeling)
    """
    import jax

    # Enable 64-bit precision
    if settings.jax.enable_x64:
        jax.config.update("jax_enable_x64", True)

    # Configure persistent compilation cache
    if settings.jax.compilation_cache_enabled:
        # Use configured path or fall back to paths.jax_cache_dir
        cache_dir = settings.jax.compilation_cache_dir or settings.paths.jax_cache_dir
        if cache_dir is not None:
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            jax.config.update("jax_compilation_cache_dir", str(cache_dir))
            jax.config.update(
                "jax_persistent_cache_min_compile_time_secs",
                settings.jax.min_compile_time_secs,
            )


@lru_cache
def get_settings() -> Settings:
    """
    Lazily load once per process. Merge precedence:
    1) Built-ins (platformdirs defaults)
    2) User config file (if present): ~/.config/yourpkg/config.toml
    3) .env / environment variables (highest precedence)

    If logging is enabled in settings, the logging system is automatically
    configured after loading settings.
    """
    # Step 1/3: start with env/.env + defaults
    s = Settings()

    # Step 2/3: merge user file if present
    if s.config_file.is_file():
        # ultra-lightweight parse: Pydantic can read TOML via tomllib in 3.11+
        import tomllib

        data = tomllib.loads(s.config_file.read_text())
        # allow both flat and nested under [yourpkg]
        payload = data.get(APP_NAME, data)
        s = s.model_copy(update=Settings(**payload).model_dump(exclude_unset=True))

    # Step 3/3: re-apply env on top (env wins)
    s = Settings(**s.model_dump(exclude_unset=True))

    s = s.ensure_dirs()

    # Configure JAX (compilation cache, precision, etc.)
    _configure_jax(s)

    # Auto-configure logging if enabled
    if s.logging.enabled:
        from .logger import configure_logging

        configure_logging(s)

    return s
