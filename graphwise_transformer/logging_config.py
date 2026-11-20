"""Logging setup: a built-in default that a deployment can replace with a file.

* Default: one stderr handler, ``LOG_FORMAT``, level from ``log_level``.
* Override: ``logging_config`` (or ``GRAPHWISE_LOGGING_CONFIG``) names a file in
  a standard :mod:`logging.config` format, which then controls everything:
  ``.yaml``/``.yml``/``.json`` hold a :func:`logging.config.dictConfig`
  schema, ``.ini``/``.conf``/``.cfg`` a :func:`logging.config.fileConfig` file.

Every record carries ``request_id`` either way, so any format may use
``%(request_id)s``. Model workers receive the same settings, so their output
matches the server's.
"""

from __future__ import annotations

import json
import logging
import logging.config
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .request_context import install_record_factory

LOG_FORMAT = "%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s"
VALID_LOG_LEVELS = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}
_YAML_SUFFIXES = {".yaml", ".yml"}
_DICT_CONFIG_SUFFIXES = _YAML_SUFFIXES | {".json"}
_FILE_CONFIG_SUFFIXES = {".ini", ".conf", ".cfg"}


@dataclass(frozen=True)
class LoggingSettings:
    level: str = "INFO"
    config_file: str | None = None


_active: LoggingSettings | None = None


def default_logging_config(level: str = "INFO") -> dict[str, Any]:
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"default": {"format": LOG_FORMAT}},
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "stream": "ext://sys.stderr",
            }
        },
        # Hugging Face Hub logs every HTTP request at INFO while workers load.
        "loggers": {"httpx": {"level": "WARNING"}},
        "root": {"level": level, "handlers": ["console"]},
    }


def validate_logging_config_file(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix not in _DICT_CONFIG_SUFFIXES | _FILE_CONFIG_SUFFIXES:
        raise ValueError(
            f"Unsupported logging config '{path}'; expected a .yaml/.yml/.json "
            "dictConfig or an .ini/.conf/.cfg fileConfig file"
        )
    if not Path(path).is_file():
        raise ValueError(f"Logging config file not found: {path}")
    return path


def configure_logging(settings: LoggingSettings | None = None) -> None:
    """Apply ``settings`` (the built-in default when omitted) to the process."""
    global _active
    settings = settings or LoggingSettings()
    install_record_factory()

    path = settings.config_file
    if path is None:
        logging.config.dictConfig(default_logging_config(settings.level))
    elif Path(path).suffix.lower() in _DICT_CONFIG_SUFFIXES:
        logging.config.dictConfig(_load_dict_config(path))
    else:
        logging.config.fileConfig(path, disable_existing_loggers=False)
    _active = settings


def _load_dict_config(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        if Path(path).suffix.lower() in _YAML_SUFFIXES:
            config = yaml.safe_load(handle)
        else:
            config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Logging config {path} must contain a mapping (dictConfig schema)")
    return config


def active_logging_settings() -> LoggingSettings:
    """Settings to hand to a child process so it logs like this one."""
    if _active is not None:
        return _active
    # Logging was set up by an embedding application, not configure_logging;
    # give workers the default format at the current root level.
    return LoggingSettings(level=logging.getLevelName(logging.getLogger().getEffectiveLevel()))
