import json
import logging

import pytest

from graphwise_transformer import logging_config
from graphwise_transformer.config import load_config
from graphwise_transformer.logging_config import (
    LOG_FORMAT,
    LoggingSettings,
    active_logging_settings,
    configure_logging,
)
from graphwise_transformer.request_context import request_id_var


@pytest.fixture(autouse=True)
def _restore_logging(monkeypatch):
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    monkeypatch.setattr(logging_config, "_active", None)
    for name in ("GRAPHWISE_LOG_LEVEL", "GRAPHWISE_LOGGING_CONFIG"):
        monkeypatch.delenv(name, raising=False)
    yield
    for handler in root.handlers[:]:
        if handler not in handlers:
            root.removeHandler(handler)
            handler.close()
    for handler in handlers:
        if handler not in root.handlers:
            root.addHandler(handler)
    root.setLevel(level)


def _emit(message: str, request_id: str = "req-9") -> None:
    token = request_id_var.set(request_id)
    try:
        logging.getLogger("graphwise.test").warning(message)
    finally:
        request_id_var.reset(token)


# ----------------------------
# Applying settings
# ----------------------------


def test_default_config_uses_request_id_format(capsys):
    configure_logging(LoggingSettings(level="DEBUG"))

    assert logging.getLogger().level == logging.DEBUG
    _emit("hello")
    line = capsys.readouterr().err.strip()
    assert line.endswith("WARNING [req-9] graphwise.test: hello")
    assert active_logging_settings() == LoggingSettings(level="DEBUG")


def test_json_dict_config_replaces_default(tmp_path, capsys):
    path = tmp_path / "logging.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "disable_existing_loggers": False,
                "formatters": {"f": {"format": "custom|%(request_id)s|%(message)s"}},
                "handlers": {
                    "h": {
                        "class": "logging.StreamHandler",
                        "formatter": "f",
                        "stream": "ext://sys.stderr",
                    }
                },
                "root": {"level": "WARNING", "handlers": ["h"]},
            }
        ),
        encoding="utf-8",
    )

    configure_logging(LoggingSettings(level="DEBUG", config_file=str(path)))

    assert logging.getLogger().level == logging.WARNING  # the file wins
    _emit("from json")
    assert capsys.readouterr().err.strip() == "custom|req-9|from json"


@pytest.mark.parametrize("name", ["logging.yaml", "logging.yml"])
def test_yaml_dict_config_replaces_default(tmp_path, capsys, name):
    path = tmp_path / name
    path.write_text(
        "version: 1\n"
        "disable_existing_loggers: false\n"
        "formatters:\n"
        "  f:\n"
        '    format: "yaml|%(request_id)s|%(message)s"\n'
        "handlers:\n"
        "  h:\n"
        "    class: logging.StreamHandler\n"
        "    formatter: f\n"
        "    stream: ext://sys.stderr\n"
        "root:\n"
        "  level: INFO\n"
        "  handlers: [h]\n",
        encoding="utf-8",
    )

    configure_logging(LoggingSettings(level="DEBUG", config_file=str(path)))

    assert logging.getLogger().level == logging.INFO  # the file wins
    _emit("from yaml")
    assert capsys.readouterr().err.strip() == "yaml|req-9|from yaml"


def test_dict_config_file_must_hold_a_mapping(tmp_path):
    path = tmp_path / "logging.yaml"
    path.write_text("- not\n- a mapping\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must contain a mapping"):
        configure_logging(LoggingSettings(config_file=str(path)))


def test_ini_file_config_replaces_default(tmp_path, capsys):
    path = tmp_path / "logging.ini"
    path.write_text(
        "[loggers]\nkeys=root\n\n"
        "[handlers]\nkeys=h\n\n"
        "[formatters]\nkeys=f\n\n"
        "[logger_root]\nlevel=INFO\nhandlers=h\n\n"
        "[handler_h]\nclass=StreamHandler\nformatter=f\nargs=(sys.stderr,)\n\n"
        "[formatter_f]\nformat=ini|%(request_id)s|%(message)s\n",
        encoding="utf-8",
    )

    configure_logging(LoggingSettings(config_file=str(path)))

    _emit("from ini")
    assert capsys.readouterr().err.strip() == "ini|req-9|from ini"


def test_workers_inherit_root_level_when_logging_was_set_up_elsewhere():
    logging.getLogger().setLevel(logging.ERROR)
    assert active_logging_settings() == LoggingSettings(level="ERROR")


# ----------------------------
# config.properties / env vars
# ----------------------------


def _config(tmp_path, extra=""):
    path = tmp_path / "config.properties"
    path.write_text("port=5050\n" + extra, encoding="utf-8")
    return str(path)


def test_log_level_defaults_to_info_and_env_overrides(tmp_path, monkeypatch):
    assert load_config(_config(tmp_path)).logging == LoggingSettings("INFO", None)

    monkeypatch.setenv("GRAPHWISE_LOG_LEVEL", "debug")
    assert load_config(_config(tmp_path, "log_level=WARNING\n")).log_level == "DEBUG"


def test_invalid_log_level_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Invalid log_level"):
        load_config(_config(tmp_path, "log_level=LOUD\n"))


def test_logging_config_from_properties_and_env(tmp_path, monkeypatch):
    json_file = tmp_path / "a.json"
    json_file.write_text("{}", encoding="utf-8")
    ini_file = tmp_path / "b.ini"
    ini_file.write_text("", encoding="utf-8")

    config = load_config(_config(tmp_path, f"logging_config={json_file}\n"))
    assert config.logging.config_file == str(json_file)

    monkeypatch.setenv("GRAPHWISE_LOGGING_CONFIG", str(ini_file))
    config = load_config(_config(tmp_path, f"logging_config={json_file}\n"))
    assert config.logging.config_file == str(ini_file)


@pytest.mark.parametrize(
    "name, create, error",
    [
        ("missing.json", False, "not found"),
        ("logging.toml", True, "Unsupported logging config"),
    ],
)
def test_bad_logging_config_is_rejected(tmp_path, name, create, error):
    path = tmp_path / name
    if create:
        path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        load_config(_config(tmp_path, f"logging_config={path}\n"))


def test_default_format_constant_includes_request_id():
    assert "%(request_id)s" in LOG_FORMAT
