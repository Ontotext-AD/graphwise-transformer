import configparser
import logging
import os
from dataclasses import dataclass


@dataclass
class AppConfig:
    port: int
    log_level: str
    default_model: str
    max_workers: int
    secret: str


def load_config(config_path: str | None = None) -> AppConfig:
    if config_path is None:
        config_path = os.environ.get("GRAPHWISE_CONFIG", os.path.join(os.getcwd(), "config.properties"))

    parser = configparser.ConfigParser()
    # Treat properties as INI without sections by adding a fake section
    with open(config_path, "r", encoding="utf-8") as f:
        content = f"[DEFAULT]\n" + f.read()
    parser.read_string(content)
    d = parser["DEFAULT"]

    port = int(d.get("port", 5050))
    log_level = d.get("log_level", "INFO")
    default_model = d.get("default_model", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    max_workers = int(d.get("max_workers", 8))
    secret = d.get("secret")

    logging.getLogger().setLevel(getattr(logging, log_level.upper(), logging.INFO))

    return AppConfig(
        port=port,
        log_level=log_level,
        default_model=default_model,
        max_workers=max_workers,
        secret=secret,
    )
