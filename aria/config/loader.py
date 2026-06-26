"""TOML config load/save and XDG path resolution.

Config -> ~/.config/aria/config.toml
State  -> ~/.local/share/aria/   (sqlite memory, audit log, downloaded models)
"""

from __future__ import annotations

from pathlib import Path

import tomlkit
from platformdirs import user_config_path, user_data_path

from aria import APP_SLUG
from aria.config.schema import AriaConfig


def config_path() -> Path:
    return user_config_path(APP_SLUG) / "config.toml"


def state_dir() -> Path:
    d = user_data_path(APP_SLUG)
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_config() -> AriaConfig:
    """Load config, returning defaults if no file exists yet."""
    path = config_path()
    if not path.exists():
        return AriaConfig()
    raw = tomlkit.parse(path.read_text())
    return AriaConfig.model_validate(dict(raw))


def save_config(config: AriaConfig) -> Path:
    """Persist config as TOML. Never writes secrets (those go to keyring)."""
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # exclude_none=True: TOML has no null literal, so None fields must be
    # omitted (tomlkit raises ConvertError otherwise). load_config re-validates,
    # so omitted keys simply re-default to None on the way back in.
    data = config.model_dump(mode="json", exclude_none=True)
    path.write_text(tomlkit.dumps(data))
    # config may contain personalisation but never API keys.
    path.chmod(0o600)
    return path
