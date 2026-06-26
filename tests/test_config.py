"""Regression tests for the three first-hardware-run blockers:

1. config save must not crash on None fields (TOML has no null).
2. an invalid/missing Groq key must surface a friendly line, not a traceback.
3. an explicitly-set env var must take precedence over the stored keyring value.
"""

from __future__ import annotations

import aria.config.loader as loader
from aria.app import MissingSecret
from aria.config.keyring import SecretStore
from aria.config.schema import AriaConfig
from aria.core.runtime import _BAD_KEY_MSG, friendly_error
from aria.llm.base import LLMAuthError, LLMConnectionError


# --- Bug 1: config round-trips through TOML despite None fields ------------
def test_config_save_load_roundtrip_with_none_fields(tmp_path, monkeypatch):
    cfg_path = tmp_path / "config.toml"
    monkeypatch.setattr(loader, "config_path", lambda: cfg_path)

    original = AriaConfig()
    # These default to None and previously crashed tomlkit.dumps.
    assert original.llm.base_url is None
    assert original.stt.language is None
    assert original.tts.model_path is None
    assert original.user_name is None

    loader.save_config(original)  # must not raise ConvertError
    assert cfg_path.exists()

    reloaded = loader.load_config()
    assert reloaded == original  # None fields re-default on reload


def test_config_save_preserves_set_values(tmp_path, monkeypatch):
    cfg_path = tmp_path / "config.toml"
    monkeypatch.setattr(loader, "config_path", lambda: cfg_path)

    cfg = AriaConfig()
    cfg.user_name = "Sam"
    cfg.tts.voice = "en_GB-alba-medium"
    cfg.setup_complete = True
    loader.save_config(cfg)

    reloaded = loader.load_config()
    assert reloaded.user_name == "Sam"
    assert reloaded.tts.voice == "en_GB-alba-medium"
    assert reloaded.setup_complete is True


# --- Bug 2: bad key -> friendly message, not a traceback ------------------
def test_friendly_error_for_auth():
    assert friendly_error(LLMAuthError("401")) == _BAD_KEY_MSG
    assert friendly_error(MissingSecret("no key")) == _BAD_KEY_MSG


def test_friendly_error_for_connection():
    assert friendly_error(LLMConnectionError("dns")) is not None


def test_friendly_error_passes_through_unknown():
    # Unhandled errors return None so the caller re-raises (no silent swallow).
    assert friendly_error(ValueError("boom")) is None


def test_groq_sdk_auth_error_is_translated():
    # The real path: a Groq SDK AuthenticationError must become an LLMAuthError
    # so the runtime maps it to the friendly line instead of a raw traceback.
    import httpx
    from groq import AuthenticationError

    from aria.llm.groq_provider import _translate

    sdk_exc = AuthenticationError(
        "invalid api key",
        response=httpx.Response(401, request=httpx.Request("POST", "https://api.groq.com")),
        body=None,
    )
    translated = _translate(sdk_exc)
    assert isinstance(translated, LLMAuthError)
    assert friendly_error(translated) == _BAD_KEY_MSG


# --- Bug 3: explicit env var beats the stored keyring value ---------------
def test_env_var_takes_precedence_over_keyring(monkeypatch):
    monkeypatch.setattr(
        "aria.config.keyring.keyring.get_password", lambda *a, **k: "stored-stale-key"
    )
    monkeypatch.setenv("GROQ_API_KEY", "fresh-env-key")
    assert SecretStore().get("groq_api_key") == "fresh-env-key"


def test_keyring_used_when_env_unset(monkeypatch):
    monkeypatch.setattr(
        "aria.config.keyring.keyring.get_password", lambda *a, **k: "stored-key"
    )
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert SecretStore().get("groq_api_key") == "stored-key"
