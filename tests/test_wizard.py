"""Wizard routing: reconfigure opens a MENU (no Groq key prompt); first-run routes
to provider/key by what's detected; local config needs no secret."""

from __future__ import annotations

import aria.tui.wizard as wiz
from aria.config.schema import AriaConfig


class _SpyStore:
    """A SecretStore stand-in that EXPLODES if anyone touches the Groq key, so a
    test can prove the voice/menu paths never read or write it."""

    def __init__(self):
        self.touched: list[str] = []

    def _guard(self, name):
        self.touched.append(name)
        if name == "groq_api_key":
            raise AssertionError("groq_api_key must not be accessed here")

    def has(self, name):
        self._guard(name)
        return False

    def get(self, name):
        self._guard(name)
        return None

    def set(self, name, value, **k):
        self._guard(name)
        return "keyring"


def _make_wizard(monkeypatch, config, *, ollama=None, secrets=None):
    monkeypatch.setattr(wiz, "load_config", lambda: config)
    monkeypatch.setattr(wiz, "SecretStore", lambda: secrets or _SpyStore())
    monkeypatch.setattr(wiz, "detect_ollama", lambda *a, **k: ollama or [])
    return wiz.WizardApp()


# --- the core fix: reconfigure shows the menu, never the key prompt --------
def test_reconfigure_opens_menu_not_key_prompt(monkeypatch):
    cfg = AriaConfig()
    cfg.setup_complete = True
    app = _make_wizard(monkeypatch, cfg)  # _SpyStore raises on any groq access
    assert app.mode == "menu"
    assert app.step == "menu"  # NOT "key"
    # Building the menu wizard must not have probed for the Groq key at all.
    assert "groq_api_key" not in app.secrets.touched


def test_menu_actions_never_route_to_key_prompt():
    # Only the explicit "change provider -> Groq" path can reach the key step.
    assert "key" not in wiz.MENU_NEXT.values()
    assert wiz.MENU_NEXT["act_voice"] == "voice"
    assert wiz.MENU_NEXT["act_provider"] == "provider"


# --- first-run routing ----------------------------------------------------
def test_first_run_without_ollama_goes_to_key(monkeypatch):
    cfg = AriaConfig()  # setup_complete defaults False
    app = _make_wizard(monkeypatch, cfg, ollama=[])
    assert app.mode == "setup"
    assert app.step == "key"  # no local models -> Groq key, exactly as before


def test_first_run_with_ollama_offers_provider_choice(monkeypatch):
    from aria.llm.ollama import ModelInfo

    cfg = AriaConfig()
    app = _make_wizard(monkeypatch, cfg, ollama=[ModelInfo("llama3.1:8b", 8.0, "llama")])
    assert app.step == "provider"
    assert app.picks and app.picks["reasoning_model"] == "llama3.1:8b"


# --- local config application is key-free ---------------------------------
def test_apply_local_config_sets_provider_and_models_without_secret():
    cfg = AriaConfig()
    picks = {
        "reasoning_model": "qwen2.5:32b",
        "fast_model": "llama3.2:3b",
        "synthesis_model": None,
        "tool_capable": True,
        "reasoning_params_b": 32.0,
        "ram_warning": None,
    }
    warning = wiz.apply_local_config(cfg, picks)
    assert cfg.llm.provider == "ollama"
    assert cfg.llm.base_url == "http://localhost:11434/v1"
    assert cfg.llm.reasoning_model == "qwen2.5:32b"
    assert cfg.llm.fast_model == "llama3.2:3b"
    # STT either goes fully local (faster-whisper) or warns it needs a Groq key.
    assert cfg.stt.provider in {"faster_whisper", "groq"}
    if cfg.stt.provider == "groq":
        assert warning and "faster-whisper" in warning


def test_local_to_groq_resets_models_to_schema_defaults():
    from aria.config.schema import LLMConfig

    cfg = AriaConfig()
    # Simulate a prior Local switch that left Ollama names behind.
    cfg.llm.provider = "ollama"
    cfg.llm.reasoning_model = "qwen2.5:3b"
    cfg.llm.fast_model = "qwen2.5:3b"
    cfg.llm.synthesis_model = "qwen2.5:32b"
    cfg.llm.base_url = "http://localhost:11434/v1"
    cfg.stt.provider = "faster_whisper"

    wiz.apply_groq_config(cfg)

    d = LLMConfig.model_fields
    assert cfg.llm.provider == "groq"
    assert cfg.llm.reasoning_model == d["reasoning_model"].default  # back to Groq model
    assert cfg.llm.fast_model == d["fast_model"].default
    assert cfg.llm.synthesis_model is None
    assert cfg.llm.base_url is None  # no stale localhost:11434 -> no 404
    assert cfg.stt.provider == "groq"


def test_describe_picks_warns_when_not_tool_capable():
    line = wiz.describe_picks(
        {"reasoning_model": "gemma2:2b", "tool_capable": False, "reasoning_params_b": 2.0}
    )
    assert "may NOT reliably perform actions" in line


def test_describe_picks_celebrates_tool_capable():
    line = wiz.describe_picks(
        {"reasoning_model": "qwen2.5:32b", "tool_capable": True,
         "reasoning_params_b": 32.0, "ram_warning": None}
    )
    assert "tool-capable" in line and "qwen2.5:32b" in line
