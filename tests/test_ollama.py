"""Local LLM (Ollama) detection + capability ranking + adaptive picks, build_llm
wiring, and wizard routing (menu vs key prompt)."""

from __future__ import annotations

import httpx
import pytest

from aria.config.schema import AriaConfig
from aria.llm.ollama import (
    ModelInfo,
    _parse_param_size,
    detect_ollama,
    pick_models,
    rank_models,
)


def _tags_payload(*specs):
    # specs: (name, parameter_size, family, size_bytes)
    return {
        "models": [
            {
                "name": n,
                "size": sz,
                "details": {"parameter_size": ps, "family": fam},
            }
            for (n, ps, fam, sz) in specs
        ]
    }


class _FakeResp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("bad", request=None, response=None)

    def json(self):
        return self._payload


# --- detection ------------------------------------------------------------
def test_parse_param_size():
    assert _parse_param_size("8.0B") == 8.0
    assert _parse_param_size("70.6B") == pytest.approx(70.6)
    assert _parse_param_size("1.5B") == 1.5
    assert _parse_param_size("350M") == pytest.approx(0.35)
    assert _parse_param_size("") == 0.0
    assert _parse_param_size(None) == 0.0


def test_detect_ollama_parses_api_tags(monkeypatch):
    payload = _tags_payload(
        ("llama3.1:8b", "8.0B", "llama", 4_700_000_000),
        ("qwen2.5:32b", "32.8B", "qwen2", 20_000_000_000),
    )
    monkeypatch.setattr(httpx, "get", lambda url, **k: _FakeResp(payload))
    models = detect_ollama()
    assert {m.name for m in models} == {"llama3.1:8b", "qwen2.5:32b"}
    big = next(m for m in models if m.name == "qwen2.5:32b")
    assert big.params_b == pytest.approx(32.8)
    assert big.size_bytes == 20_000_000_000


def test_detect_ollama_returns_empty_on_connection_error(monkeypatch):
    def boom(url, **k):
        raise httpx.ConnectError("ollama not running")

    monkeypatch.setattr(httpx, "get", boom)
    assert detect_ollama() == []  # never raises


def test_detect_ollama_falls_back_to_v1_models(monkeypatch):
    def fake_get(url, **k):
        if url.endswith("/api/tags"):
            raise httpx.ConnectError("no native api")
        return _FakeResp({"data": [{"id": "llama3.2:3b"}, {"id": "gemma2:2b"}]})

    monkeypatch.setattr(httpx, "get", fake_get)
    models = detect_ollama()
    assert {m.name for m in models} == {"llama3.2:3b", "gemma2:2b"}
    assert all(m.params_b == 0.0 for m in models)  # /v1 has no sizes


# --- ranking + picking ----------------------------------------------------
def test_pick_prefers_large_tool_capable_over_small(monkeypatch):
    monkeypatch.setattr("aria.llm.ollama._system_ram_bytes", lambda: 0)  # ignore RAM
    models = [
        ModelInfo("gemma2:2b", 2.0, "gemma"),       # tiny, NOT tool-capable
        ModelInfo("llama3.2:3b", 3.0, "llama"),     # small, tool-capable
        ModelInfo("llama3.1:8b", 8.0, "llama"),     # medium, tool-capable
        ModelInfo("qwen2.5:32b", 32.0, "qwen2"),    # large, tool-capable
    ]
    picks = pick_models(rank_models(models))
    assert picks["tool_capable"] is True
    assert picks["reasoning_model"] == "qwen2.5:32b"  # 32B, not the 8B
    assert picks["fast_model"] == "llama3.2:3b"       # smallest tool-capable >= 3B
    assert picks["synthesis_model"] is None            # 32B isn't "huge"


def test_huge_model_reused_as_synthesis(monkeypatch):
    monkeypatch.setattr("aria.llm.ollama._system_ram_bytes", lambda: 0)
    picks = pick_models(rank_models([ModelInfo("llama3.3:70b", 70.6, "llama")]))
    assert picks["reasoning_model"] == "llama3.3:70b"
    assert picks["synthesis_model"] == "llama3.3:70b"  # huge -> reuse


def test_only_tiny_non_tool_models_flags_not_capable(monkeypatch):
    monkeypatch.setattr("aria.llm.ollama._system_ram_bytes", lambda: 0)
    picks = pick_models(rank_models([
        ModelInfo("gemma2:2b", 2.0, "gemma"),
        ModelInfo("phi3:mini", 3.8, "phi3"),
    ]))
    assert picks["tool_capable"] is False  # warn the user; Groq recommended
    assert picks["reasoning_model"] in {"gemma2:2b", "phi3:mini"}


def test_ram_downgrade_picks_a_smaller_model_that_fits(monkeypatch):
    # 16 GB RAM; the 70B (~40GB) won't fit -> fall back to the 8B that does.
    monkeypatch.setattr("aria.llm.ollama._system_ram_bytes", lambda: 16 * 1024**3)
    models = [
        ModelInfo("llama3.3:70b", 70.0, "llama", size_bytes=40 * 1024**3),
        ModelInfo("llama3.1:8b", 8.0, "llama", size_bytes=5 * 1024**3),
    ]
    picks = pick_models(rank_models(models))
    assert picks["reasoning_model"] == "llama3.1:8b"
    assert picks["ram_warning"] and "llama3.3:70b" in picks["ram_warning"]


# --- build_llm wiring -----------------------------------------------------
def test_build_llm_ollama_needs_no_groq_key():
    from aria.app import build_llm
    from aria.llm.openai_compat import OpenAICompatProvider

    cfg = AriaConfig()
    cfg.llm.provider = "ollama"

    class _NoSecrets:
        def get(self, name):
            return None  # no keys at all

    llm = build_llm(cfg, _NoSecrets())  # must NOT raise MissingSecret
    assert isinstance(llm, OpenAICompatProvider)
    assert llm._base == "http://localhost:11434/v1"


def test_build_llm_groq_still_requires_key():
    from aria.app import MissingSecret, build_llm

    cfg = AriaConfig()  # provider defaults to groq

    class _NoSecrets:
        def get(self, name):
            return None

    with pytest.raises(MissingSecret):
        build_llm(cfg, _NoSecrets())
