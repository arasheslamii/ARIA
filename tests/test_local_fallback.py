"""Automatic local (Ollama) fallback: when the cloud is rate-limited or offline,
the turn runs on the user's local models instead of apologizing."""

from __future__ import annotations

import pytest

import aria.llm.local_fallback as lf
from aria.config.hardware import MachineProfile, recommend_local_model
from aria.llm.base import ChatResult, LLMConnectionError, LLMRateLimitError
from aria.llm.local_fallback import LocalFallbackProvider
from aria.llm.ollama import ModelInfo


class LimitedCloud:
    """A primary that is always rate-limited (the user's actual situation)."""

    def __init__(self, exc=LLMRateLimitError) -> None:
        self.exc = exc
        self.calls = 0

    async def chat(self, messages, *, model, tools=None, temperature=None, max_tokens=None):
        self.calls += 1
        raise self.exc("TPD limit reached")

    async def stream(self, messages, *, model, temperature=None, max_tokens=None):
        raise self.exc("TPD limit reached")
        yield ""  # pragma: no cover - makes this an async generator


class LocalSpy:
    def __init__(self) -> None:
        self.chat_models: list[str] = []
        self.stream_models: list[str] = []

    async def chat(self, messages, *, model, tools=None, temperature=None, max_tokens=None):
        self.chat_models.append(model)
        return ChatResult(content="local answer", model=model)

    async def stream(self, messages, *, model, temperature=None, max_tokens=None):
        self.stream_models.append(model)
        for w in ("local ", "stream"):
            yield w


def _stub_detection(monkeypatch, models=None):
    found = models if models is not None else [
        ModelInfo(name="llama3.1:latest", params_b=8.0),
        ModelInfo(name="qwen2.5:3b", params_b=3.1),
    ]
    calls = {"n": 0}

    def fake_detect(base):
        calls["n"] += 1
        return found

    monkeypatch.setattr(lf, "detect_ollama", fake_detect)
    return calls


async def test_rate_limited_chat_runs_locally_with_mapped_models(monkeypatch):
    calls = _stub_detection(monkeypatch)
    local = LocalSpy()
    provider = LocalFallbackProvider(
        LimitedCloud(), fast_model="llama-3.1-8b-instant", local=local
    )
    # A reasoning-model call maps to the biggest local tool-capable model...
    out = await provider.chat([], model="llama-3.3-70b-versatile")
    assert out.content == "local answer"
    assert local.chat_models == ["llama3.1:latest"]
    # ...and a fast/router call maps to the small local pick.
    await provider.chat([], model="llama-3.1-8b-instant")
    assert local.chat_models[-1] == "qwen2.5:3b"
    assert calls["n"] == 1  # detection ran once, then cached


async def test_offline_cloud_also_falls_back_and_streams_locally(monkeypatch):
    _stub_detection(monkeypatch)
    local = LocalSpy()
    provider = LocalFallbackProvider(
        LimitedCloud(LLMConnectionError), fast_model="fast", local=local
    )
    text = "".join([d async for d in provider.stream([], model="llama-3.3-70b-versatile")])
    assert text == "local stream"
    assert local.stream_models == ["llama3.1:latest"]


async def test_no_ollama_means_the_original_error_surfaces(monkeypatch):
    _stub_detection(monkeypatch, models=[])  # nothing installed / not running
    provider = LocalFallbackProvider(LimitedCloud(), fast_model="fast")
    with pytest.raises(LLMRateLimitError):
        await provider.chat([], model="big")


async def test_local_failure_reports_the_cloud_story(monkeypatch):
    # Ollama detected but its call dies -> the user should hear the TRUE cause
    # (rate limit), not a misleading "offline" from the dead local server.
    _stub_detection(monkeypatch)

    class DeadLocal(LocalSpy):
        async def chat(self, messages, *, model, tools=None, temperature=None,
                       max_tokens=None):
            raise LLMConnectionError("ollama stopped")

    provider = LocalFallbackProvider(
        LimitedCloud(), fast_model="fast", local=DeadLocal()
    )
    with pytest.raises(LLMRateLimitError):
        await provider.chat([], model="big")
    assert provider._picks is None  # re-detects next time instead of trusting stale


async def test_partial_cloud_stream_is_never_restarted(monkeypatch):
    # If the cloud died MID-sentence, replaying from local would repeat words.
    _stub_detection(monkeypatch)

    class MidStreamDeath:
        async def chat(self, *a, **k):
            raise AssertionError("unused")

        async def stream(self, messages, *, model, temperature=None, max_tokens=None):
            yield "Half an ans"
            raise LLMRateLimitError("died mid-stream")

    local = LocalSpy()
    provider = LocalFallbackProvider(MidStreamDeath(), fast_model="fast", local=local)
    got = []
    with pytest.raises(LLMRateLimitError):
        async for d in provider.stream([], model="big"):
            got.append(d)
    assert got == ["Half an ans"]
    assert local.stream_models == []  # local was NOT consulted


async def test_healthy_cloud_never_touches_detection(monkeypatch):
    calls = _stub_detection(monkeypatch)

    class HealthyCloud:
        async def chat(self, messages, *, model, tools=None, temperature=None,
                       max_tokens=None):
            return ChatResult(content="cloud answer", model=model)

        async def stream(self, messages, *, model, temperature=None, max_tokens=None):
            yield "cloud"

    provider = LocalFallbackProvider(HealthyCloud(), fast_model="fast")
    assert (await provider.chat([], model="big")).content == "cloud answer"
    assert calls["n"] == 0  # zero cost for users without Ollama


# --- hardware probe -> recommendation ---------------------------------------
def _profile(ram, disk=100.0, gpu=None):
    return MachineProfile(ram_gb=ram, cpu_cores=8, gpu=gpu, free_disk_gb=disk)


def test_recommendation_scales_with_ram():
    assert recommend_local_model(_profile(32))[0] == "qwen2.5:14b"
    assert recommend_local_model(_profile(16))[0] == "qwen2.5:7b"
    assert recommend_local_model(_profile(8))[0] == "qwen2.5:3b"
    assert recommend_local_model(_profile(4))[0] == "qwen2.5:1.5b"
    model, note = recommend_local_model(_profile(2))
    assert model is None and "too little RAM" in note


def test_recommendation_respects_free_disk_and_mentions_gpu():
    model, note = recommend_local_model(_profile(16, disk=5.0))
    assert model is None and "free disk" in note
    _, note = recommend_local_model(_profile(16, gpu="NVIDIA RTX 4060"))
    assert "RTX 4060" in note
    _, note = recommend_local_model(_profile(16))
    assert "CPU" in note  # honest about local speed without a GPU


def test_local_fallback_is_on_by_default_for_groq_users():
    from aria.config.schema import AriaConfig

    assert AriaConfig().llm.local_fallback is True
