"""Typed configuration schema.

All tunables live here so model names, voices, and provider choices are config,
never hardcoded. Secrets do NOT live here — they go to the OS keyring
(see :mod:`aria.config.keyring`).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    provider: str = "groq"  # groq | openai | anthropic | ollama
    # Default reasoning model and the small fast router/intent model.
    reasoning_model: str = "llama-3.3-70b-versatile"
    fast_model: str = "llama-3.1-8b-instant"
    base_url: str | None = None  # for openai-compatible / ollama
    temperature: float = 0.4
    max_tokens: int = 1024
    request_timeout_s: float = 30.0


class STTConfig(BaseModel):
    provider: str = "groq"  # groq | faster_whisper
    model: str = "whisper-large-v3-turbo"
    language: str | None = None  # None = autodetect
    # Local fallback settings (used when provider == faster_whisper).
    local_model_size: str = "base.en"
    local_compute_type: str = "int8"


class TTSConfig(BaseModel):
    provider: str = "piper"  # piper | (cloud later)
    voice: str = "en_US-amy-medium"
    # Path to the bundled piper voice (.onnx). Resolved at runtime if None.
    model_path: str | None = None
    speed: float = 1.0
    sample_rate: int = 22050


class WakeWordConfig(BaseModel):
    enabled: bool = True
    model: str = "hey_jarvis"  # openWakeWord model name (placeholder)
    threshold: float = 0.5


class VADConfig(BaseModel):
    backend: str = "silero"  # silero | energy
    # Speech is considered ended after this much trailing silence.
    silence_ms: int = 700
    speech_threshold: float = 0.5
    # Barge-in (talk over Aria to stop her) is OFF by default: on a laptop's
    # built-in mic+speakers there's no echo cancellation, so her own voice leaks
    # into the mic and would self-interrupt her mid-answer. Opt in only if you
    # have a headset / good echo isolation. When on, an energy gate (mic level
    # must clearly exceed her current output) reduces false self-triggers.
    barge_in: bool = False


class AudioConfig(BaseModel):
    sample_rate: int = 16000
    block_ms: int = 30
    input_device: int | str | None = None
    output_device: int | str | None = None


class SafetyConfig(BaseModel):
    # If True, "confirm"-class actions require explicit yes; if False they are
    # still logged but auto-approved (useful for trusted headless runs).
    require_confirmation: bool = True
    audit_log: bool = True


class MCPServer(BaseModel):
    name: str
    command: str | None = None  # stdio server launch command
    args: list[str] = Field(default_factory=list)
    url: str | None = None  # for http/sse servers
    enabled: bool = True


class AriaConfig(BaseModel):
    """Top-level config persisted to ~/.config/aria/config.toml."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    stt: STTConfig = Field(default_factory=STTConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    wakeword: WakeWordConfig = Field(default_factory=WakeWordConfig)
    vad: VADConfig = Field(default_factory=VADConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    mcp_servers: list[MCPServer] = Field(default_factory=list)

    # User-facing personalisation.
    user_name: str | None = None
    # First-run wizard sets this True once setup completes.
    setup_complete: bool = False
