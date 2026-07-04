"""Typed configuration schema.

All tunables live here so model names, voices, and provider choices are config,
never hardcoded. Secrets do NOT live here — they go to the OS keyring
(see :mod:`aria.config.keyring`).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    provider: str = "groq"  # groq | openai | anthropic | ollama
    # Tool-calling / reasoning model (kept on llama-3.3-70b for reliable Groq
    # tool calls) and the small fast router/intent model.
    reasoning_model: str = "llama-3.3-70b-versatile"
    fast_model: str = "llama-3.1-8b-instant"
    # Optional STRONGER model used only for the final spoken synthesis (not for
    # tool-calling). Default None = just use reasoning_model (no failed startup
    # probe). Set it to a model your account actually has — run scripts/pick_model.py
    # to see what's available — e.g. "moonshotai/kimi-k2-instruct".
    synthesis_model: str | None = None
    base_url: str | None = None  # for openai-compatible / ollama
    temperature: float = 0.4
    max_tokens: int = 1024
    request_timeout_s: float = 30.0

    # Optional FREE fallback provider used only when the primary (Groq) is rate-
    # limited or unreachable, so a daily cap never takes Aria offline. Behind the
    # same LLMProvider interface; primary stays Groq. Presets fill base_url/model.
    fallback_provider: str | None = None  # cerebras | gemini | openai_compat | None
    fallback_base_url: str | None = None  # overrides the preset if set
    fallback_model: str | None = None     # overrides the preset's default model

    # Last resort behind everything above: if the cloud chain is rate-limited or
    # unreachable AND a local Ollama with a usable model is running, the turn runs
    # locally (models picked adaptively). Free, unlimited, offline. Costs nothing
    # when Ollama isn't installed (detection is lazy, only after a cloud failure).
    local_fallback: bool = True


class ActivationConfig(BaseModel):
    """How Aria starts listening.

    * ``wake_word`` — say the wake phrase ("hey jarvis").
    * ``hotkey``    — hold a key, talk, release (walkie-talkie; needs read access
                      to /dev/input, i.e. membership of the ``input`` group).
    * ``hybrid``    — both at once.
    A short rising chime confirms she's actually listening (``chime``)."""

    mode: str = "wake_word"  # wake_word | hotkey | hybrid
    hotkey: str = "right ctrl"  # see aria.voice.hotkey.KEY_CHOICES
    chime: bool = True


class CommerceConfig(BaseModel):
    """Agentic web ordering (food delivery, v1). The browser agent reaches the
    checkout/payment page and STOPS — it never pays. Address is PII: stored locally
    only and redacted from the audit trail (the order_food tool is ``sensitive``)."""

    # Which LLM drives the browser agent. Default is Gemini 2.0 Flash via its free
    # OpenAI-compatible endpoint (key in SecretStore as ``commerce_api_key``), so
    # this never touches the Groq budget. Swappable to groq | ollama | openai_compat.
    engine: str = "gemini"
    base_url: str | None = None  # overrides the engine preset's base_url
    model: str | None = None     # overrides the engine preset's default model

    # The user's ordering profile (set in the wizard reconfigure menu).
    delivery_address: str | None = None
    dietary_prefs: str | None = None             # e.g. "vegetarian, no nuts"
    favorite_vendors: list[str] = Field(default_factory=list)
    default_food_app: str | None = None          # e.g. "Uber Eats", "Deliveroo"
    max_order_value: float | None = None         # read-back ceiling, in local currency

    # Hard budgets so a stuck browse never loops forever, and headful so the user
    # can take over at the payment page.
    max_steps: int = 40
    max_seconds: float = 240.0
    headful: bool = True


class STTConfig(BaseModel):
    provider: str = "groq"  # groq | faster_whisper
    model: str = "whisper-large-v3-turbo"
    language: str | None = None  # None = autodetect
    # Local fallback settings (used when provider == faster_whisper).
    local_model_size: str = "base.en"
    local_compute_type: str = "int8"


class TTSConfig(BaseModel):
    provider: str = "piper"  # piper | kokoro
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
    # Speech is considered ended after this much trailing silence. Kept short so she
    # reacts soon after you stop talking (lower endpoint lag → faster first word).
    silence_ms: int = 450
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


class ConversationConfig(BaseModel):
    """Flowing conversation: after Aria speaks, the mic re-opens briefly with NO
    wake word so the user can just keep talking. A fast-LLM relevance gate drops
    background speech (TV, side conversations) so she never butts in uninvited."""

    enabled: bool = True
    # How long she listens for a follow-up before falling back to wake-word idle.
    # Kept SHORT on purpose: a long open mic feels like being watched, and every
    # extra second is another chance for background noise to become a ghost turn.
    window_s: float = 4.0


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
    commerce: CommerceConfig = Field(default_factory=CommerceConfig)
    stt: STTConfig = Field(default_factory=STTConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    wakeword: WakeWordConfig = Field(default_factory=WakeWordConfig)
    activation: ActivationConfig = Field(default_factory=ActivationConfig)
    vad: VADConfig = Field(default_factory=VADConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    conversation: ConversationConfig = Field(default_factory=ConversationConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    mcp_servers: list[MCPServer] = Field(default_factory=list)

    # User-facing personalisation.
    user_name: str | None = None
    # Default city for "what's the weather" with no city named (set in the wizard).
    home_location: str | None = None
    # First-run wizard sets this True once setup completes.
    setup_complete: bool = False
