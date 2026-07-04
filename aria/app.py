"""Composition root.

Builds concrete providers from config + secrets and assembles the registry,
orchestrator, and voice pipeline. This is the one place that knows which
concrete classes back each interface, so swapping a provider is a one-line edit.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from aria.config.keyring import SecretStore
from aria.config.schema import AriaConfig
from aria.core.memory import Memory
from aria.core.orchestrator import Orchestrator
from aria.llm.base import LLMProvider
from aria.tools.base import ToolRegistry
from aria.voice.base import STT, TTS

if TYPE_CHECKING:
    from aria.core.scheduler import SchedulerService


class MissingSecret(RuntimeError):
    pass


def _google_service_provider(api: str, version: str, secrets: SecretStore):
    """A lazy, memoized provider for a Google API service. Builds it on first use
    (raising GoogleNotConnected until the user runs `aria connect google`); the
    underlying credentials refresh themselves, so the cached service stays valid."""
    cache: dict = {}

    def provider():
        from aria.integrations.google_auth import build_service

        if "svc" not in cache:
            cache["svc"] = build_service(api, version, secrets)  # raises if no token
        return cache["svc"]

    return provider


# Free fallback providers: (base_url, default_model). The user supplies the key
# (keyring/`ARIA_FALLBACK_API_KEY`) and sets `[llm] fallback_provider`.
_FALLBACK_PRESETS = {
    "cerebras": ("https://api.cerebras.ai/v1", "llama-3.3-70b"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "gemini-2.0-flash"),
    "openai_compat": (None, None),  # user must set fallback_base_url + fallback_model
}

# Engine presets for the commerce browser-agent: (base_url, default_model, key_name).
# Default is Gemini's FREE OpenAI-compatible endpoint (a separate AI Studio key in
# `commerce_api_key`, NOT the Google OAuth token) so it never spends the Groq budget.
_COMMERCE_PRESETS = {
    "gemini": (
        "https://generativelanguage.googleapis.com/v1beta/openai",
        "gemini-2.0-flash",
        "commerce_api_key",
    ),
    "groq": ("https://api.groq.com/openai/v1", "llama-3.3-70b-versatile", "groq_api_key"),
    "ollama": ("http://localhost:11434/v1", "llama3.1:8b", None),
    "openai_compat": (None, None, "commerce_api_key"),
}


def commerce_engine(config: AriaConfig, secrets: SecretStore) -> dict:
    """Resolve the LLM engine that drives the browser agent — base_url, model, and
    api_key — from config + secrets. Default (Gemini) has NO Groq dependency."""
    name = config.commerce.engine
    base, model, key_name = _COMMERCE_PRESETS.get(name, _COMMERCE_PRESETS["gemini"])
    base = config.commerce.base_url or base
    model = config.commerce.model or model
    # Ollama needs no key; everything else reads its key from the secret store.
    key = "ollama" if name == "ollama" else (secrets.get(key_name) if key_name else "")
    return {"engine": name, "base_url": base, "model": model, "api_key": key or ""}


def _build_fallback(config: AriaConfig, secrets: SecretStore) -> LLMProvider | None:
    """Build the optional free fallback provider, or None if unconfigured."""
    name = config.llm.fallback_provider
    if not name:
        return None
    key = secrets.get("fallback_api_key")
    if not key:
        return None  # configured but no key -> silently skip (primary still works)
    base, model = _FALLBACK_PRESETS.get(name, (None, None))
    base = config.llm.fallback_base_url or base
    model = config.llm.fallback_model or model
    if not (base and model):
        return None
    from aria.llm.openai_compat import OpenAICompatProvider

    return OpenAICompatProvider(key, base, timeout=config.llm.request_timeout_s)


def build_llm(config: AriaConfig, secrets: SecretStore) -> LLMProvider:
    provider = config.llm.provider
    if provider in ("ollama", "openai_compat"):
        # Local (Ollama) or any OpenAI-compatible server. No API key required —
        # Ollama ignores the bearer token, so we send a harmless placeholder and
        # NEVER raise MissingSecret here (that's the whole point of local mode).
        from aria.llm.openai_compat import OpenAICompatProvider

        base = config.llm.base_url or "http://localhost:11434/v1"
        key = secrets.get("fallback_api_key") or "ollama"
        return OpenAICompatProvider(key, base, timeout=config.llm.request_timeout_s)
    if provider == "groq":
        key = secrets.get("groq_api_key")
        if not key:
            raise MissingSecret("No Groq API key. Run `aria setup`.")
        from aria.llm.groq_provider import GroqProvider

        primary: LLMProvider = GroqProvider(key, timeout=config.llm.request_timeout_s)
        llm: LLMProvider = primary
        fallback = _build_fallback(config, secrets)
        if fallback is not None:
            from aria.llm.fallback import FallbackProvider

            fb_model = config.llm.fallback_model or _FALLBACK_PRESETS.get(
                config.llm.fallback_provider, (None, None)
            )[1]
            llm = FallbackProvider(primary, fallback, fb_model)
        if config.llm.local_fallback:
            # Outermost layer: if the whole cloud chain is down/rate-limited and a
            # local Ollama exists, the turn runs locally instead of apologizing.
            from aria.llm.local_fallback import LocalFallbackProvider

            llm = LocalFallbackProvider(llm, fast_model=config.llm.fast_model)
        return llm
    raise NotImplementedError(f"LLM provider '{config.llm.provider}' not wired yet")


def build_stt(config: AriaConfig, secrets: SecretStore) -> STT:
    if config.stt.provider == "groq":
        key = secrets.get("groq_api_key")
        if not key:
            raise MissingSecret("No Groq API key for STT.")
        from aria.voice.stt_groq import GroqSTT

        return GroqSTT(key, model=config.stt.model)
    if config.stt.provider == "faster_whisper":
        from aria.voice.stt_faster_whisper import FasterWhisperSTT

        return FasterWhisperSTT(config.stt.local_model_size, config.stt.local_compute_type)
    raise NotImplementedError(f"STT provider '{config.stt.provider}' not wired yet")


def resolve_piper_model(config: AriaConfig) -> Path:
    """Find the bundled/installed Piper voice .onnx for the configured voice.

    Search order: explicit config path, then ``$ARIA_MODELS_DIR`` (set by the
    packaged ``/usr/bin/aria`` launcher to ``/opt/aria/models``), then the
    in-package copy, then the per-user state dir (where the wizard downloads).
    """
    import os

    if config.tts.model_path:
        return Path(config.tts.model_path).expanduser()
    from aria.config.loader import state_dir

    voice = f"{config.tts.voice}.onnx"
    candidates: list[Path] = []
    env_dir = os.environ.get("ARIA_MODELS_DIR")
    if env_dir:
        candidates.append(Path(env_dir) / voice)
    candidates.append(Path(__file__).parent / "packaging" / "models" / voice)
    candidates.append(state_dir() / "models" / voice)
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]  # let PiperTTS raise a clear FileNotFoundError


def resolve_kokoro_paths(config: AriaConfig) -> tuple[Path, Path]:
    """Find the shared Kokoro model + voice-pack files. Same search order as the
    Piper resolver: ``$ARIA_MODELS_DIR``, the in-package copy, the state dir."""
    import os

    from aria.config.loader import state_dir
    from aria.voice.tts_kokoro import MODEL_FILE, VOICES_FILE

    dirs: list[Path] = []
    env_dir = os.environ.get("ARIA_MODELS_DIR")
    if env_dir:
        dirs.append(Path(env_dir))
    dirs.append(Path(__file__).parent / "packaging" / "models")
    dirs.append(state_dir() / "models")
    for d in dirs:
        if (d / MODEL_FILE).exists() and (d / VOICES_FILE).exists():
            return d / MODEL_FILE, d / VOICES_FILE
    # Let KokoroTTS raise its clear FileNotFoundError for the preferred location.
    return dirs[-1] / MODEL_FILE, dirs[-1] / VOICES_FILE


def build_tts(config: AriaConfig) -> TTS:
    if config.tts.provider == "kokoro":
        from aria.voice.tts_kokoro import KokoroTTS

        try:
            model, voices = resolve_kokoro_paths(config)
            return KokoroTTS(model, voices, voice=config.tts.voice, speed=config.tts.speed)
        except FileNotFoundError as exc:
            # Degrade to the bundled Piper voice rather than dying voiceless —
            # but if no Piper voice exists either, surface the Kokoro error
            # (it names the fix: run `aria setup`).
            import logging

            from aria.voice.tts_piper import PiperTTS

            fallback = config.model_copy(deep=True)
            fallback.tts.provider = "piper"
            fallback.tts.voice = "en_US-amy-medium"
            fallback.tts.model_path = None
            piper_path = resolve_piper_model(fallback)
            if not piper_path.exists():
                raise exc from None
            logging.getLogger("aria").warning(
                "Kokoro voice unavailable (%s); falling back to Piper %s.",
                exc, fallback.tts.voice,
            )
            return PiperTTS(piper_path, speed=config.tts.speed)
    if config.tts.provider == "piper":
        from aria.voice.tts_piper import PiperTTS

        return PiperTTS(resolve_piper_model(config), speed=config.tts.speed)
    raise NotImplementedError(f"TTS provider '{config.tts.provider}' not wired yet")


async def build_registry(
    config: AriaConfig,
    llm: LLMProvider,
    memory: Memory,
    scheduler: SchedulerService,
    secrets: SecretStore | None = None,
) -> tuple[ToolRegistry, list]:
    """Assemble native tools + MCP tools + specialist sub-agents into a registry.

    Returns (registry, mcp_managers) so the caller can close MCP connections.
    """
    from aria.agents.specialists import build_specialists
    from aria.mcp.client import MCPManager
    from aria.tools.calendar_tool import calendar_tools
    from aria.tools.commerce import commerce_tools
    from aria.tools.errands import errand_tools
    from aria.tools.files import file_tools
    from aria.tools.gmail_tool import gmail_tools
    from aria.tools.math_tool import MathTool
    from aria.tools.memory_tool import memory_tools
    from aria.tools.search import WebSearchTool
    from aria.tools.system import system_tools
    from aria.tools.timeinfo import TimeTool
    from aria.tools.timers import timer_tools
    from aria.tools.weather import WeatherTool
    from aria.tools.web import GetHeadlinesTool, ReadWebpageTool

    secrets = secrets or SecretStore()

    registry = ToolRegistry()
    registry.register_all(
        [
            WebSearchTool(),
            ReadWebpageTool(),
            GetHeadlinesTool(),
            WeatherTool(config.home_location),
            MathTool(),
            TimeTool(),
            *timer_tools(scheduler),
            *system_tools(),
            *file_tools(),
            *memory_tools(memory),
            *calendar_tools(_google_service_provider("calendar", "v3", secrets)),
            *gmail_tools(_google_service_provider("gmail", "v1", secrets)),
            *commerce_tools(lambda: config.commerce, lambda: commerce_engine(config, secrets)),
            *errand_tools(),
        ]
    )

    mcp_tools: list = []
    manager = MCPManager(config.mcp_servers)
    if config.mcp_servers:
        mcp_tools = await manager.connect_all()
        registry.register_all(mcp_tools)

    # Specialist sub-agents (callable as tools). Research/files/comms do multi-step
    # read-and-synthesize work and need the reasoning model; trivial agents use 8B.
    for agent_tool in build_specialists(
        llm, config.llm.reasoning_model, config.llm.fast_model, mcp_tools
    ):
        registry.register(agent_tool)

    return registry, [manager]


async def build_orchestrator(
    config: AriaConfig,
    secrets: SecretStore,
    memory: Memory,
    *,
    scheduler: SchedulerService | None = None,
    voice: bool = False,
) -> tuple[Orchestrator, list]:
    from aria.core.scheduler import SchedulerService

    llm = build_llm(config, secrets)
    managers: list = []
    if scheduler is None:
        # No scheduler supplied (text mode / scripts): make one so timer tools
        # persist, and hand it to the caller for cleanup via the manager list. It
        # isn't started here, so alarms persist but won't fire without a run loop.
        scheduler = SchedulerService()
        await scheduler.open()
        managers.append(scheduler)
    registry, mcp_managers = await build_registry(config, llm, memory, scheduler, secrets)
    managers.extend(mcp_managers)
    orch = Orchestrator(
        llm=llm,
        registry=registry,
        memory=memory,
        reasoning_model=config.llm.reasoning_model,
        fast_model=config.llm.fast_model,
        synthesis_model=config.llm.synthesis_model,
        require_confirmation=config.safety.require_confirmation,
        voice=voice,
    )
    return orch, managers
