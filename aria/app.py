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


def build_llm(config: AriaConfig, secrets: SecretStore) -> LLMProvider:
    if config.llm.provider == "groq":
        key = secrets.get("groq_api_key")
        if not key:
            raise MissingSecret("No Groq API key. Run `aria setup`.")
        from aria.llm.groq_provider import GroqProvider

        return GroqProvider(key, timeout=config.llm.request_timeout_s)
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


def build_tts(config: AriaConfig) -> TTS:
    if config.tts.provider == "piper":
        from aria.voice.tts_piper import PiperTTS

        return PiperTTS(resolve_piper_model(config), speed=config.tts.speed)
    raise NotImplementedError(f"TTS provider '{config.tts.provider}' not wired yet")


async def build_registry(
    config: AriaConfig, llm: LLMProvider, memory: Memory, scheduler: SchedulerService
) -> tuple[ToolRegistry, list]:
    """Assemble native tools + MCP tools + specialist sub-agents into a registry.

    Returns (registry, mcp_managers) so the caller can close MCP connections.
    """
    from aria.agents.specialists import build_specialists
    from aria.mcp.client import MCPManager
    from aria.tools.files import file_tools
    from aria.tools.math_tool import MathTool
    from aria.tools.memory_tool import memory_tools
    from aria.tools.search import WebSearchTool
    from aria.tools.system import system_tools
    from aria.tools.timeinfo import TimeTool
    from aria.tools.timers import timer_tools
    from aria.tools.web import ReadWebpageTool

    registry = ToolRegistry()
    registry.register_all(
        [
            WebSearchTool(),
            ReadWebpageTool(),
            MathTool(),
            TimeTool(),
            *timer_tools(scheduler),
            *system_tools(),
            *file_tools(),
            *memory_tools(memory),
        ]
    )

    mcp_tools: list = []
    manager = MCPManager(config.mcp_servers)
    if config.mcp_servers:
        mcp_tools = await manager.connect_all()
        registry.register_all(mcp_tools)

    # Specialist sub-agents (callable as tools) — use the fast model internally.
    for agent_tool in build_specialists(llm, config.llm.fast_model, mcp_tools):
        registry.register(agent_tool)

    return registry, [manager]


async def build_orchestrator(
    config: AriaConfig,
    secrets: SecretStore,
    memory: Memory,
    *,
    scheduler: SchedulerService | None = None,
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
    registry, mcp_managers = await build_registry(config, llm, memory, scheduler)
    managers.extend(mcp_managers)
    orch = Orchestrator(
        llm=llm,
        registry=registry,
        memory=memory,
        reasoning_model=config.llm.reasoning_model,
        fast_model=config.llm.fast_model,
        require_confirmation=config.safety.require_confirmation,
    )
    return orch, managers
