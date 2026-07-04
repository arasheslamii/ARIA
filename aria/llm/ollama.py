"""Local LLM via Ollama: detection, capability ranking, and adaptive model-picking.

Ollama serves an OpenAI-compatible endpoint (…/v1/chat/completions), so we reuse
:class:`aria.llm.openai_compat.OpenAICompatProvider` to actually talk to it — this
module is ONLY about discovering what the user has installed locally and choosing
the best models for them. The choice is adaptive (a 32B beats an 8B; a tool-capable
model beats a tiny instruct-only one) rather than hardcoded to specific names.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import httpx

_DEFAULT_BASE = "http://localhost:11434"
_DETECT_TIMEOUT_S = 1.5

# Model families / name prefixes that reliably do OpenAI-style tool calls. Aria is
# agentic (timers, email, calendar, system control), so tool-calling is the single
# most important capability. Tiny instruct-only models (gemma2:2b, phi3:mini) do
# NOT reliably emit tool calls, so they're flagged non-tool-capable below.
_TOOL_CAPABLE_PREFIXES = (
    "llama3.1", "llama3.2", "llama3.3", "llama4",
    "llama-3.1", "llama-3.2", "llama-3.3",
    "qwen2.5", "qwen3", "qwen2",
    "mistral", "mixtral", "mistral-nemo", "mistral-small", "ministral",
    "command-r", "hermes", "firefunction", "nous-hermes",
)
# Small/instruct-only families that generally can't tool-call — checked first so a
# name like "gemma2:27b" is treated as non-tool even though it's large.
_NON_TOOL_PREFIXES = (
    "gemma", "phi", "phi3", "phi4", "tinyllama", "tinydolphin",
    "smollm", "orca-mini", "stablelm", "deepseek-coder",
)

# A tool-capable model is treated as if it were this many billions of params bigger
# when ranking — enough to win a similar-size tie, not enough to beat a much larger
# model outright. The real driver of the score stays the parameter count.
_TOOL_BONUS_B = 2.0
# A "huge" model worth reusing as the synthesis model (vs. just None).
_HUGE_PARAMS_B = 65.0
# Smallest size we want for the fast router/intent model (snappy, still tool-capable).
_FAST_MIN_B = 3.0


@dataclass
class ModelInfo:
    name: str
    params_b: float = 0.0  # parameter count in billions (0.0 if unknown)
    family: str = ""
    size_bytes: int = 0

    @property
    def tool_capable(self) -> bool:
        return _is_tool_capable(self.name, self.family)


def _is_tool_capable(name: str, family: str = "") -> bool:
    n = (name or "").lower()
    if any(n.startswith(p) for p in _NON_TOOL_PREFIXES):
        return False
    if any(n.startswith(p) for p in _TOOL_CAPABLE_PREFIXES):
        return True
    f = (family or "").lower()
    return any(f.startswith(p) for p in _TOOL_CAPABLE_PREFIXES)


def _parse_param_size(text: object) -> float:
    """'8.0B' -> 8.0, '70.6B' -> 70.6, '1.5B' -> 1.5, '350M' -> 0.35, '' -> 0.0."""
    if not text:
        return 0.0
    s = str(text).strip().upper().replace(",", "")
    mult = 1.0
    if s.endswith("B"):
        s = s[:-1]
    elif s.endswith("M"):
        s, mult = s[:-1], 0.001
    try:
        return float(s) * mult
    except ValueError:
        return 0.0


def _normalize_base(base: str) -> str:
    """Accept either the native base (…:11434) or the OpenAI base (…:11434/v1)."""
    b = base.rstrip("/")
    if b.endswith("/v1"):
        b = b[: -len("/v1")]
    return b


def detect_ollama(base: str = _DEFAULT_BASE) -> list[ModelInfo]:
    """List locally-installed Ollama models via ``GET /api/tags`` (parsing real
    parameter sizes). Returns ``[]`` and NEVER raises if Ollama isn't running. Falls
    back to ``GET /v1/models`` (names only) if ``/api/tags`` is unavailable."""
    root = _normalize_base(base)
    try:
        resp = httpx.get(f"{root}/api/tags", timeout=_DETECT_TIMEOUT_S)
        resp.raise_for_status()
        data = resp.json()
    except Exception:  # noqa: BLE001 - any failure means "no local models"
        return _detect_via_v1(root)

    models: list[ModelInfo] = []
    for m in data.get("models", []):
        name = m.get("name") or m.get("model") or ""
        if not name:
            continue
        details = m.get("details") or {}
        models.append(
            ModelInfo(
                name=name,
                params_b=_parse_param_size(details.get("parameter_size")),
                family=details.get("family", "") or "",
                size_bytes=int(m.get("size", 0) or 0),
            )
        )
    return models


def _detect_via_v1(root: str) -> list[ModelInfo]:
    """Fallback: the OpenAI-compatible ``/v1/models`` gives names but no sizes."""
    try:
        resp = httpx.get(f"{root}/v1/models", timeout=_DETECT_TIMEOUT_S)
        resp.raise_for_status()
        data = resp.json()
    except Exception:  # noqa: BLE001
        return []
    return [ModelInfo(name=m["id"]) for m in data.get("data", []) if m.get("id")]


def _score(m: ModelInfo) -> float:
    return m.params_b + (_TOOL_BONUS_B if m.tool_capable else 0.0)


def rank_models(models: list[ModelInfo]) -> list[ModelInfo]:
    """Best first: primarily by real parameter count, with a tool-calling boost so a
    tool-capable model outranks a similar-size non-tool one. Size breaks ties."""
    return sorted(models, key=lambda m: (_score(m), m.size_bytes), reverse=True)


def _system_ram_bytes() -> int:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return 0


def pick_models(ranked: list[ModelInfo]) -> dict:
    """Choose models adaptively from a ranked list. Returns the picks plus a
    ``tool_capable`` flag and chosen sizes for the wizard to show honestly.

    * ``reasoning_model``: the top-ranked TOOL-CAPABLE model (a 32B/70B is picked
      over an 8B automatically); if none are tool-capable, the biggest overall.
    * ``fast_model``: the smallest tool-capable model ≥ ~3B for snappy routing,
      else the reasoning model.
    * ``synthesis_model``: ``None`` (reuse reasoning) unless the top model is huge.
    """
    ranked = list(ranked)
    tool_models = [m for m in ranked if m.tool_capable]
    tool_capable = bool(tool_models)

    if tool_models:
        reasoning = tool_models[0]
        fast_candidates = [m for m in tool_models if m.params_b >= _FAST_MIN_B]
        fast = min(fast_candidates, key=lambda m: m.params_b) if fast_candidates else reasoning
    elif ranked:
        reasoning = ranked[0]
        fast = reasoning
    else:
        return {
            "reasoning_model": None, "fast_model": None, "synthesis_model": None,
            "tool_capable": False, "reasoning_params_b": 0.0, "fast_params_b": 0.0,
            "ram_warning": None,
        }

    # Optional RAM sanity: if the pick is far bigger than RAM, prefer the largest
    # tool-capable model that comfortably fits (Ollama will still load big ones, but
    # they swap and crawl). Best-effort and silent if we can't read RAM/sizes.
    ram = _system_ram_bytes()
    ram_warning: str | None = None
    if ram and reasoning.size_bytes and reasoning.size_bytes > 0.9 * ram:
        fits = [m for m in tool_models if m.size_bytes and m.size_bytes < 0.8 * ram]
        if fits:
            smaller = max(fits, key=lambda m: m.params_b)
            if smaller.name != reasoning.name:
                ram_warning = (
                    f"{reasoning.name} (~{reasoning.size_bytes / 1e9:.0f}GB) likely won't "
                    f"fit in {ram / 1e9:.0f}GB RAM — using {smaller.name} instead."
                )
                reasoning = smaller
                if fast.params_b > reasoning.params_b:
                    fast = reasoning
        else:
            ram_warning = (
                f"{reasoning.name} may exceed your {ram / 1e9:.0f}GB RAM and load slowly."
            )

    synthesis = reasoning.name if reasoning.params_b >= _HUGE_PARAMS_B else None
    return {
        "reasoning_model": reasoning.name,
        "fast_model": fast.name,
        "synthesis_model": synthesis,
        "tool_capable": tool_capable,
        "reasoning_params_b": reasoning.params_b,
        "fast_params_b": fast.params_b,
        "ram_warning": ram_warning,
    }
