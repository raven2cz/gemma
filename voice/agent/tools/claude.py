"""Claude bridge tool - `ask_claude(prompt, system?, max_tokens?)`.

Deleguje na **Claude Code CLI** (`claude -p`) jako sub-agent subprocess, ne
přímo na Anthropic Messages REST API. Vzor: `avatar-engine/avatar_engine/
bridges/claude.py`. Implementace bridge je v `voice/agent/claude_bridge.py`,
tool je tenký wrapper s validací args.

Použití:
  - když lokální Gemma narazí na limit (kontext / kvalita reasoningu / unknown)
  - jako "expert review" druhý názor v rámci řetězu úvah

Bezpečnostní model viz `claude_bridge.py`:
  - Prompt přes stdin (žádný argv leak přes `ps`)
  - `--bare` + `--tools ""` + `--no-session-persistence` + empty cwd
  - Subprocess s scrubbed env, process group + killpg cleanup
  - Per-chunk output cap, stderr drain async task
  - Cancel přes turn_state.cancel_event
"""
from __future__ import annotations

from voice.agent.claude_bridge import ask_claude_oneshot
from voice.agent.config import (
    CLAUDE_CLI_BIN,
    CLAUDE_DEFAULT_MODEL,
    CLAUDE_MAX_PROMPT_BYTES,
    CLAUDE_MAX_SYSTEM_BYTES,
    CLAUDE_MAX_TOKENS_DEFAULT,  # ponecháno pro kompat schema, no-op
    CLAUDE_MAX_TOKENS_LIMIT,    # ponecháno pro kompat schema
    CLAUDE_OUTPUT_CAP_BYTES,
    CLAUDE_TIMEOUT_SEC,
)
from voice.agent.tools.base import ExecuteContext, Tool


async def _ask_claude_exec(args: dict, ctx: ExecuteContext) -> dict:
    """Validace args + delegace na claude_bridge.ask_claude_oneshot."""
    # Re-validace (defense in depth - classifier už filtroval).
    prompt_arg = args.get("prompt")
    if not isinstance(prompt_arg, str):
        return {"ok": False, "error": "prompt must be string"}
    prompt = prompt_arg.strip()
    if not prompt:
        return {"ok": False, "error": "empty prompt"}
    try:
        if len(prompt.encode("utf-8")) > CLAUDE_MAX_PROMPT_BYTES:
            return {"ok": False, "error": "prompt too large"}
    except UnicodeEncodeError:
        return {"ok": False, "error": "prompt not valid utf-8"}

    system_arg = args.get("system")
    if system_arg is not None:
        if not isinstance(system_arg, str):
            return {"ok": False, "error": "system must be string"}
        try:
            if len(system_arg.encode("utf-8")) > CLAUDE_MAX_SYSTEM_BYTES:
                return {"ok": False, "error": "system too large"}
        except UnicodeEncodeError:
            return {"ok": False, "error": "system not valid utf-8"}

    # max_tokens je v Claude CLI no-op (CLI nemá direct mapping); ponecháno
    # v schemě kvůli backward-kompat klientů. Pokud chce user reálný cap,
    # musí na CLI úrovni (max_budget_usd, nebo system prompt instrukce).
    max_tokens_arg = args.get("max_tokens")
    if max_tokens_arg is not None:
        if isinstance(max_tokens_arg, bool):
            return {"ok": False, "error": "max_tokens must be int, not bool"}
        try:
            max_tokens = int(max_tokens_arg)
        except (TypeError, ValueError):
            return {"ok": False, "error": "max_tokens must be integer"}
        if max_tokens < 1 or max_tokens > CLAUDE_MAX_TOKENS_LIMIT:
            return {"ok": False, "error": f"max_tokens out of range 1..{CLAUDE_MAX_TOKENS_LIMIT}"}

    cancel_event = None
    if ctx is not None and getattr(ctx, "cancel_event", None) is not None:
        # ExecuteContext.cancel_event může být threading.Event (z agent loop)
        # nebo asyncio.Event. Bridge očekává asyncio.Event interface (.wait()).
        # Pokud je threading, balíme ho - ale v praxi se používá asyncio.
        ce = ctx.cancel_event
        if hasattr(ce, "wait") and asyncio_compatible(ce):
            cancel_event = ce

    result = await ask_claude_oneshot(
        prompt=prompt,
        system=system_arg,
        model=CLAUDE_DEFAULT_MODEL,
        timeout_sec=CLAUDE_TIMEOUT_SEC,
        output_cap_bytes=CLAUDE_OUTPUT_CAP_BYTES,
        claude_bin=CLAUDE_CLI_BIN,
        cancel_event=cancel_event,
    )
    return result


def asyncio_compatible(event) -> bool:
    """True pokud event vypadá jako asyncio.Event (má awaitable .wait())."""
    import asyncio
    return isinstance(event, asyncio.Event)


ASK_CLAUDE_TOOL = Tool(
    name="ask_claude",
    description=(
        "Delegate a question or task to Claude (Anthropic LLM via Claude Code "
        "CLI subprocess). Use this for problems that exceed local model "
        "capability: complex reasoning, expert second opinion, code review, "
        "or when context is too large for local model. The sub-agent has NO "
        "tools (--tools '') and runs in an empty workdir - pure consult, no "
        "file/shell access. Each call costs money - use sparingly."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": (
                    f"User prompt for Claude. Max {CLAUDE_MAX_PROMPT_BYTES // 1024} KiB UTF-8."
                ),
            },
            "system": {
                "type": "string",
                "description": (
                    "Optional system prompt (role/persona). Max "
                    f"{CLAUDE_MAX_SYSTEM_BYTES // 1024} KiB UTF-8."
                ),
            },
            "max_tokens": {
                "type": "integer",
                "minimum": 1,
                "maximum": CLAUDE_MAX_TOKENS_LIMIT,
                "description": (
                    "Maximum output tokens (default "
                    f"{CLAUDE_MAX_TOKENS_DEFAULT}, max {CLAUDE_MAX_TOKENS_LIMIT}). "
                    "Note: currently no-op with Claude CLI backend (no direct "
                    "mapping); kept for forward-compat with REST adapter."
                ),
            },
        },
        "required": ["prompt"],
        "additionalProperties": False,
    },
    execute=_ask_claude_exec,
)


ALL_TOOLS = (ASK_CLAUDE_TOOL,)
