"""ask_claude tool - deleguje na Claude Code CLI subprocess přes bridge.

Dva módy:
  consult (default) - pure consult, žádné Claude tools, žádný FS přístup.
                       Classifier ASK medium (cena).
  edit               - Claude má Read/Edit/Write/Bash/Glob/Grep v --add-dir
                       workdir (= full shell delegace v projektu).
                       Classifier ASK destructive (vyžaduje frázi).

Model:
  Args.model > ctx.model_hint > CLAUDE_DEFAULT_MODEL
  Enum: "opus" | "sonnet" | full alias (např. "claude-opus-4-7"). Krátké
  aliasy expand-uje resolver před voláním bridge.
"""
from __future__ import annotations

import asyncio

from voice.agent.claude_bridge import ask_claude_oneshot
from voice.agent.config import (
    CLAUDE_CLI_BIN,
    CLAUDE_DEFAULT_MODEL,
    CLAUDE_MAX_PROMPT_BYTES,
    CLAUDE_MAX_SYSTEM_BYTES,
    CLAUDE_TIMEOUT_SEC,
)
from voice.agent.tools.base import ExecuteContext, Tool


# Krátké aliasy pro user-friendly "Opus" / "Sonnet" / "Haiku" v promptu.
# Expand-uje na full model name před voláním bridge.
_MODEL_ALIAS = {
    "opus": "claude-opus-4-7",
    "sonnet": "claude-sonnet-4-6",
    "haiku": "claude-haiku-4-5",
}

# Allowlist pro `model` arg - shared mezi tool execute a permission classifier
# (Codex iter-9: schema enum není security boundary, runtime check nutný).
# Akceptujeme: krátké aliasy + plné claude-* jména.
ALLOWED_MODEL_ALIASES = frozenset(_MODEL_ALIAS.keys())
ALLOWED_MODEL_FULLNAMES = frozenset(_MODEL_ALIAS.values())


def is_allowed_model(value: str) -> bool:
    """True pokud value je validní krátký alias nebo plné claude-* jméno."""
    if not isinstance(value, str):
        return False
    v = value.lower().strip()
    if not v:
        return False
    return v in ALLOWED_MODEL_ALIASES or v in ALLOWED_MODEL_FULLNAMES


# Allowed mode values.
_ALLOWED_MODES = frozenset({"consult", "edit"})


def _resolve_model(args_model: str | None, ctx_hint: str | None) -> str:
    """Precedence: args > ctx_hint > default. Aliasy expand-uje."""
    for cand in (args_model, ctx_hint):
        if not cand:
            continue
        c = cand.lower().strip()
        if c in _MODEL_ALIAS:
            return _MODEL_ALIAS[c]
        # Plain full name passes through (defense: pokud uživatel/router posune
        # nesmysl, bridge stejně failne v CLI a my zachytíme stderr).
        return cand.strip()
    return CLAUDE_DEFAULT_MODEL


async def _ask_claude_exec(args: dict, ctx: ExecuteContext) -> dict:
    """Validace args + delegace na claude_bridge.ask_claude_oneshot."""
    # prompt
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

    # system (optional)
    system_arg = args.get("system")
    if system_arg is not None:
        if not isinstance(system_arg, str):
            return {"ok": False, "error": "system must be string"}
        try:
            if len(system_arg.encode("utf-8")) > CLAUDE_MAX_SYSTEM_BYTES:
                return {"ok": False, "error": "system too large"}
        except UnicodeEncodeError:
            return {"ok": False, "error": "system not valid utf-8"}

    # mode validation. Codex audit HIGH: nemůžeme spoléhat na JSON schema -
    # tool execute musí runtime check pro non-string typy (`mode=true` by jinak
    # spadl na .lower() AttributeError).
    mode_arg = args.get("mode")
    if mode_arg is None:
        mode = "consult"  # missing key = default
    elif not isinstance(mode_arg, str):
        return {"ok": False, "error": f"mode must be string, got {type(mode_arg).__name__}"}
    else:
        # Blank string nebo whitespace = error (NE default consult). Classifier
        # i execute musí mít stejnou normalizaci (Codex iter-8 symmetry fix).
        mode = mode_arg.lower().strip()
        if not mode:
            return {"ok": False, "error": "mode must not be empty string"}
    if mode not in _ALLOWED_MODES:
        return {"ok": False, "error": f"unknown mode {mode!r}, allowed: {sorted(_ALLOWED_MODES)}"}

    # model arg validation (volitelný - None = default fallback)
    model_arg = args.get("model")
    if model_arg is not None:
        if not isinstance(model_arg, str):
            return {"ok": False, "error": f"model must be string, got {type(model_arg).__name__}"}
        # Codex iter-9: runtime allowlist check (schema enum není security boundary).
        if not is_allowed_model(model_arg):
            return {"ok": False,
                    "error": f"model {model_arg!r} not in allowlist (opus/sonnet/haiku/full name)"}

    # model resolution
    model = _resolve_model(model_arg, ctx.model_hint if ctx else None)

    # workdir (jen pro edit)
    workdir = None
    if mode == "edit":
        if ctx is None or not getattr(ctx, "workdir", None):
            return {"ok": False, "error": "mode=edit requires workdir from execute context"}
        workdir = ctx.workdir

    # cancel_event (jen pokud je asyncio.Event compatible)
    cancel_event = None
    if ctx is not None and getattr(ctx, "cancel_event", None) is not None:
        ce = ctx.cancel_event
        if isinstance(ce, asyncio.Event):
            cancel_event = ce

    # progress_callback wrapper: tool dostane payload, předá do ctx.progress_emitter
    # který má per-tool kontext (tool_call_id) a posílá jako tool_progress event.
    progress_cb = None
    if ctx is not None and getattr(ctx, "progress_emitter", None) is not None:
        emitter = ctx.progress_emitter

        async def progress_cb(payload: dict) -> None:
            await emitter(payload)

    result = await ask_claude_oneshot(
        prompt=prompt, system=system_arg,
        model=model, mode=mode, workdir=workdir,
        timeout_sec=CLAUDE_TIMEOUT_SEC,
        claude_bin=CLAUDE_CLI_BIN,
        cancel_event=cancel_event,
        progress_callback=progress_cb,
    )
    return result


ASK_CLAUDE_TOOL = Tool(
    name="ask_claude",
    description=(
        "Delegate a question or task to Claude (Anthropic LLM via Claude Code "
        "CLI subprocess). Two modes:\n"
        "- mode='consult' (default): pure expert review, no tools, no file "
        "  access. Use for code review, complex reasoning, second opinion.\n"
        "- mode='edit': Claude has Read/Edit/Write/Bash/Glob/Grep in the "
        "  workdir. Use when user explicitly asks Claude to modify code, "
        "  create files, run tests in the project. Requires user approval "
        "  phrase ('ano povoluju') - full shell delegation.\n"
        "Optional model: 'opus' (claude-opus-4-7, deeper reasoning) or "
        "'sonnet' (claude-sonnet-4-6, faster). Defaults to opus."
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
            "mode": {
                "type": "string",
                "enum": ["consult", "edit"],
                "description": (
                    "consult (default) = no tools/FS access. "
                    "edit = full shell delegation in workdir, requires approval phrase."
                ),
            },
            "model": {
                "type": "string",
                "enum": ["opus", "sonnet", "haiku"],
                "description": (
                    "Optional model selection. Default opus. Use sonnet for "
                    "faster cheaper consultation."
                ),
            },
        },
        "required": ["prompt"],
        "additionalProperties": False,
    },
    execute=_ask_claude_exec,
)


ALL_TOOLS = (ASK_CLAUDE_TOOL,)
