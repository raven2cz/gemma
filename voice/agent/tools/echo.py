"""Testovací echo tool — bez side-effects. Slouží jako baseline pro
verifikaci tool-calling pipeline."""
from __future__ import annotations

from voice.agent.tools.base import ExecuteContext, Tool


async def _execute(args: dict, ctx: ExecuteContext) -> dict:
    text = str(args.get("text", ""))
    return {"echoed": text, "length": len(text)}


TOOL = Tool(
    name="echo",
    description=(
        "Echo the given text back. Use this only when explicitly testing "
        "the tool pipeline; otherwise just answer directly."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "Text to echo back."},
        },
        "required": ["text"],
        "additionalProperties": False,
    },
    execute=_execute,
)
