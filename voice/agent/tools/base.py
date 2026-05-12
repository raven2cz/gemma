"""Tool dataclass + ToolRegistry. Agnostické vůči konkrétním toolům."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable


@dataclass
class ExecuteContext:
    """Předáno do `Tool.execute` při každém volání.

    `resolved_path` (volitelná) = canonical path, kterou classifier resolvoval
    a na které spočinulo permission rozhodnutí. Tooly, které manipulují s FS,
    by měly preferovat tuto cestu před vlastním re-resolve, aby se eliminoval
    TOCTOU gap mezi klasifikací a exekucí.
    """
    turn_id: str
    cancel_event: asyncio.Event | None
    workdir: Path
    resolved_path: Path | None = None


ExecuteFn = Callable[[dict, ExecuteContext], Awaitable[Any]]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str          # short, used in tool_choice listing for LLM
    parameters_schema: dict   # JSON Schema (OpenAI function schema)
    execute: ExecuteFn        # async callable, returns JSON-serializable or str


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def to_openai_schema(self) -> list[dict]:
        """Serializace pro Ollama/OpenAI `tools` parametr."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters_schema,
                },
            }
            for t in self._tools.values()
        ]
