"""Agent loop = tool-calling smyčka kolem Ollama /api/chat.

Použití::

    loop = AgentLoop(
        model="ollama/gemma4-31b-gguf",
        messages=[...],
        registry=default_registry("agent"),
        turn_state=turn_state,
        workdir=WORKDIR,
    )
    loop.set_approval_resolver(my_resolver)  # async (approval_id) -> bool
    async for ev in loop.run():
        ...

Event typy (dict)::

    {"type": "text", "delta": "..."}                          # streamed token
    {"type": "tool_call", "id": "tc_X", "name": "...", "args": {...}}
    {"type": "approval_required", "approval_id": "ap_X", "tool_call_id": "tc_X",
     "tool": "...", "summary": "...", "risk": "low|medium|high",
     "requires_explicit": False, "args": {...}}
    {"type": "tool_result", "id": "tc_X", "ok": True|False, "content": "..."}
    {"type": "agent_error", "msg": "..."}
    {"type": "agent_canceled"}
    {"type": "agent_done"}

`agent_done` znamená že LLM dokončil bez dalších tool calls (final assistant text
je v deltách těsně před).
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import AsyncIterator, Awaitable, Callable

import httpx

from voice.agent import config
from voice.agent.messages import (
    assistant_message,
    normalize_tool_calls,
    parse_tool_args,
    tool_message,
)
from voice.agent.permissions import Decision, decide
from voice.agent.tools.base import ExecuteContext, ToolRegistry

log = logging.getLogger("agent-loop")

OLLAMA = "http://localhost:11434"


ApprovalResolver = Callable[[str, dict], Awaitable[bool]]
# approval_resolver(approval_id, event_payload) -> True (approve) / False (deny)


class AgentLoop:
    def __init__(
        self,
        *,
        model: str,
        messages: list[dict],
        registry: ToolRegistry,
        turn_state: dict,
        workdir: Path,
    ) -> None:
        self.model = model
        self.messages: list[dict] = list(messages)
        self.registry = registry
        self.turn_state = turn_state
        self.workdir = workdir
        self.deadline = time.monotonic() + config.MAX_WALL_TIME_SEC
        self._approval_resolver: ApprovalResolver | None = None
        self._rounds = 0
        self._tool_calls_used = 0

    def set_approval_resolver(self, resolver: ApprovalResolver) -> None:
        self._approval_resolver = resolver

    def _canceled(self) -> bool:
        return bool(self.turn_state.get("canceled"))

    def _over_deadline(self) -> bool:
        return time.monotonic() > self.deadline

    def _remaining_budget(self) -> float:
        """Sekundy zbývající do wall-time deadlinu. <=0 → expired."""
        return max(0.0, self.deadline - time.monotonic())

    async def _stream_model_turn(self) -> tuple[str, list[dict], str | None]:
        """Provede jednu iteraci LLM streamování. Vrátí
        (assistant_text, raw_tool_calls, error_msg). Yieldy textových delt
        jsou poslány přes self._text_buffer (callback handler je nahrazen
        v `run()`).

        Read-timeout je odvozený od zbývajícího wall-time budgetu — bez něj
        by hangující Ollama (TCP idle) drželo task navždy a wall-time check
        v `driver()` by se nedostal ke slovu.
        """
        payload = {
            "model": self.model,
            "messages": self.messages,
            "stream": True,
            "tools": self.registry.to_openai_schema(),
            "think": False,
        }
        text = ""
        tool_calls: list[dict] = []
        # Cap stream-level timeoutu rozumnou hranicí, ať i s plným 10min budgetem
        # nezpůsobíme příliš dlouhý hang při idle TCP.
        remaining = self._remaining_budget()
        if remaining <= 0:
            return text, tool_calls, f"wall-time {config.MAX_WALL_TIME_SEC}s exceeded"
        timeout = httpx.Timeout(connect=10.0, read=min(remaining, 120.0), write=10.0, pool=10.0)
        try:
            async with httpx.AsyncClient(timeout=timeout) as c:
                async with c.stream("POST", f"{OLLAMA}/api/chat", json=payload) as r:
                    if r.status_code != 200:
                        body = (await r.aread()).decode(errors="ignore")[:500]
                        return text, tool_calls, f"ollama {r.status_code}: {body}"
                    async for line in r.aiter_lines():
                        if self._canceled():
                            return text, tool_calls, None
                        if self._over_deadline():
                            return text, tool_calls, (
                                f"wall-time {config.MAX_WALL_TIME_SEC}s exceeded mid-stream"
                            )
                        if not line.strip():
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        msg = obj.get("message") or {}
                        tok = msg.get("content", "")
                        if tok:
                            text += tok
                            await self._emit_text(tok)
                        tc = msg.get("tool_calls") or []
                        if tc:
                            tool_calls.extend(tc)
                        if obj.get("done"):
                            break
        except asyncio.CancelledError:
            raise
        except httpx.TimeoutException as e:
            return text, tool_calls, f"ollama timeout: {e}"
        except Exception as e:
            log.exception("agent loop: ollama stream failed")
            return text, tool_calls, f"{type(e).__name__}: {e}"
        return text, tool_calls, None

    async def _emit_text(self, delta: str) -> None:
        # Bridge to outer async generator via internal queue.
        await self._out_queue.put({"type": "text", "delta": delta})

    async def run(self) -> AsyncIterator[dict]:
        """Hlavní smyčka. Vrací async generator eventů."""
        self._out_queue: asyncio.Queue = asyncio.Queue()

        async def driver() -> None:
            try:
                while True:
                    if self._canceled():
                        await self._out_queue.put({"type": "agent_canceled"})
                        return
                    if self._over_deadline():
                        await self._out_queue.put(
                            {"type": "agent_error",
                             "msg": f"agent wall-time {config.MAX_WALL_TIME_SEC}s exceeded"}
                        )
                        return
                    if self._rounds >= config.MAX_MODEL_ROUNDS_PER_TURN:
                        await self._out_queue.put(
                            {"type": "agent_error",
                             "msg": f"agent round limit {config.MAX_MODEL_ROUNDS_PER_TURN} exceeded"}
                        )
                        return
                    self._rounds += 1

                    text, raw_tcs, err = await self._stream_model_turn()
                    if err is not None:
                        await self._out_queue.put({"type": "agent_error", "msg": err})
                        return
                    if self._canceled():
                        await self._out_queue.put({"type": "agent_canceled"})
                        return

                    if not raw_tcs:
                        # No tools → final answer reached.
                        if text:
                            self.messages.append(assistant_message(text))
                        await self._out_queue.put({"type": "agent_done"})
                        return

                    tcs = normalize_tool_calls(raw_tcs)
                    if not tcs:
                        # All chunks malformed/empty — treat as no-tool round.
                        if text:
                            self.messages.append(assistant_message(text))
                        await self._out_queue.put({"type": "agent_done"})
                        return
                    # Cap tool calls per turn (cumulative across rounds).
                    remaining = config.MAX_TOOL_CALLS_PER_TURN - self._tool_calls_used
                    if remaining <= 0:
                        await self._out_queue.put(
                            {"type": "agent_error",
                             "msg": f"agent tool-call limit {config.MAX_TOOL_CALLS_PER_TURN} exceeded"}
                        )
                        return
                    # Truncate na cap, ALE neztratíme dropped tool calls z LLM kontextu —
                    # historii necháme úplnou a pro dropped vygenerujeme syntetické
                    # tool_result, jinak by model viděl assistant.tool_calls bez párového
                    # tool messagu (malformed OpenAI history → halucinace / 400).
                    executed = tcs[:remaining]
                    dropped = tcs[remaining:]
                    self._tool_calls_used += len(executed)
                    self.messages.append(assistant_message(text, tool_calls=tcs))

                    for tc in executed:
                        if self._canceled():
                            await self._out_queue.put({"type": "agent_canceled"})
                            return
                        await self._execute_one(tc)

                    if dropped:
                        for tc in dropped:
                            tcid = tc["id"]
                            tname = tc["function"]["name"]
                            payload = {"ok": False,
                                       "error": f"tool-call limit {config.MAX_TOOL_CALLS_PER_TURN} reached — skipped"}
                            content = json.dumps(payload, ensure_ascii=False)
                            await self._out_queue.put(
                                {"type": "tool_result", "id": tcid, "ok": False, "content": content}
                            )
                            self.messages.append(tool_message(tcid, tname, content))
                        await self._out_queue.put(
                            {"type": "agent_error",
                             "msg": f"agent tool-call limit {config.MAX_TOOL_CALLS_PER_TURN} exceeded"}
                        )
                        return
            except asyncio.CancelledError:
                await self._out_queue.put({"type": "agent_canceled"})
                raise
            except Exception as e:
                log.exception("agent loop driver crashed")
                await self._out_queue.put(
                    {"type": "agent_error", "msg": f"{type(e).__name__}: {e}"}
                )
            finally:
                await self._out_queue.put(_SENTINEL)

        task = asyncio.create_task(driver())
        try:
            while True:
                ev = await self._out_queue.get()
                if ev is _SENTINEL:
                    break
                yield ev
                # Pokud driver vrátil terminal event, počkáme na ukončení tasku.
                if ev.get("type") in ("agent_done", "agent_error", "agent_canceled"):
                    break
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

    async def _execute_one(self, tc: dict) -> None:
        tcid: str = tc["id"]
        name: str = tc["function"]["name"]
        args: dict = parse_tool_args(tc)
        await self._out_queue.put(
            {"type": "tool_call", "id": tcid, "name": name, "args": args}
        )

        perm = decide(name, args, self.workdir)
        approved: bool
        deny_reason: str | None = None

        if perm.decision == Decision.AUTO:
            approved = True
        elif perm.decision == Decision.DENY:
            approved = False
            deny_reason = f"policy: {perm.reason}"
        else:  # ASK
            if self._approval_resolver is None:
                approved = False
                deny_reason = "no approval resolver configured"
            else:
                from voice.agent.messages import new_approval_id
                approval_id = new_approval_id()
                event = {
                    "type": "approval_required",
                    "approval_id": approval_id,
                    "tool_call_id": tcid,
                    "tool": name,
                    "summary": perm.summary,
                    "risk": perm.risk,
                    "requires_explicit": perm.requires_explicit,
                    "args": args,
                }
                await self._out_queue.put(event)
                # Approval wait je bounded wall-time deadlinem — user nemá UI timeout,
                # ale agent jako celek nesmí překročit MAX_WALL_TIME_SEC. Bez bounded
                # waitu by zapomenutý approval blokoval task navždy.
                remaining = self._remaining_budget()
                if remaining <= 0:
                    approved = False
                    deny_reason = f"wall-time {config.MAX_WALL_TIME_SEC}s exceeded before approval"
                else:
                    try:
                        approved = await asyncio.wait_for(
                            self._approval_resolver(approval_id, event),
                            timeout=remaining,
                        )
                    except asyncio.TimeoutError:
                        approved = False
                        deny_reason = f"wall-time {config.MAX_WALL_TIME_SEC}s exceeded waiting for approval"
                    except asyncio.CancelledError:
                        raise
                    except Exception as e:
                        log.exception("approval resolver raised")
                        approved = False
                        deny_reason = f"approval error: {type(e).__name__}: {e}"
                    else:
                        if not approved:
                            deny_reason = "user denied"

        if not approved:
            payload = {"ok": False, "error": deny_reason or "denied"}
            content = json.dumps(payload, ensure_ascii=False)
            await self._out_queue.put(
                {"type": "tool_result", "id": tcid, "ok": False, "content": content}
            )
            self.messages.append(tool_message(tcid, name, content))
            return

        tool = self.registry.get(name)
        if tool is None:
            payload = {"ok": False, "error": f"unknown tool {name!r}"}
            content = json.dumps(payload, ensure_ascii=False)
            await self._out_queue.put(
                {"type": "tool_result", "id": tcid, "ok": False, "content": content}
            )
            self.messages.append(tool_message(tcid, name, content))
            return

        ctx = ExecuteContext(
            turn_id=self.turn_state.get("id", ""),
            cancel_event=self.turn_state.get("cancel_event"),
            workdir=self.workdir,
            resolved_path=perm.resolved_path,
        )
        remaining = self._remaining_budget()
        if remaining <= 0:
            payload = {"ok": False, "error": f"wall-time {config.MAX_WALL_TIME_SEC}s exceeded before tool"}
            content = json.dumps(payload, ensure_ascii=False)
            await self._out_queue.put(
                {"type": "tool_result", "id": tcid, "ok": False, "content": content}
            )
            self.messages.append(tool_message(tcid, name, content))
            return
        try:
            out = await asyncio.wait_for(tool.execute(args, ctx), timeout=remaining)
        except asyncio.TimeoutError:
            payload = {"ok": False, "error": f"wall-time {config.MAX_WALL_TIME_SEC}s exceeded during tool"}
            content = json.dumps(payload, ensure_ascii=False)
            await self._out_queue.put(
                {"type": "tool_result", "id": tcid, "ok": False, "content": content}
            )
            self.messages.append(tool_message(tcid, name, content))
            return
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.exception("tool %s failed", name)
            payload = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            content = json.dumps(payload, ensure_ascii=False)
            await self._out_queue.put(
                {"type": "tool_result", "id": tcid, "ok": False, "content": content}
            )
            self.messages.append(tool_message(tcid, name, content))
            return

        if isinstance(out, str):
            content = out
        else:
            try:
                content = json.dumps(out, ensure_ascii=False)
            except (TypeError, ValueError) as e:
                # Non-serializable tool output — degrade gracefully místo crashe driveru.
                log.warning("tool %s returned non-serializable output: %s", name, e)
                payload = {"ok": False, "error": f"non-serializable tool output: {type(e).__name__}: {e}"}
                content = json.dumps(payload, ensure_ascii=False)
                await self._out_queue.put(
                    {"type": "tool_result", "id": tcid, "ok": False, "content": content}
                )
                self.messages.append(tool_message(tcid, name, content))
                return
        encoded = content.encode("utf-8")
        cap = config.TOOL_OUTPUT_CAP_BYTES
        if len(encoded) > cap:
            # Byte-level slice + decode errors='ignore' uřízne neúplný UTF-8 prefix.
            content = encoded[:cap].decode("utf-8", errors="ignore") + "\n…[truncated]"

        await self._out_queue.put(
            {"type": "tool_result", "id": tcid, "ok": True, "content": content}
        )
        self.messages.append(tool_message(tcid, name, content))


_SENTINEL = object()
