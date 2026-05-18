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
    {"type": "audio_filler", "tool_call_id": "tc_X", "tool": "..."}  # Phase 9.3
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
from voice.agent.audit import AuditLog
from voice.agent.messages import (
    assistant_message,
    is_malformed_args,
    normalize_tool_calls,
    parse_tool_args,
    tool_message,
)
from voice.agent.permissions import Decision, PermissionResult, decide
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
        audit_log: AuditLog | None = None,
    ) -> None:
        self.model = model
        self.messages: list[dict] = list(messages)
        self.registry = registry
        self.turn_state = turn_state
        self.workdir = workdir
        self.audit_log = audit_log
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
        log.debug(
            "stream_turn round=%d model=%s msgs=%d tools=%d tool_names=%s",
            self._rounds, self.model, len(self.messages),
            len(payload["tools"]),
            [t["function"]["name"] for t in payload["tools"]],
        )
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
                        log.error(
                            "ollama /api/chat non-200: status=%d body=%s payload_keys=%s msgs=%d model=%s",
                            r.status_code, body, sorted(payload.keys()),
                            len(payload.get("messages", [])), self.model,
                        )
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
                            log.debug(
                                "raw tool_calls chunk: %s",
                                json.dumps(tc, ensure_ascii=False)[:1000],
                            )
                            # Ollama native /api/chat emituje tool_calls až v
                            # done-chunku jako kompletní pole (žádné partial
                            # arguments deltas). Pokud by se to v budoucí
                            # verzi Ollama změnilo na OpenAI-style streaming
                            # s delta accumulationem, normalize_tool_calls
                            # zachová last-write-wins per id — což je bezpečné
                            # pro duplikáty, ale ztratí prefix u true deltas.
                            # Canonicalize + sentinel + short-circuit v
                            # `_execute_one` zajistí, že malformed partial
                            # NIKDY nezpůsobí Ollama 400 ani spuštění toolu
                            # s incomplete args.
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
        log.debug(
            "stream_turn done: text_len=%d raw_tool_calls=%d",
            len(text), len(tool_calls),
        )
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
                        log.error("agent loop terminating with error: %s", err)
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
                    log.debug(
                        "normalize_tool_calls: raw=%d → normalized=%d names=%s",
                        len(raw_tcs), len(tcs),
                        [t["function"]["name"] for t in tcs],
                    )
                    if not tcs:
                        # All chunks malformed/empty — treat as no-tool round.
                        if text:
                            self.messages.append(assistant_message(text))
                        await self._out_queue.put({"type": "agent_done"})
                        return
                    # Cap tool calls per turn (cumulative across rounds).
                    # Phase 8 iter-4 fix (Codex): pokud limit byl vyčerpán v
                    # předchozím kole (remaining <= 0), všechny tcs v tomto
                    # kole jsou dropped. Bez tohoto by ranní calls minuly
                    # auditní stopu — LLM by mohl schovat malicious args.
                    remaining = max(0, config.MAX_TOOL_CALLS_PER_TURN - self._tool_calls_used)
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
                            err = f"tool-call limit {config.MAX_TOOL_CALLS_PER_TURN} reached — skipped"
                            payload = {"ok": False, "error": err}
                            content = json.dumps(payload, ensure_ascii=False)
                            await self._out_queue.put(
                                {"type": "tool_result", "id": tcid, "ok": False, "content": content}
                            )
                            self.messages.append(tool_message(tcid, tname, content))
                            # Phase 8 iter-3 fix (Codex): audit dropped calls
                            # taky. Bez tohoto by LLM mohl schovat malicious
                            # tool za rate-limit a nezůstala by forensic stopa.
                            if self.audit_log is not None and self.audit_log.enabled:
                                try:
                                    dropped_args = parse_tool_args(tc)
                                except Exception:
                                    dropped_args = {}
                                skipped_perm = {
                                    "decision": "skipped_limit",
                                    "risk": "n/a",
                                    "reason": err,
                                    "requires_explicit": False,
                                    "summary": f"{tname}: skipped (rate limit)",
                                }
                                audit_task = asyncio.create_task(self.audit_log.record(
                                    turn_id=str(self.turn_state.get("id", "")),
                                    tool_call_id=tcid,
                                    tool_name=tname,
                                    args=dropped_args,
                                    permission=skipped_perm,
                                    approval=None,
                                    ok=False,
                                    error=err,
                                    result_bytes=len(content.encode("utf-8")),
                                    duration_ms=0,
                                ))
                                audit_task.add_done_callback(_swallow_task_exc)
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
        # Agent-mode policy override (MUSÍ být před tool_call eventem aby UI
        # i klasifikátor viděly skutečný mode). ask_claude vždy mode="edit" -
        # consult v agent módu nedává smysl (Claude by neviděl soubory).
        if name == "ask_claude" and self.workdir and isinstance(args, dict):
            requested_mode = args.get("mode")
            if requested_mode != "edit":
                log.info(
                    "ask_claude tcid=%s: agent-mode policy override mode=%r → 'edit'",
                    tcid, requested_mode,
                )
                args["mode"] = "edit"
        log.debug(
            "execute_one tcid=%s name=%s args=%s",
            tcid, name, json.dumps(args, ensure_ascii=False)[:500],
        )
        await self._out_queue.put(
            {"type": "tool_call", "id": tcid, "name": name, "args": args}
        )

        t0 = time.monotonic()

        # Bezpečnostní short-circuit: model vygeneroval malformed JSON v
        # arguments (typicky uříznutý EOS/max_tokens uprostřed `{...}`).
        # `parse_tool_args` vrací sentinel s `_parse_error` místo prázdného
        # `{}` (security fix: bez tohoto by classifier viděl prázdné args
        # a model mohl schovat malicious destruktivní obsah za malformed JSON).
        # Tool NESPOUŠTÍME, ani neposíláme do permission classifieru —
        # rovnou emit paired tool_result s ok=False a audit zápisem.
        if is_malformed_args(args):
            reason = args.get("_parse_error", "invalid")
            err = f"malformed tool_call arguments: {reason}"
            log.warning(
                "tool %s rejected: malformed args (%s), raw_hash=%s",
                name, reason, args.get("raw_hash", "")[:16],
            )
            malformed_perm = PermissionResult(
                decision=Decision.DENY,
                risk="high",
                reason=err,
                summary=f"{name}: malformed args (no execution)",
                requires_explicit=False,
            )
            await self._finalize_tool_call(
                tcid=tcid, name=name, args=args, perm=malformed_perm,
                approval=None, ok=False,
                content=json.dumps(
                    {"ok": False, "error": err, "_parse_error": reason,
                     "raw_hash": args.get("raw_hash", "")},
                    ensure_ascii=False,
                ),
                error=err, t0=t0,
            )
            return

        perm = decide(name, args, self.workdir)
        log.debug(
            "permission tcid=%s name=%s decision=%s risk=%s reason=%s",
            tcid, name,
            perm.decision.value if hasattr(perm.decision, "value") else perm.decision,
            perm.risk, perm.reason,
        )
        approved: bool
        approval_outcome: str | None  # "approved" | "denied" | None (= nebylo soliciováno)
        deny_reason: str | None = None

        if perm.decision == Decision.AUTO:
            approved = True
            approval_outcome = None
        elif perm.decision == Decision.DENY:
            approved = False
            approval_outcome = None
            deny_reason = f"policy: {perm.reason}"
        else:  # ASK
            if self._approval_resolver is None:
                approved = False
                approval_outcome = "denied"
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
                    approval_outcome = "denied"
                    deny_reason = f"wall-time {config.MAX_WALL_TIME_SEC}s exceeded before approval"
                else:
                    try:
                        approved = await asyncio.wait_for(
                            self._approval_resolver(approval_id, event),
                            timeout=remaining,
                        )
                    except asyncio.TimeoutError:
                        approved = False
                        approval_outcome = "denied"
                        deny_reason = f"wall-time {config.MAX_WALL_TIME_SEC}s exceeded waiting for approval"
                    except asyncio.CancelledError:
                        # Best-effort audit před re-raise — forensic gap fix
                        # (Phase 8 iter-1 review). Cancellation se může stát
                        # mezi approval a tool exec, audit nesmí zmizet.
                        self._audit_cancellation_fire_and_forget(
                            tcid, name, args, perm, "denied", t0,
                            stage="cancelled during approval wait",
                        )
                        raise
                    except Exception as e:
                        log.exception("approval resolver raised")
                        approved = False
                        approval_outcome = "denied"
                        deny_reason = f"approval error: {type(e).__name__}: {e}"
                    else:
                        approval_outcome = "approved" if approved else "denied"
                        if not approved:
                            deny_reason = "user denied"

        if not approved:
            await self._finalize_tool_call(
                tcid=tcid, name=name, args=args, perm=perm,
                approval=approval_outcome, ok=False,
                content=json.dumps({"ok": False, "error": deny_reason or "denied"}, ensure_ascii=False),
                error=deny_reason or "denied", t0=t0,
            )
            return

        tool = self.registry.get(name)
        if tool is None:
            err = f"unknown tool {name!r}"
            await self._finalize_tool_call(
                tcid=tcid, name=name, args=args, perm=perm,
                approval=approval_outcome, ok=False,
                content=json.dumps({"ok": False, "error": err}, ensure_ascii=False),
                error=err, t0=t0,
            )
            return

        # Progress emitter per tool call - tool může emit dict payload,
        # loop ho wrapne jako `tool_progress` event s tool_call_id + tool name.
        async def _progress_emit(payload: dict) -> None:
            await self._out_queue.put({
                "type": "tool_progress",
                "tool_call_id": tcid,
                "tool": name,
                "payload": payload,
            })

        # Model hint z router_decision (server může uložit do turn_state).
        # Tool ho použije pokud user/Gemma nezadal `model` arg explicitně.
        ctx = ExecuteContext(
            turn_id=self.turn_state.get("id", ""),
            cancel_event=self.turn_state.get("cancel_event"),
            workdir=self.workdir,
            resolved_path=perm.resolved_path,
            progress_emitter=_progress_emit,
            model_hint=self.turn_state.get("model_hint"),
        )
        remaining = self._remaining_budget()
        if remaining <= 0:
            err = f"wall-time {config.MAX_WALL_TIME_SEC}s exceeded before tool"
            await self._finalize_tool_call(
                tcid=tcid, name=name, args=args, perm=perm,
                approval=approval_outcome, ok=False,
                content=json.dumps({"ok": False, "error": err}, ensure_ascii=False),
                error=err, t0=t0,
            )
            return
        # Phase 9.3: Audio filler watchdog — pokud tool běží > AUDIO_FILLER_DELAY_SEC,
        # emit `audio_filler` event aby frontend řekl „moment, hledám…". Spuštěno
        # paraleln s tool.execute, kanselováno IHNED po návratu z execute (před
        # finalize await chainem). Gemini iter-1 finding: cancel ve vnějším
        # finally umožní race — finalize_tool_call (queue.put + audit shield)
        # await-uje I/O, během kterého stihne filler.sleep vypršet a vyemitovat
        # event AŽ PO tool_result.
        filler_task = asyncio.create_task(
            self._emit_audio_filler_after_delay(tcid, name)
        )
        try:
            out = await asyncio.wait_for(tool.execute(args, ctx), timeout=remaining)
        except asyncio.TimeoutError:
            filler_task.cancel()
            err = f"wall-time {config.MAX_WALL_TIME_SEC}s exceeded during tool"
            await self._finalize_tool_call(
                tcid=tcid, name=name, args=args, perm=perm,
                approval=approval_outcome, ok=False,
                content=json.dumps({"ok": False, "error": err}, ensure_ascii=False),
                error=err, t0=t0,
            )
            return
        except asyncio.CancelledError:
            filler_task.cancel()
            # Tool byl approved (jinak by sem nedošlo) a possibly partially
            # executed. Best-effort audit s ok=False, error="cancelled".
            self._audit_cancellation_fire_and_forget(
                tcid, name, args, perm, approval_outcome, t0,
                stage="cancelled during tool execution",
            )
            raise
        except Exception as e:
            filler_task.cancel()
            log.exception("tool %s failed", name)
            err = f"{type(e).__name__}: {e}"
            await self._finalize_tool_call(
                tcid=tcid, name=name, args=args, perm=perm,
                approval=approval_outcome, ok=False,
                content=json.dumps({"ok": False, "error": err}, ensure_ascii=False),
                error=err, t0=t0,
            )
            return
        # Happy path: nejdřív zruš watchdog (žádný další await mezi nimi).
        filler_task.cancel()
        log.debug("tool %s executed OK, output_type=%s", name, type(out).__name__)

        if isinstance(out, str):
            content = out
        else:
            try:
                content = json.dumps(out, ensure_ascii=False)
            except (TypeError, ValueError) as e:
                # Non-serializable tool output — degrade gracefully místo crashe driveru.
                log.warning("tool %s returned non-serializable output: %s", name, e)
                err = f"non-serializable tool output: {type(e).__name__}: {e}"
                await self._finalize_tool_call(
                    tcid=tcid, name=name, args=args, perm=perm,
                    approval=approval_outcome, ok=False,
                    content=json.dumps({"ok": False, "error": err}, ensure_ascii=False),
                    error=err, t0=t0,
                )
                return
        encoded = content.encode("utf-8")
        cap = config.TOOL_OUTPUT_CAP_BYTES
        if len(encoded) > cap:
            # Byte-level slice + decode errors='ignore' uřízne neúplný UTF-8 prefix.
            content = encoded[:cap].decode("utf-8", errors="ignore") + "\n…[truncated]"

        await self._finalize_tool_call(
            tcid=tcid, name=name, args=args, perm=perm,
            approval=approval_outcome, ok=True,
            content=content, error=None, t0=t0,
        )

    async def _emit_audio_filler_after_delay(self, tcid: str, name: str) -> None:
        """Phase 9.3: Po `AUDIO_FILLER_DELAY_SEC` emit jeden `audio_filler` event.
        Pokud delay=0 nebo task je cancelled předtím, neemituje nic.
        Helper neraise-uje (nesmí zabít hlavní loop)."""
        delay = config.AUDIO_FILLER_DELAY_SEC
        if delay <= 0:
            return
        try:
            await asyncio.sleep(delay)
            await self._out_queue.put({
                "type": "audio_filler",
                "tool_call_id": tcid,
                "tool": name,
            })
        except asyncio.CancelledError:
            # Tool skončil rychleji než delay — žádný filler ani nemá vyjít.
            raise
        except Exception:
            log.exception("audio_filler emit failed (tcid=%s, tool=%s)", tcid, name)

    def _audit_cancellation_fire_and_forget(
        self,
        tcid: str,
        name: str,
        args: dict,
        perm: PermissionResult,
        approval_outcome: str | None,
        t0: float,
        *,
        stage: str,
    ) -> None:
        """Spawn detached audit task při CancelledError. Await uvnitř cancelled
        contextu by znovu vyhodil CancelledError; create_task ho protne. Task
        doběhne v event loopu paralelně s tear-downem agenta. Helper neraise-uje."""
        if self.audit_log is None or not self.audit_log.enabled:
            return
        try:
            coro = self.audit_log.record(
                turn_id=str(self.turn_state.get("id", "")),
                tool_call_id=tcid,
                tool_name=name,
                args=args,
                permission={
                    "decision": perm.decision.value if hasattr(perm.decision, "value") else str(perm.decision),
                    "risk": perm.risk,
                    "reason": perm.reason,
                    "requires_explicit": perm.requires_explicit,
                    "summary": perm.summary,
                },
                approval=approval_outcome,
                ok=False,
                error=stage,
                result_bytes=0,
                duration_ms=int((time.monotonic() - t0) * 1000),
            )
            task = asyncio.create_task(coro)
            # Connect default exception handler — bez tohoto by event loop
            # vyhodil "Task exception was never retrieved" warning, kdyby
            # uvnitř .record() vznikla unexpected exception.
            task.add_done_callback(_swallow_task_exc)
        except Exception:
            log.exception("audit: failed to spawn cancel record task")

    async def _finalize_tool_call(
        self,
        *,
        tcid: str,
        name: str,
        args: dict,
        perm: PermissionResult,
        approval: str | None,
        ok: bool,
        content: str,
        error: str | None,
        t0: float,
    ) -> None:
        """Emit tool_result + push tool_message do history + zapsat audit.
        Cancellation hardening: pokud je task cancelled během out_queue.put,
        fire-and-forget audit aby forensic stopa nezmizela. Audit samotný
        je shieldnutý → outer cancel ho nezruší (běží detached do konce)."""
        try:
            await self._out_queue.put(
                {"type": "tool_result", "id": tcid, "ok": ok, "content": content}
            )
        except asyncio.CancelledError:
            self._audit_cancellation_fire_and_forget(
                tcid, name, args, perm, approval, t0,
                stage="cancelled during tool_result emission",
            )
            raise
        self.messages.append(tool_message(tcid, name, content))
        if self.audit_log is not None and self.audit_log.enabled:
            duration_ms = int((time.monotonic() - t0) * 1000)
            audit_task = asyncio.create_task(self.audit_log.record(
                turn_id=str(self.turn_state.get("id", "")),
                tool_call_id=tcid,
                tool_name=name,
                args=args,
                permission={
                    "decision": perm.decision.value if hasattr(perm.decision, "value") else str(perm.decision),
                    "risk": perm.risk,
                    "reason": perm.reason,
                    "requires_explicit": perm.requires_explicit,
                    "summary": perm.summary,
                },
                approval=approval,
                ok=ok,
                error=error,
                result_bytes=len(content.encode("utf-8")),
                duration_ms=duration_ms,
            ))
            audit_task.add_done_callback(_swallow_task_exc)
            try:
                # Shield: pokud caller (_execute_one) je cancelled, audit_task
                # pokračuje detached. Tady jen čekáme do dokončení v happy path.
                await asyncio.shield(audit_task)
            except asyncio.CancelledError:
                # audit_task běží dál; přidaný done callback zaloguje případnou exc.
                raise
            except Exception:
                log.exception("audit: record failed for %s", name)


_SENTINEL = object()


def _swallow_task_exc(task: "asyncio.Task[None]") -> None:
    """Done-callback pro fire-and-forget audit tasky. Bez tohoto by event loop
    při .exception() != None vyhodil 'Task exception was never retrieved' warning."""
    try:
        task.exception()
    except (asyncio.CancelledError, Exception):
        pass
