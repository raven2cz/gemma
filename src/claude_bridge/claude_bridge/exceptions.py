"""Exception taxonomy pro claude_bridge.

Všechny adapter-specific chyby derivují z ClaudeBridgeError, aby caller
mohl chytit jednotlivé kategorie nebo všechny.

Hierarchie:
    ClaudeBridgeError                # base
      AdapterConfigError              # missing tmux bin, missing pyte, ...
      SessionBusy                     # konkurentní ask() na running session
      SessionDead                     # session je v terminal DEAD state
      SessionNotFound                 # session_id neexistuje
      SubprocessError                 # spawn/exec error v claude CLI
"""
from __future__ import annotations


class ClaudeBridgeError(Exception):
    """Base class pro všechny claude_bridge exceptions."""


class AdapterConfigError(ClaudeBridgeError):
    """Config nebo dependencies pro selected adapter incomplete.

    Příklady:
      - BridgeMode.TMUX zvolen, ale `tmux` binary není v PATH
      - BridgeMode.TMUX zvolen, ale `pyte` library není naistalovaná
      - mode="edit" bez workdir
    """


class SessionBusy(ClaudeBridgeError):
    """Konkurentní ask() volání na session co je RUNNING / CANCELING.

    Adapter používá per-session asyncio.Lock - druhý request by se zablokoval.
    Místo silent queueing raise tuto chybu, caller se může rozhodnout
    (zařadit, čekat manuálně, vrátit error userovi).
    """


class SessionDead(ClaudeBridgeError):
    """Session je v terminal DEAD state - subprocess zemřel, byl timeoutován,
    nebo externí kill. NIKDY silent recreate (codex iter-2 high #13);
    UI musí explicit nabídnout fresh start s novým session_id.
    """


class SessionNotFound(ClaudeBridgeError):
    """session_id neexistuje v adapter registru."""


class SubprocessError(ClaudeBridgeError):
    """Claude CLI subprocess error (spawn failed, non-zero exit, no result
    event). Detail je v message a optional `exit_code` / `stderr_preview`
    attributech.
    """

    def __init__(self, message: str, *, exit_code: int | None = None,
                 stderr_preview: str | None = None) -> None:
        super().__init__(message)
        self.exit_code = exit_code
        self.stderr_preview = stderr_preview
