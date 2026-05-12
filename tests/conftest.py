"""Pytest config — přidá project root na sys.path aby testy mohly importovat
`voice.*` bez instalace.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _reset_permission_degrade_state():
    """Phase 9.4: module-level state v `permissions._DESTRUCTIVE_APPROVAL_TS`
    přežívá napříč testy ve stejném pytest procesu. E2E test, který schválí
    destruktivní operaci (test_e2e_destructive_requires_phrase atd.), by jinak
    degradoval AUTO classifier rozhodnutí v následujících testech a způsobil
    nečekané ASK approval modaly → test timeout.
    """
    yield
    try:
        from voice.agent.permissions import clear_degrade_state
        clear_degrade_state()
    except ImportError:
        pass
