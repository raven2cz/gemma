"""Pytest config — přidá project root na sys.path aby testy mohly importovat
`voice.*` bez instalace.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
