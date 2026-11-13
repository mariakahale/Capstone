"""
Shared setup for the tests. pytest runs this file automatically before the tests.

It tells Python where to find the code being tested (the same folders the scripts add
to the path themselves when they run).
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for folder in [REPO_ROOT, REPO_ROOT / "pipeline", REPO_ROOT / "code_weather"]:
    sys.path.insert(0, str(folder))
