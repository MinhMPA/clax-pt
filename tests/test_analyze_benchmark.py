# tests/test_analyze_benchmark.py
import json
import sys
from pathlib import Path
import pytest

FIXTURES = Path(__file__).parent / "fixtures" / "benchmark"
sys.path.insert(0, str(Path(__file__).parent.parent))
