"""``--fast`` must skip ``@pytest.mark.slow`` tests, not merely subsample grids.

Regression test for a selection gap that made the project's own prescribed
pre-commit command unusable. ``CLAUDE.md`` says to run
``pytest tests/ --fast -x -q`` before every commit, but ``--fast`` only fed the
``fast_mode`` fixture (within-test subsampling); nothing acted on the ``slow``
marker, so all 21 slow tests ran and the command could not finish. Measured on
bare ``main``: terminated at 3:00:01 by its harness timeout, with an identical
timeout on a branch -- so every "full suite" run in that state was silently
truncated rather than green.

These call the real ``pytest_collection_modifyitems`` hook from ``conftest``
directly, rather than re-implementing its logic, so a change to the hook is
caught here.
"""
import importlib.util
from pathlib import Path

# Load the real tests/conftest.py by path: the project does not place tests/ on
# sys.path, so a bare ``import conftest`` fails. Loading the actual file (rather
# than copying its logic here) is the point -- these tests must break if the
# hook changes.
_CONFTEST_PATH = Path(__file__).resolve().parent / "conftest.py"
_spec = importlib.util.spec_from_file_location("_clax_tests_conftest", _CONFTEST_PATH)
conftest = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(conftest)


class _Config:
    """Minimal stand-in for pytest's ``config`` exposing only ``--fast``."""

    def __init__(self, fast: bool):
        self._fast = fast

    def getoption(self, name):
        assert name == "--fast", f"hook queried an unexpected option: {name}"
        return self._fast


class _Item:
    """Minimal stand-in for a collected item: keywords plus added markers."""

    def __init__(self, *, slow: bool):
        self.keywords = {"slow": True} if slow else {}
        self.added_markers = []

    def add_marker(self, marker):
        self.added_markers.append(marker)


def _run_hook(fast: bool):
    slow_item, quick_item = _Item(slow=True), _Item(slow=False)
    conftest.pytest_collection_modifyitems(_Config(fast), [slow_item, quick_item])
    return slow_item, quick_item


def test_fast_skips_slow_items():
    slow_item, quick_item = _run_hook(fast=True)
    assert len(slow_item.added_markers) == 1, "slow test was not skipped under --fast"
    assert slow_item.added_markers[0].name == "skip"
    assert quick_item.added_markers == [], "a non-slow test must never be skipped"


def test_fast_skip_reason_is_actionable():
    """The reason must tell the reader how to get the full suite back."""
    slow_item, _ = _run_hook(fast=True)
    reason = slow_item.added_markers[0].kwargs["reason"]
    assert "--fast" in reason and "full suite" in reason, reason


def test_without_fast_nothing_is_skipped():
    slow_item, quick_item = _run_hook(fast=False)
    assert slow_item.added_markers == [], "slow tests must still run without --fast"
    assert quick_item.added_markers == []


def test_slow_marker_is_declared():
    """The hook keys on ``slow``; keep it a declared marker so it is not a typo."""
    import tomllib
    from pathlib import Path

    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    cfg = tomllib.loads(pyproject.read_text())
    markers = cfg["tool"]["pytest"]["ini_options"]["markers"]
    assert any(m.startswith("slow:") for m in markers), markers
