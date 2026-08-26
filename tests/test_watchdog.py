"""Meta-tests: the integrity watchdog must actually catch violations."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_WATCHDOG_PATH = Path(__file__).resolve().parent.parent / "scripts" / "watchdog.py"


def _load_watchdog():
    spec = importlib.util.spec_from_file_location("sorbed_watchdog", _WATCHDOG_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclasses can resolve the module namespace.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def watchdog():
    return _load_watchdog()


def test_detects_forbidden_marker(watchdog, tmp_path: Path):
    f = tmp_path / "bad.py"
    f.write_text("def analyze():\n    return 0  # TODO real computation\n")
    report = watchdog.Report()
    watchdog.scan_markers(f, report)
    assert any("marker" in v.kind for v in report.violations)


def test_detects_empty_stub_body(watchdog, tmp_path: Path):
    f = tmp_path / "stubbed.py"
    f.write_text("def compute_area():\n    pass\n")
    report = watchdog.Report()
    watchdog.scan_empty_bodies(f, report)
    assert any(v.kind == "empty-body" for v in report.violations)


def test_allows_abstract_method(watchdog, tmp_path: Path):
    f = tmp_path / "ok.py"
    f.write_text(
        "from abc import abstractmethod\n"
        "class A:\n"
        "    @abstractmethod\n"
        "    def f(self):\n"
        "        ...\n"
    )
    report = watchdog.Report()
    watchdog.scan_empty_bodies(f, report)
    assert not report.violations


def test_real_source_tree_passes(watchdog):
    # The actual shipping source must contain zero violations.
    assert watchdog.main([]) == 0
