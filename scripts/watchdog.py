#!/usr/bin/env python3
"""Sorbed integrity watchdog.

This guard exists because Sorbed is clinical software: every value it reports
must come from a real computation on real pixels. The watchdog fails the build
when it finds code that fakes, stubs, or short-circuits that contract.

It checks three things:

1. **Forbidden markers** — placeholder/stub/TODO-style tokens in shipping source
   (``src/sorbed``). Legitimate abstract methods (``@abstractmethod`` +
   ``raise NotImplementedError`` / ``...``) and anything on a line ending with
   ``# watchdog: allow`` are exempt.
2. **Empty implementations** — non-abstract functions whose entire body is
   ``pass`` or ``...`` (a silent stub) outside of Protocol/typing-only files.
3. **Phase manifest** — every phase marked ``done`` in ``PHASES.yaml`` must have
   its declared ``artifacts`` present on disk, so a phase cannot be reported
   complete while its deliverables are missing.

Run it directly (``python scripts/watchdog.py``) or via ``make check``. It is
also wired as a pre-commit hook and a CI job.

Exit code 0 = clean, 1 = violations found, 2 = watchdog misconfiguration.
"""

from __future__ import annotations

import ast
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_ROOT = REPO_ROOT / "src" / "sorbed"
PHASES_FILE = REPO_ROOT / "PHASES.yaml"

ALLOW_MARKER = "# watchdog: allow"

# Tokens that must never appear in shipping source. Matched case-insensitively
# as whole-ish words so "stubborn" or "mockingbird" don't trip it.
FORBIDDEN_PATTERNS: dict[str, re.Pattern[str]] = {
    "TODO": re.compile(r"\bTODO\b", re.IGNORECASE),
    "FIXME": re.compile(r"\bFIXME\b", re.IGNORECASE),
    "XXX-marker": re.compile(r"\bXXX\b"),
    "placeholder": re.compile(r"\bplaceholder\b", re.IGNORECASE),
    "hardcoded-note": re.compile(r"\bhard[\s-]?cod", re.IGNORECASE),
    "dummy-data": re.compile(r"\bdummy\b", re.IGNORECASE),
    "fake-output": re.compile(r"\bfake\b", re.IGNORECASE),
    "stub": re.compile(r"\bstub(?:bed|s)?\b", re.IGNORECASE),
    "not-implemented-comment": re.compile(r"not[\s_]?implemented", re.IGNORECASE),
    "for-now": re.compile(r"\bfor now\b", re.IGNORECASE),
    "temporary-hack": re.compile(r"\b(hack|kludge|kluge)\b", re.IGNORECASE),
}

# Files/dirs whose *names* are exempt from marker scanning (they legitimately
# discuss these words), relative to SOURCE_ROOT.
MARKER_SCAN_EXEMPT_NAMES: set[str] = set()


@dataclass
class Violation:
    path: Path
    line: int
    kind: str
    detail: str

    def render(self) -> str:
        rel = self.path.relative_to(REPO_ROOT)
        return f"  {rel}:{self.line}  [{self.kind}] {self.detail}"


@dataclass
class Report:
    violations: list[Violation] = field(default_factory=list)
    files_scanned: int = 0
    config_error: str | None = None

    def add(self, v: Violation) -> None:
        self.violations.append(v)


def _iter_source_files() -> list[Path]:
    if not SOURCE_ROOT.exists():
        return []
    return sorted(p for p in SOURCE_ROOT.rglob("*.py") if "__pycache__" not in p.parts)


def _line_is_allowed(line: str) -> bool:
    return ALLOW_MARKER in line


def _abstract_functions(tree: ast.AST) -> set[int]:
    """Return line numbers of functions decorated as abstract."""
    abstract_lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in node.decorator_list:
                name = _decorator_name(dec)
                if name in {"abstractmethod", "abstractproperty"}:
                    abstract_lines.add(node.lineno)
    return abstract_lines


def _decorator_name(dec: ast.expr) -> str:
    if isinstance(dec, ast.Name):
        return dec.id
    if isinstance(dec, ast.Attribute):
        return dec.attr
    if isinstance(dec, ast.Call):
        return _decorator_name(dec.func)
    return ""


def _is_protocol_class_body(node: ast.ClassDef) -> bool:
    return any(_decorator_name(base) in {"Protocol", "ABC"} for base in node.bases)


def scan_markers(path: Path, report: Report) -> None:
    if path.name in MARKER_SCAN_EXEMPT_NAMES:
        return
    text = path.read_text(encoding="utf-8", errors="replace")
    for lineno, line in enumerate(text.splitlines(), start=1):
        if _line_is_allowed(line):
            continue
        for kind, pattern in FORBIDDEN_PATTERNS.items():
            if pattern.search(line):
                report.add(
                    Violation(
                        path=path,
                        line=lineno,
                        kind=f"marker:{kind}",
                        detail=f"forbidden token in shipping source: {line.strip()!r}",
                    )
                )


def scan_empty_bodies(path: Path, report: Report) -> None:
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as exc:
        report.add(
            Violation(path=path, line=exc.lineno or 0, kind="syntax", detail=str(exc))
        )
        return

    abstract_lines = _abstract_functions(tree)
    # Map each function to whether it lives directly in a Protocol/ABC class.
    protocol_func_lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and _is_protocol_class_body(node):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    protocol_func_lines.add(child.lineno)

    lines = text.splitlines()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.lineno in abstract_lines or node.lineno in protocol_func_lines:
            continue
        body = _effective_body(node)
        if _body_is_empty_stub(body):
            # Allow if the def line carries the allow marker.
            def_line = lines[node.lineno - 1] if node.lineno - 1 < len(lines) else ""
            if _line_is_allowed(def_line):
                continue
            report.add(
                Violation(
                    path=path,
                    line=node.lineno,
                    kind="empty-body",
                    detail=(
                        f"function {node.name!r} has no real implementation "
                        "(body is only pass/...); implement it or mark it "
                        "@abstractmethod"
                    ),
                )
            )
        elif _body_raises_not_implemented(body):
            report.add(
                Violation(
                    path=path,
                    line=node.lineno,
                    kind="not-implemented",
                    detail=(
                        f"function {node.name!r} raises NotImplementedError but is "
                        "not @abstractmethod"
                    ),
                )
            )


def _effective_body(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.stmt]:
    body = list(node.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(
        body[0].value, ast.Constant
    ) and isinstance(body[0].value.value, str):
        # Drop leading docstring.
        body = body[1:]
    return body


def _body_is_empty_stub(body: list[ast.stmt]) -> bool:
    if not body:
        return True
    for stmt in body:
        if isinstance(stmt, ast.Pass):
            continue
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and (
            stmt.value.value is Ellipsis
        ):
            continue
        return False
    return True


def _body_raises_not_implemented(body: list[ast.stmt]) -> bool:
    if len(body) != 1:
        return False
    stmt = body[0]
    if not isinstance(stmt, ast.Raise) or stmt.exc is None:
        return False
    name = _decorator_name(stmt.exc if not isinstance(stmt.exc, ast.Call) else stmt.exc.func)
    return name == "NotImplementedError"


def check_phase_manifest(report: Report) -> None:
    if not PHASES_FILE.exists():
        return  # Manifest is optional until the roadmap lands.
    try:
        phases = _load_phase_manifest(PHASES_FILE)
    except Exception as exc:
        report.config_error = f"could not parse {PHASES_FILE.name}: {exc}"
        return
    for phase in phases:
        if phase.get("status") != "done":
            continue
        for artifact in phase.get("artifacts", []):
            target = REPO_ROOT / artifact
            if not target.exists():
                report.add(
                    Violation(
                        path=PHASES_FILE,
                        line=0,
                        kind="phase-artifact-missing",
                        detail=(
                            f"phase {phase.get('id', '?')} is marked done but "
                            f"artifact {artifact!r} does not exist"
                        ),
                    )
                )


def _load_phase_manifest(path: Path) -> list[dict]:
    """Minimal YAML-list loader.

    Uses PyYAML when available; otherwise falls back to a tiny parser that
    understands the restricted structure of PHASES.yaml so the watchdog has no
    hard third-party dependency.
    """
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
        return list(data.get("phases", [])) if isinstance(data, dict) else list(data)
    except ModuleNotFoundError:
        return _parse_phases_fallback(text)


def _parse_phases_fallback(text: str) -> list[dict]:
    phases: list[dict] = []
    current: dict | None = None
    current_list_key: str | None = None
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        stripped = raw.strip()
        indent = len(raw) - len(raw.lstrip())
        if stripped.startswith("- ") and indent <= 2:
            if current is not None:
                phases.append(current)
            current = {}
            current_list_key = None
            stripped = stripped[2:]
        if current is None:
            continue
        if stripped.startswith("- ") and current_list_key is not None:
            current.setdefault(current_list_key, []).append(
                _coerce(stripped[2:].strip())
            )
            continue
        if ":" in stripped:
            key, _, value = stripped.partition(":")
            key = key.strip()
            value = value.strip()
            if value == "":
                current_list_key = key
                current.setdefault(key, [])
            else:
                current[key] = _coerce(value)
                current_list_key = None
    if current is not None:
        phases.append(current)
    return phases


def _coerce(value: str) -> object:
    value = value.strip().strip('"').strip("'")
    return value


def main(argv: list[str] | None = None) -> int:
    report = Report()
    files = _iter_source_files()
    for path in files:
        report.files_scanned += 1
        scan_markers(path, report)
        scan_empty_bodies(path, report)
    check_phase_manifest(report)

    if report.config_error:
        print(f"watchdog: configuration error: {report.config_error}", file=sys.stderr)
        return 2

    if report.violations:
        print(
            f"watchdog: FAILED — {len(report.violations)} integrity violation(s) "
            f"across {report.files_scanned} source file(s):\n",
            file=sys.stderr,
        )
        for v in sorted(report.violations, key=lambda x: (str(x.path), x.line)):
            print(v.render(), file=sys.stderr)
        print(
            "\nSorbed forbids placeholder, stub, or fabricated-output code paths.\n"
            "Implement the real behavior, or if a line is a false positive append "
            f"'{ALLOW_MARKER}' with justification.",
            file=sys.stderr,
        )
        return 1

    print(
        f"watchdog: OK — {report.files_scanned} source file(s) scanned, "
        "no placeholder/stub/fabricated-output violations."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
