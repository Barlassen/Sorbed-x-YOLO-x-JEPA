# Contributing to Sorbed

Thanks for helping build a fairer, more transparent wound-assessment tool. This
document explains how to get set up, the standards every change is held to, and
the one rule we never bend.

## The one rule: no fabricated outputs

Sorbed is clinical software. **Every number it reports must come from a real
computation on real pixels.** We do not merge:

- Hardcoded or stubbed analysis results
- `TODO`/placeholder returns in production code paths
- Tests that assert against fabricated "expected" values instead of computed
  ones
- Demo modes that fake a grade

The `scripts/watchdog.py` guard runs in CI and locally to enforce this. If your
change legitimately needs a not-yet-implemented path, raise it in an issue
first — don't stub it silently.

## Getting set up

```bash
git clone https://github.com/ArioMoniri/Sorbed.git
cd Sorbed
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,cpu]"      # CPU stack + dev tools
pre-commit install                # formatting, linting, watchdog hooks
```

Run the full check suite the way CI does:

```bash
make check      # ruff + mypy + watchdog + pytest
```

## Development workflow

1. **Open an issue** describing the change (bug, feature, or clinical concern).
   Use the templates.
2. **Branch** from `main`: `git checkout -b feat/short-description`.
3. **Write the code and a real test.** New analysis code needs a test that runs
   the real pipeline on a real (committed, de-identified or synthetic-yet-honest)
   fixture image and checks properties of the output.
4. **Run `make check`.** Everything must pass, including the watchdog.
5. **Update `CHANGELOG.md`** under `[Unreleased]`.
6. **Open a pull request** using the template. Link the issue.

## Code standards

- **Python 3.10+**, typed. `mypy --strict` on `src/sorbed`.
- **Formatting & linting**: `ruff format` and `ruff check` (config in
  `pyproject.toml`). No manual style debates.
- **Docstrings**: every public function/class. Clinical logic must cite its
  source (guideline, paper) in the docstring.
- **Commits**: [Conventional Commits](https://www.conventionalcommits.org/)
  (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`).

## Clinical & equity review

Changes that touch staging logic, tissue classification, or thresholds require a
second reviewer and a note in the PR describing the expected effect on
**performance across skin tones**. Equity regressions are treated as bugs.

## Reporting security or privacy issues

Do not open a public issue. See `SECURITY.md`.

## License of contributions

By contributing, you agree that your contributions are licensed under the
Apache License 2.0, consistent with the rest of the project.
