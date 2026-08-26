<!--
Thanks for contributing to Sorbed! Fill in the sections below.
Delete any that genuinely don't apply — but read each one first.
-->

## Summary

<!-- What does this PR do and why? Link the issue it closes. -->

Closes #

## Type of change

- [ ] 🐛 Bug fix
- [ ] ✨ Feature
- [ ] 🩺 Clinical / staging logic change
- [ ] 📊 Model or training change
- [ ] 📝 Documentation
- [ ] 🔧 Refactor / tooling (no behavior change)

## What changed

<!-- Bullet the concrete changes. -->

-

## How it was verified

<!-- Real verification only. Show the command and the actual output, or the
     test that runs the real pipeline on a real fixture. -->

```
$ make check
...
```

## Integrity checklist

- [ ] No hardcoded, stubbed, or fabricated analysis outputs were introduced.
- [ ] `scripts/watchdog.py` passes (run via `make check`).
- [ ] New analysis code has a test that exercises the **real** pipeline on a
      committed fixture and asserts on **computed** values.
- [ ] `CHANGELOG.md` updated under `[Unreleased]`.

## Clinical & equity impact

<!-- Required if this touches staging, tissue classification, or thresholds. -->

- Expected effect on staging behavior:
- Expected effect on performance **across skin tones**:
- Sources / guidelines cited in code:

## Notes for reviewers

<!-- Anything that needs a closer look, known limitations, follow-ups. -->
