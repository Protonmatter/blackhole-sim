# Repository Guidelines

## Operating Posture

This repository is production-adjacent scientific and systems tooling. Treat changes as deterministic, auditable, testable, and easy to roll back. Preserve public interfaces unless the task explicitly changes them. Prefer small diffs and avoid broad rewrites.

## Layout

- `python/` contains the Python package, tests, examples, public-data manifests, and notebooks.
- `native/` contains native kernel assets and the `native/core` Rust/PyO3 scaffold.
- `web/` contains the WebGPU renderer and interactive browser showcase.
- `docs/` contains build state, native roadmap, and operational notes.
- `.github/workflows/` contains CI only; it must not publish releases without an explicit request.

## Implementation Rules

- Do not hardcode secrets, internal URLs, tokens, tenant IDs, or credentials.
- Keep Python logic typed where practical, deterministic, and separated from I/O.
- Keep the Python fallback path working when optional native or GPU packages are absent.
- Native and GPU code must not claim physics parity until regression tests prove it.
- Do not add generated render outputs, dumps, wheel artifacts, or app bundles to Git.

## Validation

Use the repository commands before claiming a fix:

```bash
cd python
python -m compileall blackhole_sim
python -m pytest -q
cd ..
cargo fmt --check --manifest-path native/core/Cargo.toml
cargo test --manifest-path native/core/Cargo.toml
python -m blackhole_sim.accelerator_cli doctor --json --fail-on-emulation
```

If a tool is missing, state the exact command that could not run and why.

## Git

Do not commit, push, tag, or publish artifacts unless explicitly requested. For release work, record validation evidence in `docs/BUILD_STATE.md`.
