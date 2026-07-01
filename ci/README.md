# CI/CD Notes

CI/CD is defined in `.github/workflows/`.

## Pull Request And Main CI

`ci.yml` is the required validation pipeline for pull requests and pushes to
`main`. It runs:

- repository hygiene and workflow YAML parsing,
- Python compile/test/accelerator-doctor gates across Linux, Windows, and macOS,
- focused science-contract tests plus benchmark and accelerated-render smoke,
- Rust format and native-core tests,
- native wheel builds on x64/ARM Linux, Windows, and macOS,
- native wheel install, architecture doctor, parity tests, and benchmark smoke.

The final `ci-required` job is the branch-protection target. It fails if any
upstream validation job fails or is skipped.

## Artifact Delivery

`release-artifacts.yml` runs manually or on `v*` tags. It builds Python package
artifacts and native wheels, smoke-installs them, and uploads artifacts for
review. It does not publish to PyPI, create GitHub releases, sign binaries, or
notarize macOS bundles.

## Dependency Maintenance

`.github/dependabot.yml` opens weekly update PRs for GitHub Actions, Python
package metadata, and the Rust native core.
