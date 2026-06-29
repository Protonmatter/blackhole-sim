# CI Notes

CI is defined in `.github/workflows/`.

The current workflows are smoke gates for the native-foundation milestone. They upload build artifacts where useful but do not publish packages, create releases, sign apps, or notarize macOS bundles.

Native wheel jobs must install the wheel they just built and run `blackhole-accelerators doctor --json --fail-on-emulation` on the target runner before uploading artifacts.
