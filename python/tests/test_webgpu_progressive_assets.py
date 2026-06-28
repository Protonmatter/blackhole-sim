from pathlib import Path


def test_webgpu_main_has_progressive_compute_renderer_hooks():
    root = Path(__file__).resolve().parents[2]
    text = (root / "web/webgpu/src/main.js").read_text(encoding="utf-8")
    required = ["dispatchWorkgroups", "progressiveLevels", "stokes", "requestAnimationFrame"]
    for token in required:
        assert token in text
