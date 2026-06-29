from pathlib import Path


def test_showcase_links_resolve_to_repo_webgpu_renderer():
    root = Path(__file__).resolve().parents[2]
    html = (root / "web/showcase/index.html").read_text(encoding="utf-8")

    assert "../webgpu/index.html?shader=stokes" in html
    assert "../simulator/webgpu" not in html
    assert (root / "web/webgpu/index.html").exists()


def test_webgpu_entrypoint_has_recoverable_unavailable_state():
    root = Path(__file__).resolve().parents[2]
    js = (root / "web/webgpu/src/main.js").read_text(encoding="utf-8")

    assert "showWebGPUFallback" in js
    assert "requestAdapter().catch" in js
    assert "requestDevice().catch" in js
    assert "throw new Error('WebGPU unavailable')" not in js
