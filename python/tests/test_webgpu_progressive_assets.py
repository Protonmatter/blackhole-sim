from pathlib import Path


def test_webgpu_main_has_progressive_compute_renderer_hooks():
    root = Path(__file__).resolve().parents[2]
    text = (root / "web/webgpu/src/main.js").read_text(encoding="utf-8")
    required = [
        "adapterInfoLabel",
        "activeRenderScale",
        "applyControlState",
        "blackholeWebgpuDiagnostics",
        "captureFrame",
        "diagnosticsEnabled",
        "fetchShaderText",
        "cache: 'no-store'",
        "copyBufferToBuffer",
        "GPUMapMode.READ",
        "GPUBufferUsage.STORAGE",
        "PRESETS",
        "QUALITY",
        "ultra: { steps: 520, scale: 1.9 }",
        "simulationState",
        "setPlayState",
        "setQuality",
        "updateSimulationClock",
        "updateTelemetry",
        "WebGPU direct compute: Stokes coefficient bricks",
        "writeFragmentParams",
        "writeStokesParams",
        "dispatchWorkgroups",
        "progressiveLevels",
        "Math.min(1.15",
        "stokes",
        "requestAnimationFrame",
    ]
    for token in required:
        assert token in text

    interaction_tokens = [
        "canvas.addEventListener('pointerdown'",
        "canvas.addEventListener('pointermove'",
        "canvas.addEventListener('wheel'",
        "document.querySelectorAll('[data-preset]')",
        "document.querySelectorAll('[data-quality]')",
    ]
    for token in interaction_tokens:
        assert token in text

    display = (root / "web/webgpu/src/display_stokes.wgsl").read_text(encoding="utf-8")
    assert "log(1.0 + 0.42 * I)" in display
    assert "smoothstep(0.015, 0.55, signal)" in display
    assert "thermal" in display
    assert "polarized" in display


def test_webgpu_interactive_shader_work_is_bounded():
    root = Path(__file__).resolve().parents[2]
    index = (root / "web/webgpu/index.html").read_text(encoding="utf-8")
    assert "Kerr WebGPU Simulation" in index
    assert 'id="playToggle"' in index
    assert 'id="qualityControls"' in index
    assert 'data-quality="ultra"' in index
    assert 'id="fpsValue"' in index
    assert 'data-preset="polarized"' in index
    assert 'id="inc" type="range" min="5" max="88" step="1" value="80"' in index
    assert 'id="camr" type="range" min="12" max="120" step="1" value="18"' in index
    assert 'id="steps" type="range" min="80" max="520" step="20" value="520"' in index

    for shader_name in ("kerr_raytrace.wgsl", "grrt_volume.wgsl"):
        text = (root / f"web/webgpu/src/{shader_name}").read_text(encoding="utf-8")
        assert "const MAX_FRAGMENT_STEPS: i32 = 520" in text
        assert "clamp(i32(U.steps_f), 1, MAX_FRAGMENT_STEPS)" in text
        assert "for (var i=0; i<MAX_FRAGMENT_STEPS; i=i+1)" in text
        assert "hash21" in text
        assert "skyColor" in text
        assert "captured" in text
        assert "escaped" in text
        assert "for (var i=0; i<1400; i=i+1)" not in text

    stokes = (root / "web/webgpu/src/stokes_brick_compute.wgsl").read_text(encoding="utf-8")
    assert "let maxSteps = min(params.max_steps, 520u)" in stokes
