from blackhole_sim.accelerator import detect_backends, make_render_plan, choose_backend
import blackhole_sim.accelerator as accelerator


def test_detect_backends_always_has_cpu():
    backends = detect_backends('.')
    by_name = {b.name: b for b in backends}
    assert 'cpu' in by_name
    assert by_name['cpu'].available is True
    assert 'webgpu' in by_name


def test_render_plan_full_hd_cpu_warns():
    plan = make_render_plan(1920, 1080, backend='cpu', project_root='.')
    assert plan.pixels == 1920 * 1080
    assert plan.tiles > 0
    assert plan.backend == 'cpu'
    assert any('CPU reference' in w for w in plan.warnings)


def test_interactive_plan_prefers_interactive_backend_when_available():
    plan = make_render_plan(1280, 720, backend='interactive', project_root='.')
    assert plan.pixels == 1280 * 720
    assert plan.tile_size == 64
    assert plan.backend in {'webgpu', 'metal', 'cpu', 'cuda', 'rocm'}


def test_choose_backend_returns_available():
    chosen = choose_backend(detect_backends('.'))
    assert chosen.available


def test_webgpu_backend_reports_os_gpu_devices_when_detected(monkeypatch):
    monkeypatch.setattr(accelerator, "_os_gpu_devices", lambda: ("Qualcomm Adreno test adapter",))

    backend = accelerator.detect_webgpu(project_root=".")

    assert backend.name == "webgpu"
    assert backend.devices == ("Qualcomm Adreno test adapter",)
    assert "OS GPU adapter detected" in backend.reason
    assert any("direct browser GPU compute path" in note for note in backend.notes)
