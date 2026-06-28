from blackhole_sim.native_loader import native_core_status
from blackhole_sim.platform_probe import doctor_report, normalize_arch, runtime_arch_report


def test_normalize_arch_aliases():
    assert normalize_arch("AMD64") == "x86_64"
    assert normalize_arch("aarch64") == "arm64"
    assert normalize_arch("") == "unknown"


def test_runtime_arch_report_has_release_gate_fields():
    report = runtime_arch_report()
    assert report["process_arch"]
    assert report["python_arch"]
    assert report["pointer_bits"] in {32, 64}
    assert isinstance(report["emulation_detected"], bool)


def test_native_core_status_is_non_throwing():
    status = native_core_status()
    assert "native_core_loaded" in status
    assert "reason" in status


def test_doctor_report_is_non_throwing():
    report = doctor_report(".")
    assert "gpu_backend" in report
    assert "warnings" in report
