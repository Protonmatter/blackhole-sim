use pyo3::prelude::*;
use pyo3::types::PyDict;

fn normalized_arch(value: &str) -> &'static str {
    match value {
        "x86_64" | "amd64" => "x86_64",
        "aarch64" | "arm64" => "arm64",
        "x86" | "i386" | "i686" => "x86",
        "arm" => "arm",
        _ => "unknown",
    }
}

#[pyfunction]
fn core_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pyfunction]
fn detect_arch(py: Python<'_>) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("arch", normalized_arch(std::env::consts::ARCH))?;
    dict.set_item("os", std::env::consts::OS)?;
    dict.set_item("family", std::env::consts::FAMILY)?;
    dict.set_item(
        "pointer_width",
        (std::mem::size_of::<usize>() * 8).to_string(),
    )?;
    Ok(dict.into())
}

#[pymodule]
fn blackhole_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(core_version, m)?)?;
    m.add_function(wrap_pyfunction!(detect_arch, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::normalized_arch;

    #[test]
    fn normalizes_common_architectures() {
        assert_eq!(normalized_arch("x86_64"), "x86_64");
        assert_eq!(normalized_arch("aarch64"), "arm64");
        assert_eq!(normalized_arch("unsupported"), "unknown");
    }
}
