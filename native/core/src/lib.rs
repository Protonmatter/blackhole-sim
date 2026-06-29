use pyo3::exceptions::PyValueError;
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

fn stokes_rhs(stokes: [f64; 4], coeff: &[f64]) -> [f64; 4] {
    let j_i = coeff[0];
    let j_q = coeff[1];
    let j_u = coeff[2];
    let j_v = coeff[3];
    let alpha_i = coeff[4];
    let alpha_q = coeff[5];
    let alpha_u = coeff[6];
    let alpha_v = coeff[7];
    let rho_v = coeff[8];
    let rho_q = coeff[9];
    let rho_u = coeff[10];

    [
        j_i - (alpha_i * stokes[0]
            + alpha_q * stokes[1]
            + alpha_u * stokes[2]
            + alpha_v * stokes[3]),
        j_q - (alpha_q * stokes[0] + alpha_i * stokes[1] + rho_v * stokes[2] - rho_u * stokes[3]),
        j_u - (alpha_u * stokes[0] - rho_v * stokes[1] + alpha_i * stokes[2] + rho_q * stokes[3]),
        j_v - (alpha_v * stokes[0] + rho_u * stokes[1] - rho_q * stokes[2] + alpha_i * stokes[3]),
    ]
}

fn stokes_step_rk2(stokes: [f64; 4], coeff: &[f64], ds_cm: f64) -> [f64; 4] {
    let k1 = stokes_rhs(stokes, coeff);
    let mid = [
        stokes[0] + 0.5 * ds_cm * k1[0],
        stokes[1] + 0.5 * ds_cm * k1[1],
        stokes[2] + 0.5 * ds_cm * k1[2],
        stokes[3] + 0.5 * ds_cm * k1[3],
    ];
    let k2 = stokes_rhs(mid, coeff);
    [
        stokes[0] + ds_cm * k2[0],
        stokes[1] + ds_cm * k2[1],
        stokes[2] + ds_cm * k2[2],
        stokes[3] + ds_cm * k2[3],
    ]
}

#[pyfunction]
fn stokes_rk2_brick(coeffs: Vec<f64>, ds_cm: f64, initial: Vec<f64>) -> PyResult<Vec<f64>> {
    if !ds_cm.is_finite() || ds_cm < 0.0 {
        return Err(PyValueError::new_err(
            "ds_cm must be finite and non-negative",
        ));
    }
    if coeffs.is_empty() || coeffs.len() % 11 != 0 {
        return Err(PyValueError::new_err(
            "coeffs must contain one or more flat 11-coefficient cells",
        ));
    }
    if coeffs.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err(
            "coeffs must contain only finite values",
        ));
    }
    let cells = coeffs.len() / 11;
    let broadcast_initial = initial.len() == 4;
    if !broadcast_initial && initial.len() != cells * 4 {
        return Err(PyValueError::new_err(
            "initial must contain either 4 values or 4 values per coefficient cell",
        ));
    }
    if initial.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err(
            "initial must contain only finite values",
        ));
    }

    let mut out = Vec::with_capacity(cells * 4);
    for idx in 0..cells {
        let coeff_start = idx * 11;
        let init_start = if broadcast_initial { 0 } else { idx * 4 };
        let stokes = [
            initial[init_start],
            initial[init_start + 1],
            initial[init_start + 2],
            initial[init_start + 3],
        ];
        let stepped = stokes_step_rk2(stokes, &coeffs[coeff_start..coeff_start + 11], ds_cm);
        out.extend_from_slice(&stepped);
    }
    Ok(out)
}

#[pymodule]
fn blackhole_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(core_version, m)?)?;
    m.add_function(wrap_pyfunction!(detect_arch, m)?)?;
    m.add_function(wrap_pyfunction!(stokes_rk2_brick, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{normalized_arch, stokes_rk2_brick};

    #[test]
    fn normalizes_common_architectures() {
        assert_eq!(normalized_arch("x86_64"), "x86_64");
        assert_eq!(normalized_arch("aarch64"), "arm64");
        assert_eq!(normalized_arch("unsupported"), "unknown");
    }

    #[test]
    fn stokes_rk2_brick_matches_scalar_absorption_case() {
        let coeffs = vec![1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let out = stokes_rk2_brick(coeffs, 0.1, vec![0.0, 0.0, 0.0, 0.0]).unwrap();
        assert!((out[0] - 0.0975).abs() < 1.0e-12);
        assert_eq!(out[1], 0.0);
        assert_eq!(out[2], 0.0);
        assert_eq!(out[3], 0.0);
    }

    #[test]
    fn stokes_rk2_brick_rejects_bad_shapes() {
        assert!(stokes_rk2_brick(vec![0.0; 10], 0.1, vec![0.0; 4]).is_err());
    }
}
