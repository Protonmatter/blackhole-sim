use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

const COEFFS_PER_CELL: usize = 11;
const STOKES_COMPONENTS: usize = 4;
const TWO_PI: f64 = 2.0 * std::f64::consts::PI;

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

fn search_right(grid: &[f64], value: f64) -> usize {
    let mut lo = 0usize;
    let mut hi = grid.len();
    while lo < hi {
        let mid = lo + ((hi - lo) / 2);
        if value < grid[mid] {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    lo
}

fn bracket_linear(grid: &[f64], value: f64) -> Option<(usize, usize, f64)> {
    if value < grid[0] || value > grid[grid.len() - 1] {
        return None;
    }
    if value == grid[grid.len() - 1] {
        return Some((grid.len() - 2, grid.len() - 1, 1.0));
    }
    let right = search_right(grid, value);
    let i0 = right.saturating_sub(1).min(grid.len() - 2);
    let i1 = i0 + 1;
    let denom = (grid[i1] - grid[i0]).max(1.0e-300);
    Some((i0, i1, (value - grid[i0]) / denom))
}

fn wrap_phi(phi: f64, base: f64) -> f64 {
    (phi - base).rem_euclid(TWO_PI) + base
}

fn bracket_periodic_phi(grid: &[f64], phi: f64) -> (usize, usize, f64) {
    let p = wrap_phi(phi, grid[0]);
    let n = grid.len();
    let right = search_right(grid, p);
    let mut i0 = right.saturating_sub(1);
    if i0 >= n {
        i0 = n - 1;
    }
    let i1 = (i0 + 1) % n;
    let hi = if i1 > i0 { grid[i1] } else { grid[0] + TWO_PI };
    let pp = if p >= grid[i0] { p } else { p + TWO_PI };
    let denom = (hi - grid[i0]).max(1.0e-300);
    (i0, i1, (pp - grid[i0]) / denom)
}

fn coeff_offset(ir: usize, itheta: usize, iphi: usize, ntheta: usize, nphi: usize) -> usize {
    (((ir * ntheta) + itheta) * nphi + iphi) * COEFFS_PER_CELL
}

fn validate_grid(name: &str, grid: &[f64]) -> PyResult<()> {
    if grid.len() < 2 {
        return Err(PyValueError::new_err(format!(
            "{name} grid must contain at least 2 values"
        )));
    }
    if grid.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err(format!(
            "{name} grid must contain only finite values"
        )));
    }
    if grid.windows(2).any(|pair| pair[1] <= pair[0]) {
        return Err(PyValueError::new_err(format!(
            "{name} grid must be strictly increasing"
        )));
    }
    Ok(())
}

fn validate_sampler_inputs(
    coeffs: &[f64],
    r_grid: &[f64],
    theta_grid: &[f64],
    phi_grid: &[f64],
    points: &[f64],
) -> PyResult<usize> {
    validate_grid("r", r_grid)?;
    validate_grid("theta", theta_grid)?;
    validate_grid("phi", phi_grid)?;
    if coeffs.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err(
            "coeffs must contain only finite values",
        ));
    }
    let expected = r_grid
        .len()
        .checked_mul(theta_grid.len())
        .and_then(|value| value.checked_mul(phi_grid.len()))
        .and_then(|value| value.checked_mul(COEFFS_PER_CELL))
        .ok_or_else(|| PyValueError::new_err("coefficient brick shape is too large"))?;
    if coeffs.len() != expected {
        return Err(PyValueError::new_err(format!(
            "coeffs must contain exactly r*theta*phi*11 values; expected {expected}, got {}",
            coeffs.len()
        )));
    }
    if points.len() % 3 != 0 {
        return Err(PyValueError::new_err(
            "points must contain flat r, theta, phi triples",
        ));
    }
    if points.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err(
            "points must contain only finite values",
        ));
    }
    Ok(points.len() / 3)
}

fn sample_coefficients(
    coeffs: &[f64],
    r_grid: &[f64],
    theta_grid: &[f64],
    phi_grid: &[f64],
    r: f64,
    theta: f64,
    phi: f64,
) -> [f64; COEFFS_PER_CELL] {
    let Some((r0, r1, wr)) = bracket_linear(r_grid, r) else {
        return [0.0; COEFFS_PER_CELL];
    };
    let Some((t0, t1, wt)) = bracket_linear(theta_grid, theta) else {
        return [0.0; COEFFS_PER_CELL];
    };
    let (p0, p1, wp) = bracket_periodic_phi(phi_grid, phi);
    let ntheta = theta_grid.len();
    let nphi = phi_grid.len();
    let o000 = coeff_offset(r0, t0, p0, ntheta, nphi);
    let o001 = coeff_offset(r0, t0, p1, ntheta, nphi);
    let o010 = coeff_offset(r0, t1, p0, ntheta, nphi);
    let o011 = coeff_offset(r0, t1, p1, ntheta, nphi);
    let o100 = coeff_offset(r1, t0, p0, ntheta, nphi);
    let o101 = coeff_offset(r1, t0, p1, ntheta, nphi);
    let o110 = coeff_offset(r1, t1, p0, ntheta, nphi);
    let o111 = coeff_offset(r1, t1, p1, ntheta, nphi);
    let mut out = [0.0; COEFFS_PER_CELL];
    for idx in 0..COEFFS_PER_CELL {
        let c00 = (1.0 - wp) * coeffs[o000 + idx] + wp * coeffs[o001 + idx];
        let c01 = (1.0 - wp) * coeffs[o010 + idx] + wp * coeffs[o011 + idx];
        let c10 = (1.0 - wp) * coeffs[o100 + idx] + wp * coeffs[o101 + idx];
        let c11 = (1.0 - wp) * coeffs[o110 + idx] + wp * coeffs[o111 + idx];
        let c0 = (1.0 - wt) * c00 + wt * c01;
        let c1 = (1.0 - wt) * c10 + wt * c11;
        out[idx] = (1.0 - wr) * c0 + wr * c1;
    }
    out
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
    if coeffs.is_empty() || coeffs.len() % COEFFS_PER_CELL != 0 {
        return Err(PyValueError::new_err(
            "coeffs must contain one or more flat 11-coefficient cells",
        ));
    }
    if coeffs.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err(
            "coeffs must contain only finite values",
        ));
    }
    let cells = coeffs.len() / COEFFS_PER_CELL;
    let broadcast_initial = initial.len() == STOKES_COMPONENTS;
    if !broadcast_initial && initial.len() != cells * STOKES_COMPONENTS {
        return Err(PyValueError::new_err(
            "initial must contain either 4 values or 4 values per coefficient cell",
        ));
    }
    if initial.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err(
            "initial must contain only finite values",
        ));
    }

    let mut out = Vec::with_capacity(cells * STOKES_COMPONENTS);
    for idx in 0..cells {
        let coeff_start = idx * COEFFS_PER_CELL;
        let init_start = if broadcast_initial {
            0
        } else {
            idx * STOKES_COMPONENTS
        };
        let stokes = [
            initial[init_start],
            initial[init_start + 1],
            initial[init_start + 2],
            initial[init_start + 3],
        ];
        let stepped = stokes_step_rk2(
            stokes,
            &coeffs[coeff_start..coeff_start + COEFFS_PER_CELL],
            ds_cm,
        );
        out.extend_from_slice(&stepped);
    }
    Ok(out)
}

#[pyfunction]
fn sample_brick_trilinear(
    coeffs: Vec<f64>,
    r_grid: Vec<f64>,
    theta_grid: Vec<f64>,
    phi_grid: Vec<f64>,
    points: Vec<f64>,
) -> PyResult<Vec<f64>> {
    let point_count = validate_sampler_inputs(&coeffs, &r_grid, &theta_grid, &phi_grid, &points)?;
    let mut out = Vec::with_capacity(point_count * COEFFS_PER_CELL);
    for idx in 0..point_count {
        let point_start = idx * 3;
        let sampled = sample_coefficients(
            &coeffs,
            &r_grid,
            &theta_grid,
            &phi_grid,
            points[point_start],
            points[point_start + 1],
            points[point_start + 2],
        );
        out.extend_from_slice(&sampled);
    }
    Ok(out)
}

#[pyfunction]
fn sample_and_step_stokes(
    coeffs: Vec<f64>,
    r_grid: Vec<f64>,
    theta_grid: Vec<f64>,
    phi_grid: Vec<f64>,
    points: Vec<f64>,
    ds_cm: f64,
    initial: Vec<f64>,
) -> PyResult<Vec<f64>> {
    if !ds_cm.is_finite() || ds_cm < 0.0 {
        return Err(PyValueError::new_err(
            "ds_cm must be finite and non-negative",
        ));
    }
    let point_count = validate_sampler_inputs(&coeffs, &r_grid, &theta_grid, &phi_grid, &points)?;
    let broadcast_initial = initial.len() == STOKES_COMPONENTS;
    if !broadcast_initial && initial.len() != point_count * STOKES_COMPONENTS {
        return Err(PyValueError::new_err(
            "initial must contain either 4 values or 4 values per sample point",
        ));
    }
    if initial.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err(
            "initial must contain only finite values",
        ));
    }

    let mut out = Vec::with_capacity(point_count * STOKES_COMPONENTS);
    for idx in 0..point_count {
        let point_start = idx * 3;
        let coeff = sample_coefficients(
            &coeffs,
            &r_grid,
            &theta_grid,
            &phi_grid,
            points[point_start],
            points[point_start + 1],
            points[point_start + 2],
        );
        let init_start = if broadcast_initial {
            0
        } else {
            idx * STOKES_COMPONENTS
        };
        let stokes = [
            initial[init_start],
            initial[init_start + 1],
            initial[init_start + 2],
            initial[init_start + 3],
        ];
        let stepped = stokes_step_rk2(stokes, &coeff, ds_cm);
        out.extend_from_slice(&stepped);
    }
    Ok(out)
}

#[pymodule]
fn blackhole_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(core_version, m)?)?;
    m.add_function(wrap_pyfunction!(detect_arch, m)?)?;
    m.add_function(wrap_pyfunction!(stokes_rk2_brick, m)?)?;
    m.add_function(wrap_pyfunction!(sample_brick_trilinear, m)?)?;
    m.add_function(wrap_pyfunction!(sample_and_step_stokes, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        normalized_arch, sample_and_step_stokes, sample_brick_trilinear, stokes_rk2_brick,
    };

    fn linear_fixture_coeffs() -> Vec<f64> {
        let mut coeffs = Vec::new();
        for ir in 0..2 {
            for it in 0..2 {
                for ip in 0..2 {
                    let base = (ir as f64 * 100.0) + (it as f64 * 10.0) + ip as f64;
                    for c in 0..11 {
                        coeffs.push(base + c as f64 * 0.001);
                    }
                }
            }
        }
        coeffs
    }

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

    #[test]
    fn sample_brick_trilinear_interpolates_center() {
        let out = sample_brick_trilinear(
            linear_fixture_coeffs(),
            vec![0.0, 1.0],
            vec![0.0, 1.0],
            vec![0.0, 1.0],
            vec![0.5, 0.5, 0.5],
        )
        .unwrap();
        assert_eq!(out.len(), 11);
        assert!((out[0] - 55.5).abs() < 1.0e-12);
        assert!((out[10] - 55.51).abs() < 1.0e-12);
    }

    #[test]
    fn sample_brick_trilinear_returns_zero_outside_nonperiodic_domain() {
        let out = sample_brick_trilinear(
            linear_fixture_coeffs(),
            vec![0.0, 1.0],
            vec![0.0, 1.0],
            vec![0.0, 1.0],
            vec![-0.1, 0.5, 0.5, 0.5, 1.1, 0.5],
        )
        .unwrap();
        assert_eq!(out, vec![0.0; 22]);
    }

    #[test]
    fn sample_and_step_stokes_matches_zero_coeff_noop_for_outside_sample() {
        let out = sample_and_step_stokes(
            linear_fixture_coeffs(),
            vec![0.0, 1.0],
            vec![0.0, 1.0],
            vec![0.0, 1.0],
            vec![-0.1, 0.5, 0.5],
            0.1,
            vec![1.0, 2.0, 3.0, 4.0],
        )
        .unwrap();
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0]);
    }
}
