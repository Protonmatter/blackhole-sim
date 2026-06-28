// v0.7 WebGPU compute hot-loop for Kerr + coefficient-brick Stokes rendering.
// One invocation = one pixel. Coefficients are flattened [nr][ntheta][nphi][11].

struct RenderParams {
  width: u32,
  height: u32,
  nr: u32,
  ntheta: u32,
  nphi: u32,
  spin_a: f32,
  max_steps: u32,
  step: f32,
  camera_r: f32,
  camera_theta: f32,
  fov_y: f32,
  escape_radius: f32,
  level: u32,
  tile_x0: u32,
  tile_y0: u32,
  pad0: u32,
};
struct State6 { t: f32, r: f32, th: f32, ph: f32, pr: f32, pth: f32 };

@group(0) @binding(0) var<storage, read_write> outStokes: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> coeffs: array<f32>;
@group(0) @binding(2) var<storage, read> rGrid: array<f32>;
@group(0) @binding(3) var<storage, read> thetaGrid: array<f32>;
@group(0) @binding(4) var<storage, read> phiGrid: array<f32>;
@group(0) @binding(5) var<uniform> params: RenderParams;

fn clamp_theta(x: f32) -> f32 { return clamp(x, 1.0e-6, 3.1415926 - 1.0e-6); }
fn wrap_phi(x: f32) -> f32 { var y = x % 6.28318530718; if (y < 0.0) { y = y + 6.28318530718; } return y; }
fn cidx(ir: u32, it: u32, ip: u32, c: u32) -> u32 { return (((ir * params.ntheta) + it) * params.nphi + ip) * 11u + c; }

fn metric_contravariant(r: f32, theta: f32, a: f32) -> mat4x4<f32> {
  let ct = cos(theta);
  let st = sin(theta);
  let s2 = max(st * st, 1.0e-8);
  let sig = r * r + a * a * ct * ct;
  let dlt = r * r - 2.0 * r + a * a;
  let A = (r * r + a * a) * (r * r + a * a) - a * a * dlt * s2;
  var g = mat4x4<f32>();
  g[0][0] = -A / (sig * dlt);
  g[0][3] = -2.0 * a * r / (sig * dlt);
  g[3][0] = g[0][3];
  g[1][1] = dlt / sig;
  g[2][2] = 1.0 / sig;
  g[3][3] = (dlt - a * a * s2) / (sig * dlt * s2);
  return g;
}

fn metric_derivative_r(r: f32, th: f32, a: f32) -> mat4x4<f32> {
  let e = 1.0e-3;
  return (metric_contravariant(r + e, th, a) - metric_contravariant(r - e, th, a)) / (2.0 * e);
}
fn metric_derivative_theta(r: f32, th: f32, a: f32) -> mat4x4<f32> {
  let e = 1.0e-4;
  return (metric_contravariant(r, th + e, a) - metric_contravariant(r, th - e, a)) / (2.0 * e);
}

fn kerr_rhs(y: State6, p_t: f32, p_phi: f32, a: f32) -> State6 {
  let g = metric_contravariant(y.r, y.th, a);
  let dr = metric_derivative_r(y.r, y.th, a);
  let dth = metric_derivative_theta(y.r, y.th, a);
  let p = vec4<f32>(p_t, y.pr, y.pth, p_phi);
  let xd = g * p;
  var prdot = 0.0;
  var pthdot = 0.0;
  for (var mu: u32 = 0u; mu < 4u; mu = mu + 1u) {
    for (var nu: u32 = 0u; nu < 4u; nu = nu + 1u) {
      prdot = prdot - 0.5 * p[mu] * dr[mu][nu] * p[nu];
      pthdot = pthdot - 0.5 * p[mu] * dth[mu][nu] * p[nu];
    }
  }
  return State6(xd.x, xd.y, xd.z, xd.w, prdot, pthdot);
}

fn rk2_geodesic_step(y: State6, h: f32, p_t: f32, p_phi: f32, a: f32) -> State6 {
  let k1 = kerr_rhs(y, p_t, p_phi, a);
  let mid = State6(y.t + 0.5*h*k1.t, y.r + 0.5*h*k1.r, y.th + 0.5*h*k1.th, y.ph + 0.5*h*k1.ph, y.pr + 0.5*h*k1.pr, y.pth + 0.5*h*k1.pth);
  let k2 = kerr_rhs(mid, p_t, p_phi, a);
  return State6(y.t + h*k2.t, y.r + h*k2.r, clamp_theta(y.th + h*k2.th), y.ph + h*k2.ph, y.pr + h*k2.pr, y.pth + h*k2.pth);
}

fn bracket_linear(grid_kind: u32, n: u32, x: f32) -> vec3<f32> {
  // grid_kind: 0=r, 1=theta. return (valid, i0, w); i1=i0+1.
  let first = select(thetaGrid[0], rGrid[0], grid_kind == 0u);
  let last = select(thetaGrid[n-1u], rGrid[n-1u], grid_kind == 0u);
  if (x < first || x > last) { return vec3<f32>(0.0, 0.0, 0.0); }
  var lo: u32 = 0u;
  loop { if (lo >= n - 2u) { break; } let next = select(thetaGrid[lo+1u], rGrid[lo+1u], grid_kind == 0u); if (next > x) { break; } lo = lo + 1u; }
  let a = select(thetaGrid[lo], rGrid[lo], grid_kind == 0u);
  let b = select(thetaGrid[lo+1u], rGrid[lo+1u], grid_kind == 0u);
  return vec3<f32>(1.0, f32(lo), (x - a) / max(b - a, 1.0e-20));
}

fn bracket_phi(phi: f32) -> vec3<f32> {
  let p = wrap_phi(phi);
  var lo: u32 = 0u;
  loop { if (lo >= params.nphi - 1u) { break; } if (phiGrid[lo+1u] > p) { break; } lo = lo + 1u; }
  let i1 = (lo + 1u) % params.nphi;
  let hi = select(phiGrid[0] + 6.28318530718, phiGrid[i1], i1 > lo);
  let pp = select(p + 6.28318530718, p, p >= phiGrid[lo]);
  return vec3<f32>(f32(lo), f32(i1), (pp - phiGrid[lo]) / max(hi - phiGrid[lo], 1.0e-20));
}

fn coeff_at(ir: u32, it: u32, ip: u32, c: u32) -> f32 { return coeffs[cidx(ir, it, ip, c)]; }

fn sample_brick_trilinear(r: f32, th: f32, ph: f32) -> array<f32, 11> {
  var out: array<f32, 11>;
  let rb = bracket_linear(0u, params.nr, r);
  let tb = bracket_linear(1u, params.ntheta, th);
  if (rb.x < 0.5 || tb.x < 0.5) { for (var c: u32 = 0u; c < 11u; c = c + 1u) { out[c] = 0.0; } return out; }
  let r0 = u32(rb.y); let r1 = r0 + 1u; let wr = rb.z;
  let t0 = u32(tb.y); let t1 = t0 + 1u; let wt = tb.z;
  let pb = bracket_phi(ph); let p0 = u32(pb.x); let p1 = u32(pb.y); let wp = pb.z;
  for (var c: u32 = 0u; c < 11u; c = c + 1u) {
    let c00 = mix(coeff_at(r0,t0,p0,c), coeff_at(r0,t0,p1,c), wp);
    let c01 = mix(coeff_at(r0,t1,p0,c), coeff_at(r0,t1,p1,c), wp);
    let c10 = mix(coeff_at(r1,t0,p0,c), coeff_at(r1,t0,p1,c), wp);
    let c11 = mix(coeff_at(r1,t1,p0,c), coeff_at(r1,t1,p1,c), wp);
    out[c] = mix(mix(c00, c01, wt), mix(c10, c11, wt), wr);
  }
  return out;
}

fn stokes_rhs(S: vec4<f32>, c: array<f32, 11>) -> vec4<f32> {
  return vec4<f32>(
    c[0] - (c[4]*S.x + c[5]*S.y + c[6]*S.z + c[7]*S.w),
    c[1] - (c[5]*S.x + c[4]*S.y + c[8]*S.z - c[9]*S.w),
    c[2] - (c[6]*S.x - c[8]*S.y + c[4]*S.z + c[10]*S.w),
    c[3] - (c[7]*S.x + c[9]*S.y - c[10]*S.z + c[4]*S.w)
  );
}
fn stokes_step_rk2(S: vec4<f32>, c: array<f32, 11>, ds: f32) -> vec4<f32> {
  let k1 = stokes_rhs(S, c);
  let mid = S + 0.5 * ds * k1;
  let k2 = stokes_rhs(mid, c);
  return S + ds * k2;
}

fn ray_initial_state(px: u32, py: u32) -> State6 {
  let ndcx = 2.0 * ((f32(px) + 0.5) / f32(params.width)) - 1.0;
  let ndcy = 1.0 - 2.0 * ((f32(py) + 0.5) / f32(params.height));
  return State6(0.0, params.camera_r, params.camera_theta, 0.0, -1.0, -0.22 * ndcy);
}

@compute @workgroup_size(8, 8, 1)
fn kerr_stokes_render_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.width || gid.y >= params.height) { return; }
  let pix = gid.y * params.width + gid.x;
  let ndcx = 2.0 * ((f32(gid.x) + 0.5) / f32(params.width)) - 1.0;
  var y = ray_initial_state(gid.x, gid.y);
  let p_t = -1.0;
  let p_phi = 0.12 * ndcx;
  let r_plus = 1.0 + sqrt(max(1.0 - params.spin_a * params.spin_a, 0.0));
  var S = vec4<f32>(0.0);
  for (var n: u32 = 0u; n < params.max_steps; n = n + 1u) {
    let prev = y;
    y = rk2_geodesic_step(y, params.step, p_t, p_phi, params.spin_a);
    let c = sample_brick_trilinear(0.5 * (prev.r + y.r), 0.5 * (prev.th + y.th), 0.5 * (prev.ph + y.ph));
    let ds = length(vec3<f32>(y.r - prev.r, y.th - prev.th, y.ph - prev.ph));
    S = stokes_step_rk2(S, c, ds);
    if (y.r <= r_plus * 1.0002 || y.r > params.escape_radius) { break; }
  }
  outStokes[pix] = S;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) { kerr_stokes_render_kernel(gid); }
