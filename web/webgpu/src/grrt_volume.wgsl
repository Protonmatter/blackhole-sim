struct Uniforms {
  width: f32,
  height: f32,
  spin: f32,
  inclination: f32,
  camera_r: f32,
  fov_y: f32,
  steps_f: f32,
  step_h: f32,
  time: f32,
  disk_outer: f32,
  exposure: f32,
  gamma: f32,
  pad0: f32,
  pad1: f32,
  pad2: f32,
  pad3: f32,
};
@group(0) @binding(0) var<uniform> U: Uniforms;

struct VSOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VSOut {
  var p = array<vec2<f32>,3>(vec2<f32>(-1.0,-3.0), vec2<f32>(3.0,1.0), vec2<f32>(-1.0,1.0));
  var out: VSOut;
  out.pos = vec4<f32>(p[vi], 0.0, 1.0);
  out.uv = p[vi] * 0.5 + vec2<f32>(0.5);
  return out;
}

fn safe_theta(th: f32) -> f32 { return clamp(th, 1e-4, 3.1415926 - 1e-4); }
fn delta(r: f32, a: f32) -> f32 { return r*r - 2.0*r + a*a; }
fn sigma(r: f32, th: f32, a: f32) -> f32 { return r*r + a*a*cos(th)*cos(th); }
fn bigA(r: f32, th: f32, a: f32) -> f32 {
  let s2 = sin(th)*sin(th);
  return (r*r+a*a)*(r*r+a*a) - a*a*delta(r,a)*s2;
}
fn horizon(a: f32) -> f32 { return 1.0 + sqrt(max(0.0, 1.0-a*a)); }

struct MetricInv { gtt:f32, gtph:f32, grr:f32, gthth:f32, gphph:f32 };
fn metricInv(r: f32, th0: f32, a: f32) -> MetricInv {
  let th = safe_theta(th0);
  let sig = sigma(r, th, a);
  let d = delta(r, a);
  let s2 = max(sin(th)*sin(th), 1e-8);
  let A = bigA(r, th, a);
  return MetricInv(-A/(sig*d), -2.0*a*r/(sig*d), d/sig, 1.0/sig, (d-a*a*s2)/(sig*d*s2));
}

struct MetricCov { gtt:f32, gtph:f32, grr:f32, gthth:f32, gphph:f32 };
fn metricCov(r: f32, th0: f32, a: f32) -> MetricCov {
  let th = safe_theta(th0);
  let sig = sigma(r, th, a);
  let d = delta(r,a);
  let s2 = sin(th)*sin(th);
  let A = bigA(r, th, a);
  return MetricCov(-(1.0-2.0*r/sig), -2.0*a*r*s2/sig, sig/d, sig, A*s2/sig);
}

fn ham(r: f32, th: f32, pr: f32, pth: f32, pt: f32, pph: f32, a: f32) -> f32 {
  let g = metricInv(r, th, a);
  return 0.5*(g.gtt*pt*pt + 2.0*g.gtph*pt*pph + g.grr*pr*pr + g.gthth*pth*pth + g.gphph*pph*pph);
}

struct State { t:f32, r:f32, th:f32, ph:f32, pr:f32, pth:f32 };
struct Deriv { t:f32, r:f32, th:f32, ph:f32, pr:f32, pth:f32 };

fn rhs(y: State, pt: f32, pph: f32, a: f32) -> Deriv {
  let g = metricInv(y.r, y.th, a);
  let tdot = g.gtt*pt + g.gtph*pph;
  let phdot = g.gtph*pt + g.gphph*pph;
  let rdot = g.grr*y.pr;
  let thdot = g.gthth*y.pth;
  let hr = max(1e-4, abs(y.r)*1e-4);
  let ht = 1e-4;
  let dHdr = (ham(y.r+hr,y.th,y.pr,y.pth,pt,pph,a)-ham(y.r-hr,y.th,y.pr,y.pth,pt,pph,a))/(2.0*hr);
  let dHdth = (ham(y.r,y.th+ht,y.pr,y.pth,pt,pph,a)-ham(y.r,y.th-ht,y.pr,y.pth,pt,pph,a))/(2.0*ht);
  return Deriv(tdot, rdot, thdot, phdot, -dHdr, -dHdth);
}
fn add(y: State, k: Deriv, h: f32) -> State { return State(y.t+h*k.t, y.r+h*k.r, safe_theta(y.th+h*k.th), y.ph+h*k.ph, y.pr+h*k.pr, y.pth+h*k.pth); }
fn rk4(y: State, h: f32, pt: f32, pph: f32, a: f32) -> State {
  let k1 = rhs(y, pt, pph, a);
  let k2 = rhs(add(y,k1,0.5*h), pt, pph, a);
  let k3 = rhs(add(y,k2,0.5*h), pt, pph, a);
  let k4 = rhs(add(y,k3,h), pt, pph, a);
  return State(
    y.t+h*(k1.t+2.0*k2.t+2.0*k3.t+k4.t)/6.0,
    y.r+h*(k1.r+2.0*k2.r+2.0*k3.r+k4.r)/6.0,
    safe_theta(y.th+h*(k1.th+2.0*k2.th+2.0*k3.th+k4.th)/6.0),
    y.ph+h*(k1.ph+2.0*k2.ph+2.0*k3.ph+k4.ph)/6.0,
    y.pr+h*(k1.pr+2.0*k2.pr+2.0*k3.pr+k4.pr)/6.0,
    y.pth+h*(k1.pth+2.0*k2.pth+2.0*k3.pth+k4.pth)/6.0
  );
}

fn isco(a0: f32) -> f32 {
  let a = abs(a0);
  let z1 = 1.0 + pow(1.0-a*a, 1.0/3.0)*(pow(1.0+a,1.0/3.0)+pow(1.0-a,1.0/3.0));
  let z2 = sqrt(3.0*a*a+z1*z1);
  return 3.0+z2-sqrt((3.0-z1)*(3.0+z1+2.0*z2));
}
fn omegaK(r: f32, a: f32) -> f32 { return 1.0/(pow(r, 1.5)+a); }
fn redshift(r: f32, pt: f32, pph: f32, a: f32) -> f32 {
  let g = metricCov(r, 1.5707963, a);
  let om = omegaK(r,a);
  let norm = -(g.gtt + 2.0*om*g.gtph + om*om*g.gphph);
  if (norm <= 0.0) { return 0.0; }
  let ut = inverseSqrt(norm);
  let uph = om*ut;
  let denom = -(pt*ut + pph*uph);
  if (denom <= 0.0) { return 0.0; }
  return 1.0/denom;
}

fn initRay(ndc: vec2<f32>) -> vec4<f32> {
  let a = U.spin;
  let r = U.camera_r;
  let th = safe_theta(U.inclination);
  let aspect = U.width/U.height;
  let tanY = tan(0.5*U.fov_y);
  let n = normalize(vec3<f32>(-1.0, -ndc.y*tanY, ndc.x*aspect*tanY)); // r, theta, phi local
  let sig = sigma(r,th,a);
  let d = delta(r,a);
  let A = bigA(r,th,a);
  let gc = metricCov(r, th, a);
  let lapse = sqrt(sig*d/A);
  let frame = 2.0*a*r/A;
  let et = vec4<f32>(1.0/lapse,0.0,0.0,frame/lapse);
  let er = vec4<f32>(0.0,sqrt(d/sig),0.0,0.0);
  let eth = vec4<f32>(0.0,0.0,1.0/sqrt(sig),0.0);
  let eph = vec4<f32>(0.0,0.0,0.0,1.0/sqrt(gc.gphph));
  let pcon = et + n.x*er + n.y*eth + n.z*eph;
  let pt = gc.gtt*pcon.x + gc.gtph*pcon.w;
  let pr = gc.grr*pcon.y;
  let pth = gc.gthth*pcon.z;
  let pph = gc.gtph*pcon.x + gc.gphph*pcon.w;
  return vec4<f32>(pt, pr, pth, pph);
}

fn diskColor(r: f32, g: f32, inner: f32) -> vec3<f32> {
  let radial = pow(max(r,inner)/inner, -2.6);
  let ring = exp(-0.5*pow((r-1.4*inner)/max(0.8,0.35*inner), 2.0));
  let taper = clamp((U.disk_outer-r)/max(1.0, 0.15*U.disk_outer), 0.0, 1.0);
  let I = radial*(0.35+ring)*taper*pow(max(g,0.0), 3.0);
  let mixv = clamp((g-0.55)/0.95, 0.0, 1.0);
  return I*mix(vec3<f32>(1.0,0.54,0.20), vec3<f32>(0.9,0.95,1.0), mixv);
}


fn torusDensity(r: f32, th: f32, ph: f32) -> f32 {
  let a = clamp(U.spin, -0.98, 0.98);
  let inner = isco(a);
  let r0 = max(8.0, 2.7*inner);
  let radial = log(max(r, 1e-3)/r0)/0.46;
  let vertical = (th-1.5707963)/0.32;
  let spiral = 1.0 + 0.08*sin(2.0*ph + 0.7*log(max(r,1e-3)) - 0.35*U.time);
  let turb = 1.0 + 0.035*sin(5.0*ph + 3.0*th + 0.17*r);
  return max(1e-8, exp(-0.5*(radial*radial + vertical*vertical))*spiral*turb);
}
fn thetaE(r: f32, th: f32) -> f32 {
  let inner = isco(clamp(U.spin, -0.98, 0.98));
  let vertical = (th-1.5707963)/0.32;
  return max(1e-3, 18.0*pow(max(r,inner)/inner, -0.84)*(0.82+0.18*exp(-0.5*vertical*vertical)));
}
fn angularU(r: f32, th: f32, a: f32) -> vec2<f32> {
  let g = metricCov(r, th, a);
  let om = omegaK(max(r, isco(a)), a);
  let norm = -(g.gtt + 2.0*om*g.gtph + om*om*g.gphph);
  if (norm <= 0.0) { return vec2<f32>(1.0, 0.0); }
  let ut = inverseSqrt(norm);
  return vec2<f32>(ut, om*ut);
}
fn volumeRedshift(r:f32, th:f32, pt:f32, pph:f32, a:f32) -> f32 {
  let u = angularU(r, th, a);
  let denom = -(pt*u.x + pph*u.y);
  if (denom <= 0.0) { return 0.0; }
  return 1.0/denom;
}
fn volumeColor(te: f32) -> vec3<f32> {
  let m = clamp(log(1.0+te)*0.18, 0.0, 1.0);
  return mix(vec3<f32>(1.0,0.48,0.16), vec3<f32>(0.78,0.88,1.0), m);
}
fn volumeCoeff(r:f32, th:f32, ph:f32, gshift:f32) -> vec4<f32> {
  if (r < horizon(clamp(U.spin,-0.98,0.98))*1.08 || r > U.disk_outer || th < 0.045 || th > 3.09659) { return vec4<f32>(0.0); }
  let rho = torusDensity(r, th, ph) * max(U.pad0, 0.05);
  let te = thetaE(r, th) * max(U.pad1, 0.05);
  let b = 0.08*sqrt(max(rho, 0.0))/max(r, 1.0);
  let nu = 230.0e9/max(gshift, 1e-6);
  let nuc = max(1.0, 2.8e10*b*te*te);
  let x = max(nu/nuc, 1e-12);
  let shape = pow(x, 1.0/3.0)*exp(-pow(x, 1.0/3.0));
  let j = 1.2*rho*b*te*te*shape;
  let alpha = max(U.pad2, 0.0)*0.08*j/max(te*nu*nu, 1e-20)*230.0e9*230.0e9;
  return vec4<f32>(volumeColor(te)*j, alpha);
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
  let ndc = vec2<f32>(2.0*in.uv.x-1.0, 1.0-2.0*in.uv.y);
  let a = clamp(U.spin, -0.98, 0.98);
  let p = initRay(ndc);
  let pt = p.x;
  let pph = p.w;
  var y = State(0.0, U.camera_r, safe_theta(U.inclination), 0.0, p.y, p.z);
  var col = vec3<f32>(0.0012,0.0014,0.0020)*(1.0+0.25*(1.0-dot(ndc,ndc)));
  var tau = 0.0;
  let nsteps = i32(U.steps_f);
  for (var i=0; i<1400; i=i+1) {
    if (i >= nsteps) { break; }
    let yn = rk4(y, U.step_h, pt, pph, a);
    if (yn.r <= horizon(a)*1.0002) { col = col*0.03; break; }
    if (yn.r > 220.0 && yn.r > y.r) { break; }
    let midr = 0.5*(y.r+yn.r);
    let midth = safe_theta(0.5*(y.th+yn.th));
    let midph = 0.5*(y.ph+yn.ph);
    let gg = volumeRedshift(midr, midth, pt, pph, a);
    if (gg > 1e-6) {
      let coeff = volumeCoeff(midr, midth, midph, gg);
      let dl = U.step_h*0.045/max(gg, 1e-4);
      let trans = exp(-min(tau, 18.0));
      col = col + trans*pow(gg,3.0)*coeff.rgb*dl;
      tau = tau + coeff.a*dl;
      if (tau > 18.0) { break; }
    }
    y = yn;
  }
  col = U.exposure*col;
  col = col/(vec3<f32>(1.0)+col);
  col = pow(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1.0/U.gamma));
  return vec4<f32>(col, 1.0);
}
