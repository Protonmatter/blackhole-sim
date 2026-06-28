struct RenderParams {
  width: u32, height: u32, nr: u32, ntheta: u32, nphi: u32,
  spin_a: f32, max_steps: u32, step: f32,
  camera_r: f32, camera_theta: f32, fov_y: f32, escape_radius: f32,
  level: u32, tile_x0: u32, tile_y0: u32, pad0: u32,
};
@group(0) @binding(0) var<storage, read> stokes: array<vec4<f32>>;
@group(0) @binding(1) var<uniform> params: RenderParams;

struct VSOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };
@vertex fn vs_main(@builtin(vertex_index) i: u32) -> VSOut {
  var p = array<vec2<f32>,3>(vec2<f32>(-1.0,-1.0), vec2<f32>(3.0,-1.0), vec2<f32>(-1.0,3.0));
  var o: VSOut; o.pos = vec4<f32>(p[i],0.0,1.0); o.uv = 0.5 * (p[i] + vec2<f32>(1.0)); return o;
}
@fragment fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
  let x = min(u32(in.uv.x * f32(params.width)), params.width - 1u);
  let y = min(u32((1.0 - in.uv.y) * f32(params.height)), params.height - 1u);
  let S = stokes[y * params.width + x];
  let I = max(S.x, 0.0); let lin = sqrt(S.y*S.y + S.z*S.z);
  var rgb = vec3<f32>(I + 0.35*max(S.y,0.0) + 0.15*max(S.w,0.0), I + 0.25*lin, I + 0.35*max(-S.y,0.0) + 0.15*max(-S.w,0.0));
  rgb = rgb / max(max(rgb.x, max(rgb.y, rgb.z)), 1.0e-20);
  rgb = rgb / (vec3<f32>(1.0) + rgb);
  return vec4<f32>(pow(clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1.0/2.2)), 1.0);
}
