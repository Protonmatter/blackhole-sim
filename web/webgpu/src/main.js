const canvas = document.getElementById('view');
const statusEl = document.getElementById('status');
const shaderSelect = document.getElementById('shader');
const playToggle = document.getElementById('playToggle');
const resetViewButton = document.getElementById('resetView');
const captureFrameButton = document.getElementById('captureFrame');
const qualityOut = document.getElementById('qualityOut');
const telemetry = {
  fps: document.getElementById('fpsValue'),
  resolution: document.getElementById('resolutionValue'),
  time: document.getElementById('timeValue'),
  horizon: document.getElementById('horizonValue'),
  isco: document.getElementById('iscoValue'),
  diagnostics: document.getElementById('diagnosticsValue'),
};
const sliders = {
  spin: document.getElementById('spin'),
  inc: document.getElementById('inc'),
  camr: document.getElementById('camr'),
  steps: document.getElementById('steps'),
  density: document.getElementById('density'),
  temp: document.getElementById('temp'),
  absorb: document.getElementById('absorb'),
  timeScale: document.getElementById('timeScale'),
};
const params = new URLSearchParams(location.search);
const diagnosticsEnabled = params.get('diagnostics') === '1';
const DEFAULTS = Object.freeze({
  shader: 'disk',
  quality: 'ultra',
  spin: 0.75,
  inc: 80,
  camr: 18,
  steps: 520,
  density: 1,
  temp: 1,
  absorb: 1,
  timeScale: 1,
});
const QUALITY = Object.freeze({
  fast: { steps: 180, scale: 0.62 },
  balanced: { steps: 340, scale: 0.92 },
  detail: { steps: 520, scale: 1.25 },
  ultra: { steps: 520, scale: 1.9 },
});
const PRESETS = Object.freeze({
  near: { shader: 'disk', quality: 'ultra', spin: 0.75, inc: 80, camr: 18, steps: 520, density: 1, temp: 1, absorb: 1, timeScale: 1 },
  flow: { shader: 'volume', quality: 'detail', spin: 0.62, inc: 80, camr: 18, steps: 520, density: 2.25, temp: 1.35, absorb: 0.55, timeScale: 1.45 },
  polarized: { shader: 'stokes', quality: 'balanced', spin: 0.75, inc: 80, camr: 18, steps: 340, density: 1.85, temp: 1.2, absorb: 0.75, timeScale: 1.2 },
  wide: { shader: 'disk', quality: 'detail', spin: 0.55, inc: 58, camr: 45, steps: 520, density: 0.7, temp: 0.8, absorb: 1, timeScale: 0.75 },
});
const simulationState = {
  running: params.get('running') !== '0',
  quality: params.get('quality') && QUALITY[params.get('quality')] ? params.get('quality') : DEFAULTS.quality,
  timeSeconds: Number.parseFloat(params.get('time') || '0') || 0,
  lastTimestampMs: null,
  frameMs: 0,
  telemetryAtMs: 0,
  interactionVersion: 0,
};

let device = null;
let context = null;
let format = null;
let webgpuReady = false;
let adapterInfoText = '';

function clampNumber(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function numericValue(input) {
  return Number.parseFloat(input.value);
}

function setRange(input, value) {
  const min = Number.parseFloat(input.min);
  const max = Number.parseFloat(input.max);
  const step = Number.parseFloat(input.step || '1');
  const snapped = Math.round(clampNumber(value, min, max) / step) * step;
  input.value = String(clampNumber(Number(snapped.toFixed(4)), min, max));
  syncOutput(input);
  simulationState.interactionVersion += 1;
}

function syncOutput(input) {
  const out = input.parentElement ? input.parentElement.querySelector('output') : null;
  if (out) out.value = input.value;
}

function setPlayState(running) {
  simulationState.running = running;
  playToggle.textContent = running ? 'Pause' : 'Run';
  playToggle.setAttribute('aria-pressed', running ? 'true' : 'false');
}

function setQuality(name, updateSteps = false) {
  simulationState.quality = QUALITY[name] ? name : DEFAULTS.quality;
  for (const button of document.querySelectorAll('[data-quality]')) {
    button.classList.toggle('active', button.dataset.quality === simulationState.quality);
    button.setAttribute('aria-pressed', button.dataset.quality === simulationState.quality ? 'true' : 'false');
  }
  qualityOut.value = simulationState.quality;
  if (updateSteps) setRange(sliders.steps, QUALITY[simulationState.quality].steps);
  resize();
}

function applyControlState(state) {
  for (const [key, input] of Object.entries(sliders)) {
    if (state[key] !== undefined) setRange(input, state[key]);
  }
  if (state.quality) setQuality(state.quality, false);
}

function currentUrlWithState(shader = shaderSelect.value) {
  const next = new URL(location.href);
  next.searchParams.set('shader', shader);
  next.searchParams.set('quality', simulationState.quality);
  next.searchParams.set('running', simulationState.running ? '1' : '0');
  for (const [key, input] of Object.entries(sliders)) next.searchParams.set(key, input.value);
  next.searchParams.set('time', simulationState.timeSeconds.toFixed(3));
  if (diagnosticsEnabled) next.searchParams.set('diagnostics', '1');
  return next;
}

function restoreStateFromUrl() {
  shaderSelect.value = params.get('shader') === 'volume' ? 'volume' : (params.get('shader') === 'stokes' ? 'stokes' : DEFAULTS.shader);
  for (const [key, input] of Object.entries(sliders)) {
    const raw = params.get(key);
    setRange(input, raw === null ? DEFAULTS[key] : Number.parseFloat(raw));
  }
  setQuality(simulationState.quality, false);
  setPlayState(simulationState.running);
}

async function adapterInfoLabel(adapter) {
  let info = ('info' in adapter && adapter.info) ? adapter.info : null;
  if ((!info || Object.keys(info).length === 0) && typeof adapter.requestAdapterInfo === 'function') {
    info = await adapter.requestAdapterInfo().catch(() => null);
  }
  const fields = info ? [info.vendor, info.architecture, info.device, info.description].filter(Boolean) : [];
  return fields.length ? fields.join(' / ') : 'hardware adapter';
}

async function fetchShaderText(path) {
  const response = await fetch(path, { cache: 'no-store' });
  if (!response.ok) throw new Error(`failed to load ${path}: ${response.status}`);
  return response.text();
}

function setGpuStatus(mode) {
  statusEl.textContent = adapterInfoText ? `${mode} on ${adapterInfoText}` : mode;
}

function activeRenderScale() {
  const modeBase = shaderSelect.value === 'volume' ? 0.36 : (shaderSelect.value === 'disk' ? 0.58 : 0.32);
  return Math.min(1.15, modeBase * QUALITY[simulationState.quality].scale);
}

function resize() {
  const dpr = Math.min(window.devicePixelRatio || 1, 1);
  const scale = activeRenderScale();
  const w = Math.max(1, Math.floor(canvas.clientWidth * dpr * scale));
  const h = Math.max(1, Math.floor(canvas.clientHeight * dpr * scale));
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
    return true;
  }
  return false;
}

function progressiveLevels(width, height, minWidth = 480) {
  const out = [];
  let scale = 1;
  while (Math.floor(width / (scale * 2)) >= minWidth) scale *= 2;
  let level = 0;
  while (scale >= 1) {
    out.push({ level, width: Math.max(1, Math.floor(width / scale)), height: Math.max(1, Math.floor(height / scale)) });
    scale = Math.floor(scale / 2);
    level += 1;
  }
  return out;
}

function horizonRadius(a0) {
  const a = clampNumber(a0, -0.98, 0.98);
  return 1 + Math.sqrt(Math.max(0, 1 - a * a));
}

function iscoRadius(a0) {
  const a = Math.abs(clampNumber(a0, -0.98, 0.98));
  const z1 = 1 + Math.pow(1 - a * a, 1 / 3) * (Math.pow(1 + a, 1 / 3) + Math.pow(1 - a, 1 / 3));
  const z2 = Math.sqrt(3 * a * a + z1 * z1);
  return 3 + z2 - Math.sqrt((3 - z1) * (3 + z1 + 2 * z2));
}

function updateSimulationClock(timeMs) {
  if (simulationState.lastTimestampMs === null) simulationState.lastTimestampMs = timeMs;
  const dt = clampNumber((timeMs - simulationState.lastTimestampMs) * 0.001, 0, 0.12);
  simulationState.lastTimestampMs = timeMs;
  simulationState.frameMs = 0.85 * simulationState.frameMs + 0.15 * (dt * 1000);
  if (simulationState.running) simulationState.timeSeconds += dt * numericValue(sliders.timeScale);
}

function updateTelemetry(timeMs) {
  if (timeMs - simulationState.telemetryAtMs < 180) return;
  simulationState.telemetryAtMs = timeMs;
  const fps = simulationState.frameMs > 0 ? 1000 / simulationState.frameMs : 0;
  telemetry.fps.textContent = `${fps.toFixed(1)} fps`;
  telemetry.resolution.textContent = `${canvas.width} x ${canvas.height}`;
  telemetry.time.textContent = simulationState.timeSeconds.toFixed(2);
  telemetry.horizon.textContent = horizonRadius(numericValue(sliders.spin)).toFixed(3);
  telemetry.isco.textContent = iscoRadius(numericValue(sliders.spin)).toFixed(3);
  const diagnostics = window.blackholeWebgpuDiagnostics;
  telemetry.diagnostics.textContent = diagnostics && Number.isFinite(diagnostics.maxI) ? `I ${diagnostics.maxI.toExponential(2)}` : (diagnosticsEnabled ? 'pending' : 'off');
}

function writeStokesParams(buffer, width, height, level = 0) {
  const ab = new ArrayBuffer(64);
  const dv = new DataView(ab);
  const u32 = (i, v) => dv.setUint32(i * 4, v, true);
  const f32 = (i, v) => dv.setFloat32(i * 4, v, true);
  u32(0, width);
  u32(1, height);
  u32(2, 16);
  u32(3, 12);
  u32(4, 16);
  f32(5, numericValue(sliders.spin));
  u32(6, Number.parseInt(sliders.steps.value, 10));
  f32(7, 0.055);
  f32(8, numericValue(sliders.camr));
  f32(9, numericValue(sliders.inc) * Math.PI / 180);
  f32(10, 34 * Math.PI / 180);
  f32(11, 220);
  u32(12, level);
  u32(13, 0);
  u32(14, 0);
  u32(15, 0);
  device.queue.writeBuffer(buffer, 0, ab);
}

function writeFragmentParams(buffer, width, height) {
  const ab = new ArrayBuffer(64);
  const dv = new DataView(ab);
  const f32 = (i, v) => dv.setFloat32(i * 4, v, true);
  f32(0, width);
  f32(1, height);
  f32(2, numericValue(sliders.spin));
  f32(3, numericValue(sliders.inc) * Math.PI / 180);
  f32(4, numericValue(sliders.camr));
  f32(5, 34 * Math.PI / 180);
  f32(6, numericValue(sliders.steps));
  f32(7, 0.055);
  f32(8, simulationState.timeSeconds);
  f32(9, 70);
  f32(10, shaderSelect.value === 'volume' ? 4.2 : 8.0);
  f32(11, 2.2);
  f32(12, numericValue(sliders.density));
  f32(13, numericValue(sliders.temp));
  f32(14, numericValue(sliders.absorb));
  f32(15, 0);
  device.queue.writeBuffer(buffer, 0, ab);
}

function makeUniformBuffer() {
  return device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
}

function makeSyntheticBricks(nr = 16, nt = 12, np = 16, phase = 0) {
  const r = new Float32Array(nr);
  const th = new Float32Array(nt);
  const ph = new Float32Array(np);
  const coeffs = new Float32Array(nr * nt * np * 11);
  const density = numericValue(sliders.density);
  const temp = numericValue(sliders.temp);
  const absorb = numericValue(sliders.absorb);
  for (let i = 0; i < nr; i += 1) r[i] = 2.4 + i * (58 / (nr - 1));
  for (let j = 0; j < nt; j += 1) th[j] = 0.045 + j * ((Math.PI - 0.09) / (nt - 1));
  for (let k = 0; k < np; k += 1) ph[k] = k * (2 * Math.PI / np);
  for (let i = 0; i < nr; i += 1) {
    for (let j = 0; j < nt; j += 1) {
      for (let k = 0; k < np; k += 1) {
        const rr = r[i];
        const tt = th[j];
        const pp = ph[k] - phase;
        const vert = (tt - Math.PI / 2) / 0.34;
        const rad = Math.log(rr / 12) / 0.48;
        const rho = Math.exp(-0.5 * (vert * vert + rad * rad)) * (1 + 0.12 * Math.sin(2 * pp + 0.5 * Math.log(rr)));
        const base = rho * density * Math.max(0.05, temp) * 8e-2;
        const idx = (((i * nt + j) * np + k) * 11);
        coeffs[idx + 0] = base;
        coeffs[idx + 1] = 0.22 * base * Math.cos(2 * pp);
        coeffs[idx + 2] = 0.22 * base * Math.sin(2 * pp);
        coeffs[idx + 3] = 0.02 * base * Math.sin(pp);
        coeffs[idx + 4] = absorb * base * 0.025;
        coeffs[idx + 5] = 0.002 * base;
        coeffs[idx + 6] = 0;
        coeffs[idx + 7] = 0;
        coeffs[idx + 8] = 0.001 * base;
        coeffs[idx + 9] = 0.0003 * base;
        coeffs[idx + 10] = 0.00015 * base;
      }
    }
  }
  return { r, th, ph, coeffs };
}

function storageBuffer(data, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST) {
  const b = device.createBuffer({ size: Math.max(4, data.byteLength), usage, mappedAtCreation: true });
  new data.constructor(b.getMappedRange()).set(data);
  b.unmap();
  return b;
}

function showWebGPUFallback(message) {
  statusEl.textContent = message;
  const draw = () => {
    resize();
    const fallback = canvas.getContext('2d');
    if (!fallback) return;
    fallback.fillStyle = '#050508';
    fallback.fillRect(0, 0, canvas.width, canvas.height);
    fallback.fillStyle = '#dce6f2';
    fallback.font = `${Math.max(14, Math.floor(canvas.width / 42))}px system-ui, sans-serif`;
    fallback.fillText(message, 24, 44);
    fallback.fillStyle = '#9fb0c2';
    fallback.font = `${Math.max(12, Math.floor(canvas.width / 58))}px system-ui, sans-serif`;
    fallback.fillText('Use a WebGPU-capable browser over localhost for the GPU renderer.', 24, 76);
  };
  window.addEventListener('resize', draw);
  draw();
}

async function runStokesRenderer() {
  setGpuStatus('WebGPU direct compute: Stokes coefficient bricks');
  const computeCode = await fetchShaderText('./src/stokes_brick_compute.wgsl');
  const displayCode = await fetchShaderText('./src/display_stokes.wgsl');
  const computeModule = device.createShaderModule({ code: computeCode });
  const displayModule = device.createShaderModule({ code: displayCode });
  const bricks = makeSyntheticBricks();
  const coeffBuf = storageBuffer(bricks.coeffs);
  const rBuf = storageBuffer(bricks.r);
  const thBuf = storageBuffer(bricks.th);
  const phBuf = storageBuffer(bricks.ph);
  let stokesBuffer = null;
  let uniformBuffer = makeUniformBuffer();
  let computePipeline = null;
  let computeBindGroup = null;
  let displayPipeline = null;
  let displayBindGroup = null;
  let readbackStarted = false;
  let brickKey = '';

  function refreshBricks() {
    const phaseBin = Math.floor(simulationState.timeSeconds * 10);
    const nextKey = `${sliders.density.value}:${sliders.temp.value}:${sliders.absorb.value}:${phaseBin}`;
    if (nextKey === brickKey) return;
    brickKey = nextKey;
    const updated = makeSyntheticBricks(16, 12, 16, phaseBin * 0.055);
    device.queue.writeBuffer(coeffBuf, 0, updated.coeffs);
  }

  function rebuild(width, height) {
    stokesBuffer = device.createBuffer({ size: width * height * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const cLayout = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ] });
    computePipeline = device.createComputePipeline({ layout: device.createPipelineLayout({ bindGroupLayouts: [cLayout] }), compute: { module: computeModule, entryPoint: 'main' } });
    computeBindGroup = device.createBindGroup({ layout: cLayout, entries: [
      { binding: 0, resource: { buffer: stokesBuffer } },
      { binding: 1, resource: { buffer: coeffBuf } },
      { binding: 2, resource: { buffer: rBuf } },
      { binding: 3, resource: { buffer: thBuf } },
      { binding: 4, resource: { buffer: phBuf } },
      { binding: 5, resource: { buffer: uniformBuffer } },
    ] });
    const dLayout = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
    ] });
    displayPipeline = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [dLayout] }),
      vertex: { module: displayModule, entryPoint: 'vs_main' },
      fragment: { module: displayModule, entryPoint: 'fs_main', targets: [{ format }] },
      primitive: { topology: 'triangle-list' },
    });
    displayBindGroup = device.createBindGroup({ layout: dLayout, entries: [
      { binding: 0, resource: { buffer: stokesBuffer } },
      { binding: 1, resource: { buffer: uniformBuffer } },
    ] });
  }

  function queueReadback(encoder, width, height) {
    if (!diagnosticsEnabled || readbackStarted) return null;
    readbackStarted = true;
    const samplePixels = Math.min(width * height, 4096);
    const size = samplePixels * 16;
    const buffer = device.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    encoder.copyBufferToBuffer(stokesBuffer, 0, buffer, 0, size);
    return { buffer, width, height, samplePixels };
  }

  function finishReadback(job) {
    if (!job) return;
    job.buffer.mapAsync(GPUMapMode.READ).then(() => {
      const values = new Float32Array(job.buffer.getMappedRange());
      let finite = 0;
      let maxAbs = 0;
      let maxI = 0;
      for (let i = 0; i < values.length; i += 4) {
        const I = values[i];
        const q = values[i + 1];
        const u = values[i + 2];
        const v = values[i + 3];
        if (Number.isFinite(I) && Number.isFinite(q) && Number.isFinite(u) && Number.isFinite(v)) finite += 1;
        maxI = Math.max(maxI, Math.abs(I));
        maxAbs = Math.max(maxAbs, Math.abs(I), Math.abs(q), Math.abs(u), Math.abs(v));
      }
      const diagnostics = { width: job.width, height: job.height, samplePixels: job.samplePixels, finitePixels: finite, maxI, maxAbs };
      window.blackholeWebgpuDiagnostics = diagnostics;
      job.buffer.unmap();
      job.buffer.destroy();
      setGpuStatus(`WebGPU direct compute: Stokes coefficient bricks; readback maxI=${maxI.toExponential(2)}`);
    }).catch((err) => {
      window.blackholeWebgpuDiagnostics = { error: String(err && err.message ? err.message : err) };
      setGpuStatus('WebGPU direct compute: Stokes coefficient bricks; readback failed');
    });
  }

  function frame(timeMs) {
    updateSimulationClock(timeMs);
    resize();
    refreshBricks();
    const levels = progressiveLevels(canvas.width, canvas.height, 480);
    const lvl = levels[Math.min(Math.floor(timeMs / 850) % levels.length, levels.length - 1)];
    if (!stokesBuffer || lvl.width * lvl.height * 16 > stokesBuffer.size) rebuild(lvl.width, lvl.height);
    writeStokesParams(uniformBuffer, lvl.width, lvl.height, lvl.level);
    const encoder = device.createCommandEncoder();
    const cp = encoder.beginComputePass();
    cp.setPipeline(computePipeline);
    cp.setBindGroup(0, computeBindGroup);
    cp.dispatchWorkgroups(Math.ceil(lvl.width / 8), Math.ceil(lvl.height / 8));
    cp.end();
    const readbackJob = queueReadback(encoder, lvl.width, lvl.height);
    const rp = encoder.beginRenderPass({ colorAttachments: [{ view: context.getCurrentTexture().createView(), clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: 'clear', storeOp: 'store' }] });
    rp.setPipeline(displayPipeline);
    rp.setBindGroup(0, displayBindGroup);
    rp.draw(3);
    rp.end();
    device.queue.submit([encoder.finish()]);
    finishReadback(readbackJob);
    updateTelemetry(timeMs);
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

async function runFragmentRenderer() {
  const shaderPath = shaderSelect.value === 'volume' ? './src/grrt_volume.wgsl' : './src/kerr_raytrace.wgsl';
  const shaderCode = await fetchShaderText(shaderPath);
  setGpuStatus(shaderSelect.value === 'volume' ? 'WebGPU fragment shader: GRRT volume' : 'WebGPU fragment shader: thin disk');
  const shader = device.createShaderModule({ code: shaderCode });
  const uniformBuffer = makeUniformBuffer();
  const bindGroupLayout = device.createBindGroupLayout({ entries: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }] });
  const pipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    vertex: { module: shader, entryPoint: 'vs_main' },
    fragment: { module: shader, entryPoint: 'fs_main', targets: [{ format }] },
    primitive: { topology: 'triangle-list' },
  });
  const bindGroup = device.createBindGroup({ layout: bindGroupLayout, entries: [{ binding: 0, resource: { buffer: uniformBuffer } }] });
  function frame(timeMs) {
    updateSimulationClock(timeMs);
    resize();
    writeFragmentParams(uniformBuffer, canvas.width, canvas.height);
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({ colorAttachments: [{ view: context.getCurrentTexture().createView(), clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: 'clear', storeOp: 'store' }] });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(3);
    pass.end();
    device.queue.submit([encoder.finish()]);
    updateTelemetry(timeMs);
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

function installUiHandlers() {
  shaderSelect.addEventListener('change', () => { location.href = currentUrlWithState(shaderSelect.value).toString(); });
  for (const input of Object.values(sliders)) {
    input.addEventListener('input', () => {
      syncOutput(input);
      simulationState.interactionVersion += 1;
    });
    syncOutput(input);
  }
  for (const button of document.querySelectorAll('[data-quality]')) {
    button.addEventListener('click', () => setQuality(button.dataset.quality, true));
  }
  for (const button of document.querySelectorAll('[data-preset]')) {
    button.addEventListener('click', () => {
      const preset = PRESETS[button.dataset.preset];
      applyControlState(preset);
      simulationState.timeSeconds = 0;
      if (preset.shader !== shaderSelect.value) location.href = currentUrlWithState(preset.shader).toString();
    });
  }
  playToggle.addEventListener('click', () => setPlayState(!simulationState.running));
  resetViewButton.addEventListener('click', () => {
    applyControlState(PRESETS.near);
    simulationState.timeSeconds = 0;
    if (shaderSelect.value !== PRESETS.near.shader) location.href = currentUrlWithState(PRESETS.near.shader).toString();
  });
  captureFrameButton.addEventListener('click', () => {
    canvas.toBlob((blob) => {
      if (!blob) return;
      const href = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = href;
      a.download = `blackhole-${shaderSelect.value}-${Date.now()}.png`;
      a.click();
      URL.revokeObjectURL(href);
    });
  });

  let drag = null;
  canvas.addEventListener('pointerdown', (ev) => {
    drag = { id: ev.pointerId, x: ev.clientX, y: ev.clientY, spin: numericValue(sliders.spin), inc: numericValue(sliders.inc) };
    canvas.setPointerCapture(ev.pointerId);
  });
  canvas.addEventListener('pointermove', (ev) => {
    if (!drag || drag.id !== ev.pointerId) return;
    setRange(sliders.spin, drag.spin + (ev.clientX - drag.x) * 0.0035);
    setRange(sliders.inc, drag.inc + (ev.clientY - drag.y) * 0.12);
  });
  canvas.addEventListener('pointerup', () => { drag = null; });
  canvas.addEventListener('pointercancel', () => { drag = null; });
  canvas.addEventListener('wheel', (ev) => {
    ev.preventDefault();
    setRange(sliders.camr, numericValue(sliders.camr) + ev.deltaY * 0.035);
  }, { passive: false });
  window.addEventListener('keydown', (ev) => {
    if (ev.target && ['INPUT', 'SELECT', 'BUTTON'].includes(ev.target.tagName)) return;
    if (ev.key === ' ') {
      ev.preventDefault();
      setPlayState(!simulationState.running);
    } else if (ev.key === 'r' || ev.key === 'R') {
      applyControlState(PRESETS.near);
      simulationState.timeSeconds = 0;
    }
  });
  window.addEventListener('resize', resize);
}

async function initializeWebGPU() {
  if (!navigator.gpu) {
    showWebGPUFallback('WebGPU unavailable');
    return;
  }
  const adapter = await navigator.gpu.requestAdapter().catch(() => null);
  if (!adapter) {
    showWebGPUFallback('WebGPU adapter unavailable');
    return;
  }
  adapterInfoText = await adapterInfoLabel(adapter);
  device = await adapter.requestDevice().catch(() => null);
  if (!device) {
    showWebGPUFallback('WebGPU device request failed');
    return;
  }
  context = canvas.getContext('webgpu');
  if (!context) {
    showWebGPUFallback('WebGPU canvas context unavailable');
    return;
  }
  format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: 'opaque' });
  webgpuReady = true;
}

restoreStateFromUrl();
installUiHandlers();
resize();
await initializeWebGPU();
updateTelemetry(0);
if (webgpuReady) {
  if (shaderSelect.value === 'stokes') runStokesRenderer();
  else runFragmentRenderer();
}
