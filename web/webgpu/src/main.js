const canvas = document.getElementById('view');
const statusEl = document.getElementById('status');
const sliders = {
  spin: document.getElementById('spin'), inc: document.getElementById('inc'), camr: document.getElementById('camr'),
  steps: document.getElementById('steps'), density: document.getElementById('density'), temp: document.getElementById('temp'), absorb: document.getElementById('absorb'),
};
const shaderSelect = document.getElementById('shader');
const params = new URLSearchParams(location.search);
shaderSelect.value = params.get('shader') === 'volume' ? 'volume' : (params.get('shader') === 'stokes' ? 'stokes' : 'disk');
shaderSelect.addEventListener('change', () => { const next = new URL(location.href); next.searchParams.set('shader', shaderSelect.value); location.href = next.toString(); });
for (const input of Object.values(sliders)) { const out = input.parentElement.querySelector('output'); const sync = () => { out.value = input.value; }; input.addEventListener('input', sync); sync(); }
if (!navigator.gpu) { statusEl.textContent = 'WebGPU unavailable'; throw new Error('WebGPU unavailable'); }
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice();
const context = canvas.getContext('webgpu');
const format = navigator.gpu.getPreferredCanvasFormat();
context.configure({ device, format, alphaMode: 'opaque' });

function resize() { const dpr = Math.min(window.devicePixelRatio || 1, 2); const w = Math.max(1, Math.floor(canvas.clientWidth * dpr)); const h = Math.max(1, Math.floor(canvas.clientHeight * dpr)); if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; return true; } return false; }
window.addEventListener('resize', resize);
function progressiveLevels(width, height, minWidth = 480) { const out = []; let scale = 1; while (Math.floor(width / (scale * 2)) >= minWidth) scale *= 2; let level = 0; while (scale >= 1) { out.push({ level, width: Math.max(1, Math.floor(width / scale)), height: Math.max(1, Math.floor(height / scale)) }); scale = Math.floor(scale / 2); level++; } return out; }
function writeParams(buffer, width, height, level = 0) { const ab = new ArrayBuffer(64); const dv = new DataView(ab); const u32 = (i, v) => dv.setUint32(i*4, v, true); const f32 = (i, v) => dv.setFloat32(i*4, v, true); u32(0,width); u32(1,height); u32(2,16); u32(3,12); u32(4,16); f32(5,parseFloat(sliders.spin.value)); u32(6,parseInt(sliders.steps.value,10)); f32(7,0.055); f32(8,parseFloat(sliders.camr.value)); f32(9,parseFloat(sliders.inc.value)*Math.PI/180); f32(10,34*Math.PI/180); f32(11,220); u32(12,level); u32(13,0); u32(14,0); u32(15,0); device.queue.writeBuffer(buffer, 0, ab); }
function makeUniformBuffer() { return device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }); }
function makeSyntheticBricks(nr=16, nt=12, np=16) { const r = new Float32Array(nr), th = new Float32Array(nt), ph = new Float32Array(np), coeffs = new Float32Array(nr*nt*np*11); for(let i=0;i<nr;i++) r[i]=2.4 + i*(58/(nr-1)); for(let j=0;j<nt;j++) th[j]=0.045 + j*((Math.PI-0.09)/(nt-1)); for(let k=0;k<np;k++) ph[k]=k*(2*Math.PI/np); for(let i=0;i<nr;i++) for(let j=0;j<nt;j++) for(let k=0;k<np;k++){ const rr=r[i], tt=th[j], pp=ph[k]; const vert=(tt-Math.PI/2)/0.34; const rad=Math.log(rr/12)/0.48; const rho=Math.exp(-0.5*(vert*vert+rad*rad))*(1+0.1*Math.sin(2*pp+0.5*Math.log(rr))); const base=rho*parseFloat(sliders.density.value)*8e-4; const idx=(((i*nt+j)*np+k)*11); coeffs[idx+0]=base; coeffs[idx+1]=0.22*base*Math.cos(2*pp); coeffs[idx+2]=0.22*base*Math.sin(2*pp); coeffs[idx+3]=0.02*base*Math.sin(pp); coeffs[idx+4]=parseFloat(sliders.absorb.value)*base*0.025; coeffs[idx+5]=0.002*base; coeffs[idx+6]=0; coeffs[idx+7]=0; coeffs[idx+8]=0.001*base; coeffs[idx+9]=0.0003*base; coeffs[idx+10]=0.00015*base; } return {r,th,ph,coeffs}; }
function storageBuffer(data, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST) { const b = device.createBuffer({ size: Math.max(4, data.byteLength), usage, mappedAtCreation: true }); new data.constructor(b.getMappedRange()).set(data); b.unmap(); return b; }

async function runStokesRenderer() {
  statusEl.textContent = 'progressive Stokes coefficient-brick compute';
  const computeCode = await fetch('./src/stokes_brick_compute.wgsl').then(r => r.text());
  const displayCode = await fetch('./src/display_stokes.wgsl').then(r => r.text());
  const computeModule = device.createShaderModule({ code: computeCode });
  const displayModule = device.createShaderModule({ code: displayCode });
  const bricks = makeSyntheticBricks();
  const coeffBuf = storageBuffer(bricks.coeffs); const rBuf = storageBuffer(bricks.r); const thBuf = storageBuffer(bricks.th); const phBuf = storageBuffer(bricks.ph);
  let stokesBuffer = null, uniformBuffer = makeUniformBuffer(), computePipeline = null, computeBindGroup = null, displayPipeline = null, displayBindGroup = null;
  function rebuild(width, height) {
    stokesBuffer = device.createBuffer({ size: width*height*16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const cLayout = device.createBindGroupLayout({ entries: [
      { binding:0, visibility:GPUShaderStage.COMPUTE, buffer:{ type:'storage' } }, { binding:1, visibility:GPUShaderStage.COMPUTE, buffer:{ type:'read-only-storage' } },
      { binding:2, visibility:GPUShaderStage.COMPUTE, buffer:{ type:'read-only-storage' } }, { binding:3, visibility:GPUShaderStage.COMPUTE, buffer:{ type:'read-only-storage' } },
      { binding:4, visibility:GPUShaderStage.COMPUTE, buffer:{ type:'read-only-storage' } }, { binding:5, visibility:GPUShaderStage.COMPUTE, buffer:{ type:'uniform' } },
    ]});
    computePipeline = device.createComputePipeline({ layout: device.createPipelineLayout({ bindGroupLayouts:[cLayout] }), compute:{ module:computeModule, entryPoint:'main' }});
    computeBindGroup = device.createBindGroup({ layout:cLayout, entries:[ {binding:0,resource:{buffer:stokesBuffer}}, {binding:1,resource:{buffer:coeffBuf}}, {binding:2,resource:{buffer:rBuf}}, {binding:3,resource:{buffer:thBuf}}, {binding:4,resource:{buffer:phBuf}}, {binding:5,resource:{buffer:uniformBuffer}} ] });
    const dLayout = device.createBindGroupLayout({ entries:[ {binding:0, visibility:GPUShaderStage.FRAGMENT, buffer:{ type:'read-only-storage' }}, {binding:1, visibility:GPUShaderStage.FRAGMENT, buffer:{ type:'uniform' }} ]});
    displayPipeline = device.createRenderPipeline({ layout: device.createPipelineLayout({ bindGroupLayouts:[dLayout] }), vertex:{module:displayModule,entryPoint:'vs_main'}, fragment:{module:displayModule,entryPoint:'fs_main',targets:[{format}]}, primitive:{topology:'triangle-list'} });
    displayBindGroup = device.createBindGroup({ layout:dLayout, entries:[ {binding:0,resource:{buffer:stokesBuffer}}, {binding:1,resource:{buffer:uniformBuffer}} ] });
  }
  function frame() {
    resize(); const levels = progressiveLevels(canvas.width, canvas.height, 480); const lvl = levels[Math.min(Math.floor(performance.now()/850) % levels.length, levels.length-1)];
    if (!stokesBuffer || lvl.width*lvl.height*16 > stokesBuffer.size) rebuild(lvl.width, lvl.height);
    writeParams(uniformBuffer, lvl.width, lvl.height, lvl.level);
    const encoder = device.createCommandEncoder();
    const cp = encoder.beginComputePass(); cp.setPipeline(computePipeline); cp.setBindGroup(0, computeBindGroup); cp.dispatchWorkgroups(Math.ceil(lvl.width/8), Math.ceil(lvl.height/8)); cp.end();
    const rp = encoder.beginRenderPass({ colorAttachments:[{ view:context.getCurrentTexture().createView(), clearValue:{r:0,g:0,b:0,a:1}, loadOp:'clear', storeOp:'store' }]});
    rp.setPipeline(displayPipeline); rp.setBindGroup(0, displayBindGroup); rp.draw(3); rp.end();
    device.queue.submit([encoder.finish()]); requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

async function runFragmentRenderer() {
  const shaderPath = shaderSelect.value === 'volume' ? './src/grrt_volume.wgsl' : './src/kerr_raytrace.wgsl';
  const shaderCode = await fetch(shaderPath).then(r => r.text()); statusEl.textContent = shaderSelect.value === 'volume' ? 'GRRT volume shader' : 'thin disk shader';
  const shader = device.createShaderModule({ code: shaderCode }); const uniformBuffer = makeUniformBuffer();
  const bindGroupLayout = device.createBindGroupLayout({ entries: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }] });
  const pipeline = device.createRenderPipeline({ layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }), vertex: { module: shader, entryPoint: 'vs_main' }, fragment: { module: shader, entryPoint: 'fs_main', targets: [{ format }] }, primitive: { topology: 'triangle-list' } });
  const bindGroup = device.createBindGroup({ layout: bindGroupLayout, entries: [{ binding: 0, resource: { buffer: uniformBuffer } }] });
  function frame(time) { resize(); writeParams(uniformBuffer, canvas.width, canvas.height, 0); const encoder = device.createCommandEncoder(); const pass = encoder.beginRenderPass({ colorAttachments: [{ view: context.getCurrentTexture().createView(), clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: 'clear', storeOp: 'store' }] }); pass.setPipeline(pipeline); pass.setBindGroup(0, bindGroup); pass.draw(3); pass.end(); device.queue.submit([encoder.finish()]); requestAnimationFrame(frame); }
  requestAnimationFrame(frame);
}

resize();
if (shaderSelect.value === 'stokes') runStokesRenderer(); else runFragmentRenderer();
