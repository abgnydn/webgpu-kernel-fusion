#!/usr/bin/env node
/**
 * DIM Scaling Benchmark — WebGPU Rastrigin at DIM=100 and DIM=500
 * N=30 independent 30-second runs per DIM.
 * Reports: mean fitness, std fitness, mean gen/s, total generations.
 */

const puppeteer = require('puppeteer-core');
const http = require('http');
const fs = require('fs');
const path = require('path');

const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const CHROME_ARGS = [
  '--ignore-gpu-blocklist',
  '--enable-features=WebGPU',
  '--no-sandbox',
  '--disable-setuid-sandbox',
];
const PORT = 9877;
const N = 30;
const DURATION_S = 30;

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

function generatePage(pop, dim) {
  return `<!DOCTYPE html><html><head><meta charset="UTF-8">
<title>DIM Scaling POP=${pop} DIM=${dim}</title></head>
<body><pre id="log">Loading...</pre>
<script>
const POP = ${pop}, DIM = ${dim}, WORKGROUP = 64;
const DURATION = ${DURATION_S * 1000};
const log = document.getElementById('log');

async function run() {
  if (!navigator.gpu) { log.textContent = 'NO_WEBGPU'; return; }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) { log.textContent = 'NO_ADAPTER'; return; }
  const device = await adapter.requestDevice({
    requiredLimits: { maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize }
  });

  const shader = device.createShaderModule({ code: \`
    struct Uniforms { generation: u32, seed: u32, pad1: f32, pad2: f32 }
    @group(0) @binding(0) var<uniform> u: Uniforms;
    @group(0) @binding(1) var<storage, read_write> pop: array<f32>;
    @group(0) @binding(2) var<storage, read_write> fitnesses: array<f32>;
    @group(0) @binding(3) var<storage, read_write> next_pop: array<f32>;

    fn rng(state: ptr<function, u32>) -> f32 {
      *state = *state * 1103515245u + 12345u;
      return f32((*state >> 16u) & 0x7FFFu) / 32767.0;
    }

    const POP: u32 = \${POP}u;
    const DIM: u32 = \${DIM}u;

    @compute @workgroup_size(\${WORKGROUP})
    fn fitness(@builtin(global_invocation_id) gid: vec3u) {
      let idx = gid.x;
      if (idx >= POP) { return; }
      var val = 10.0 * f32(DIM);
      for (var d = 0u; d < DIM; d++) {
        let x = pop[idx * DIM + d];
        val += x * x - 10.0 * cos(2.0 * 3.14159265 * x);
      }
      fitnesses[idx] = -val;
    }

    @compute @workgroup_size(\${WORKGROUP})
    fn evolve(@builtin(global_invocation_id) gid: vec3u) {
      let idx = gid.x;
      if (idx >= POP) { return; }
      var seed = idx + u.generation * POP + u.seed;

      if (idx == 0u) {
        var bestIdx = 0u; var bestF = fitnesses[0];
        for (var i = 1u; i < POP; i++) { if (fitnesses[i] > bestF) { bestF = fitnesses[i]; bestIdx = i; } }
        for (var d = 0u; d < DIM; d++) { next_pop[d] = pop[bestIdx * DIM + d]; }
        return;
      }
      var p1 = u32(rng(&seed) * f32(POP));
      for (var t = 0u; t < 4u; t++) { let c = u32(rng(&seed) * f32(POP)); if (fitnesses[c] > fitnesses[p1]) { p1 = c; } }
      var p2 = u32(rng(&seed) * f32(POP));
      for (var t = 0u; t < 4u; t++) { let c = u32(rng(&seed) * f32(POP)); if (fitnesses[c] > fitnesses[p2]) { p2 = c; } }
      for (var d = 0u; d < DIM; d++) {
        var gene = select(pop[p2 * DIM + d], pop[p1 * DIM + d], rng(&seed) < 0.5);
        gene += (rng(&seed) * 2.0 - 1.0) * 0.3;
        next_pop[idx * DIM + d] = gene;
      }
    }
  \` });

  const popBuf = device.createBuffer({ size: POP * DIM * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, mappedAtCreation: true });
  const arr = new Float32Array(popBuf.getMappedRange());
  for (let i = 0; i < arr.length; i++) arr[i] = (Math.random() * 10.24) - 5.12;
  popBuf.unmap();

  const pop2Buf = device.createBuffer({ size: POP * DIM * 4, usage: GPUBufferUsage.STORAGE });
  const fitBuf  = device.createBuffer({ size: POP * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const readBuf = device.createBuffer({ size: POP * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const uniBuf  = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  const fitPipe = device.createComputePipeline({ layout: 'auto', compute: { module: shader, entryPoint: 'fitness' } });
  const evoPipe = device.createComputePipeline({ layout: 'auto', compute: { module: shader, entryPoint: 'evolve' } });

  const bg = device.createBindGroup({ layout: fitPipe.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: uniBuf } },
    { binding: 1, resource: { buffer: popBuf } },
    { binding: 2, resource: { buffer: fitBuf } },
    { binding: 3, resource: { buffer: pop2Buf } },
  ]});
  const bg2 = device.createBindGroup({ layout: fitPipe.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: uniBuf } },
    { binding: 1, resource: { buffer: pop2Buf } },
    { binding: 2, resource: { buffer: fitBuf } },
    { binding: 3, resource: { buffer: popBuf } },
  ]});

  const dispatches = Math.ceil(POP / WORKGROUP);
  let gen = 0, useA = true;

  // Warmup 200 gens
  for (let i = 0; i < 200; i++) {
    device.queue.writeBuffer(uniBuf, 0, new Uint32Array([gen, Math.random() * 0xFFFFFFFF, 0, 0]));
    const enc = device.createCommandEncoder();
    const b = useA ? bg : bg2;
    let pass = enc.beginComputePass(); pass.setPipeline(fitPipe); pass.setBindGroup(0, b); pass.dispatchWorkgroups(dispatches); pass.end();
    pass = enc.beginComputePass(); pass.setPipeline(evoPipe); pass.setBindGroup(0, b); pass.dispatchWorkgroups(dispatches); pass.end();
    device.queue.submit([enc.finish()]);
    useA = !useA; gen++;
  }
  await device.queue.onSubmittedWorkDone();

  // Timed run
  const t0 = performance.now();
  let genCount = 0;
  while (performance.now() - t0 < DURATION) {
    for (let batch = 0; batch < 50; batch++) {
      device.queue.writeBuffer(uniBuf, 0, new Uint32Array([gen, Math.random() * 0xFFFFFFFF, 0, 0]));
      const enc = device.createCommandEncoder();
      const b = useA ? bg : bg2;
      let pass = enc.beginComputePass(); pass.setPipeline(fitPipe); pass.setBindGroup(0, b); pass.dispatchWorkgroups(dispatches); pass.end();
      pass = enc.beginComputePass(); pass.setPipeline(evoPipe); pass.setBindGroup(0, b); pass.dispatchWorkgroups(dispatches); pass.end();
      device.queue.submit([enc.finish()]);
      useA = !useA; gen++; genCount++;
    }
    await device.queue.onSubmittedWorkDone();
  }
  await device.queue.onSubmittedWorkDone();

  const elapsed = (performance.now() - t0) / 1000;
  const gps = genCount / elapsed;

  // Read best fitness
  const enc2 = device.createCommandEncoder();
  enc2.copyBufferToBuffer(fitBuf, 0, readBuf, 0, POP * 4);
  device.queue.submit([enc2.finish()]);
  await readBuf.mapAsync(GPUMapMode.READ);
  const fits = new Float32Array(readBuf.getMappedRange());
  let bestFit = fits[0];
  for (let i = 1; i < POP; i++) if (fits[i] > bestFit) bestFit = fits[i];
  readBuf.unmap();

  log.textContent = JSON.stringify({
    pop: POP, dim: DIM, gps, elapsed, gens: genCount, bestFit
  });
}
run().catch(e => { log.textContent = 'ERROR: ' + e.message; });
</script></body></html>`;
}

let serverPage = '';
function startServer(port) {
  return new Promise(resolve => {
    const srv = http.createServer((req, res) => {
      res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
      res.end(serverPage);
    });
    srv.listen(port, () => resolve(srv));
  });
}

async function runPage(browser, url, timeout = 120000) {
  const page = await browser.newPage();
  await page.goto(url, { waitUntil: 'domcontentloaded' });
  const deadline = Date.now() + timeout;
  while (Date.now() < deadline) {
    const text = await page.$eval('#log', el => el.textContent).catch(() => 'Loading...');
    if (text.startsWith('{')) { await page.close(); return JSON.parse(text); }
    if (text.startsWith('ERROR') || text === 'NO_WEBGPU' || text === 'NO_ADAPTER') {
      await page.close(); throw new Error(text);
    }
    await sleep(1000);
  }
  await page.close();
  throw new Error('TIMEOUT');
}

async function main() {
  const srv = await startServer(PORT);
  const browser = await puppeteer.launch({
    headless: false, defaultViewport: null,
    args: CHROME_ARGS, executablePath: CHROME,
  });

  const dims = [100, 500];

  for (const dim of dims) {
    console.log(`\n${'═'.repeat(60)}`);
    console.log(`  DIM=${dim} — N=${N} runs, ${DURATION_S}s each`);
    console.log('═'.repeat(60));

    serverPage = generatePage(4096, dim);
    const results = [];

    for (let i = 0; i < N; i++) {
      process.stdout.write(`  Run ${i + 1}/${N} ... `);
      try {
        const r = await runPage(browser, `http://127.0.0.1:${PORT}/?r=${Date.now()}`);
        console.log(`fitness=${r.bestFit.toFixed(4)}, ${r.gps.toFixed(1)} gen/s, ${r.gens} gens`);
        results.push(r);
      } catch (e) {
        console.log(`FAILED: ${e.message}`);
      }
      await sleep(2000);
    }

    const fitnesses = results.map(r => r.bestFit);
    const gpsArr = results.map(r => r.gps);
    const gensArr = results.map(r => r.gens);

    const mean = arr => arr.reduce((a, b) => a + b, 0) / arr.length;
    const std = arr => {
      const m = mean(arr);
      return Math.sqrt(arr.reduce((s, x) => s + (x - m) ** 2, 0) / (arr.length - 1));
    };

    console.log(`\n  RESULTS DIM=${dim}:`);
    console.log(`  Fitness: ${mean(fitnesses).toFixed(4)} +/- ${std(fitnesses).toFixed(4)}`);
    console.log(`  Gen/s:   ${mean(gpsArr).toFixed(1)} +/- ${std(gpsArr).toFixed(1)}`);
    console.log(`  Gens:    ${mean(gensArr).toFixed(0)} +/- ${std(gensArr).toFixed(0)}`);
    console.log(`  N:       ${results.length}`);
  }

  await browser.close();
  srv.close();
}

main().catch(err => { console.error('Fatal:', err); process.exit(1); });
