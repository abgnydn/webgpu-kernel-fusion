#!/usr/bin/env node
/**
 * Comprehensive WebGPU Benchmark Suite
 *
 * Tests:
 *   1. Population scaling (POP = 512, 1024, 2048, 4096, 8192, 16384, 32768)
 *   2. Multi-benchmark suite (Rastrigin, Sphere, Ackley, Schwefel, Griewank)
 *   3. Throughput stability over 60s (gen/s sampled every second)
 *   4. Genome size scaling (DIM = 100, 500, 1000, 2000, 4000)
 *   5. Multi-tab GPU contention (1, 2, 4, 8 tabs)
 */

const puppeteer = require('puppeteer-core');
const http = require('http');
const fs = require('fs');
const path = require('path');

const CHROME_PATH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const CHROME_ARGS = [
  '--ignore-gpu-blocklist',
  '--enable-features=WebGPU',
  '--no-sandbox',
  '--disable-setuid-sandbox',
];

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
function stats(arr) {
  const v = arr.filter(x => !isNaN(x) && isFinite(x));
  if (!v.length) return { mean: NaN, std: NaN, min: NaN, max: NaN, n: 0, cv: NaN };
  const mean = v.reduce((s, x) => s + x, 0) / v.length;
  const std = Math.sqrt(v.reduce((s, x) => s + (x - mean) ** 2, 0) / v.length);
  return { mean, std, min: Math.min(...v), max: Math.max(...v), n: v.length, cv: (std / mean * 100) };
}

// ─── Dynamic benchmark page generator ──────────────────────────────────────
function generateBenchmarkHTML(pop, dim, fitnessName = 'rastrigin') {
  const fitnessFunctions = {
    rastrigin: `
      var val = 10.0 * f32(DIM);
      for (var d = 0u; d < DIM; d++) {
        let x = pop[idx * DIM + d];
        val += x * x - 10.0 * cos(2.0 * 3.14159265 * x);
      }
      fitnesses[idx] = -val;`,
    sphere: `
      var val = 0.0;
      for (var d = 0u; d < DIM; d++) {
        let x = pop[idx * DIM + d];
        val += x * x;
      }
      fitnesses[idx] = -val;`,
    ackley: `
      var sum1 = 0.0; var sum2 = 0.0;
      for (var d = 0u; d < DIM; d++) {
        let x = pop[idx * DIM + d];
        sum1 += x * x; sum2 += cos(2.0 * 3.14159265 * x);
      }
      let n = f32(DIM);
      fitnesses[idx] = -(20.0 * (1.0 - exp(-0.2 * sqrt(sum1 / n))) + 2.71828 - exp(sum2 / n));`,
    schwefel: `
      var val = 418.9829 * f32(DIM);
      for (var d = 0u; d < DIM; d++) {
        let x = pop[idx * DIM + d] * 100.0;
        let ax = abs(x);
        val -= select(x * sin(sqrt(ax)), 0.0, ax < 0.001);
      }
      fitnesses[idx] = -val;`,
    griewank: `
      var sum_sq = 0.0; var prod_cos = 1.0;
      for (var d = 0u; d < DIM; d++) {
        let x = pop[idx * DIM + d];
        sum_sq += x * x;
        prod_cos *= cos(x / sqrt(f32(d + 1u)));
      }
      fitnesses[idx] = -(sum_sq / 4000.0 - prod_cos + 1.0);`,
  };

  const fitCode = fitnessFunctions[fitnessName] || fitnessFunctions.rastrigin;

  return `<!DOCTYPE html><html><head><meta charset="UTF-8">
<title>Bench ${fitnessName} POP=${pop} DIM=${dim}</title></head>
<body><pre id="log">Loading...</pre>
<script>
const POP = ${pop}, DIM = ${dim}, WORKGROUP = 64;
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
      ${fitCode}
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
  let gen = 0;
  let useA = true;

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

  // Timed: run for 10s, count generations
  const t0 = performance.now();
  const DURATION = 10000;
  let genCount = 0;
  const samples = []; // per-second samples
  let lastSample = t0;

  while (performance.now() - t0 < DURATION) {
    // Run 50 gens between checks
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

    const now = performance.now();
    if (now - lastSample >= 1000) {
      samples.push(genCount / ((now - t0) / 1000));
      lastSample = now;
    }
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

  // Report GPU memory estimate
  const memMB = ((POP * DIM * 4 * 2) + (POP * 4 * 2) + 16) / (1024 * 1024);

  log.textContent = JSON.stringify({
    pop: POP, dim: DIM, fitness: '${fitnessName}',
    gps: gps, elapsed: elapsed, gens: genCount,
    bestFit: bestFit, memMB: memMB.toFixed(1),
    samples: samples,
    gpu: (await adapter.requestAdapterInfo?.())?.description || 'unknown'
  });
}
run().catch(e => { log.textContent = 'ERROR: ' + e.message; });
</script></body></html>`;
}

// ─── Server ────────────────────────────────────────────────────────────────
let serverPages = {};
function startServer(port) {
  return new Promise(resolve => {
    const srv = http.createServer((req, res) => {
      const url = new URL(req.url, 'http://localhost');
      const key = url.pathname.replace('/', '') || 'index';
      const html = serverPages[key];
      if (html) {
        res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
        res.end(html);
      } else {
        res.writeHead(404); res.end('Not found');
      }
    });
    srv.listen(port, () => resolve(srv));
  });
}

async function runBenchPage(browser, url, timeout = 30000) {
  const page = await browser.newPage();
  await page.goto(url, { waitUntil: 'domcontentloaded' });

  const deadline = Date.now() + timeout;
  while (Date.now() < deadline) {
    const text = await page.$eval('#log', el => el.textContent).catch(() => 'Loading...');
    if (text.startsWith('{') || text.startsWith('ERROR') || text === 'NO_WEBGPU' || text === 'NO_ADAPTER') {
      await page.close();
      if (text.startsWith('{')) return JSON.parse(text);
      throw new Error(text);
    }
    await sleep(500);
  }
  await page.close();
  throw new Error('TIMEOUT');
}

// ═══════════════════════════════════════════════════════════════════════════
//  MAIN
// ═══════════════════════════════════════════════════════════════════════════
async function main() {
  const PORT = 9876;
  const srv = await startServer(PORT);
  const browser = await puppeteer.launch({
    headless: false, defaultViewport: null,
    args: CHROME_ARGS, executablePath: CHROME_PATH,
  });

  const allResults = {};

  // ── TEST 1: Population Scaling ──────────────────────────────────────────
  console.log('\n' + '═'.repeat(64));
  console.log('  TEST 1: POPULATION SCALING (Rastrigin, DIM=2000)');
  console.log('═'.repeat(64));

  const popSizes = [512, 1024, 2048, 4096, 8192, 16384, 32768];
  const popResults = [];

  for (const pop of popSizes) {
    serverPages['bench'] = generateBenchmarkHTML(pop, 2000, 'rastrigin');
    process.stdout.write(`  POP=${String(pop).padStart(5)} ... `);
    try {
      const r = await runBenchPage(browser, `http://127.0.0.1:${PORT}/bench`, 60000);
      console.log(`${r.gps.toFixed(1)} gen/s  (mem: ${r.memMB} MB, best: ${r.bestFit.toFixed(0)})`);
      popResults.push({ pop, gps: r.gps, memMB: parseFloat(r.memMB), bestFit: r.bestFit });
    } catch (e) {
      console.log(`FAILED: ${e.message}`);
      popResults.push({ pop, gps: 0, memMB: 0, error: e.message });
    }
    await sleep(3000);
  }
  allResults.populationScaling = popResults;

  // ── TEST 2: Multi-Benchmark Suite ───────────────────────────────────────
  console.log('\n' + '═'.repeat(64));
  console.log('  TEST 2: MULTI-BENCHMARK SUITE (POP=4096, DIM=2000)');
  console.log('═'.repeat(64));

  const benchmarks = ['sphere', 'rastrigin', 'ackley', 'schwefel', 'griewank'];
  const benchResults = [];

  for (const fname of benchmarks) {
    serverPages['bench'] = generateBenchmarkHTML(4096, 2000, fname);
    process.stdout.write(`  ${fname.padEnd(12)} ... `);
    try {
      const r = await runBenchPage(browser, `http://127.0.0.1:${PORT}/bench`, 60000);
      console.log(`${r.gps.toFixed(1)} gen/s  (best: ${r.bestFit.toFixed(2)})`);
      benchResults.push({ name: fname, gps: r.gps, bestFit: r.bestFit });
    } catch (e) {
      console.log(`FAILED: ${e.message}`);
      benchResults.push({ name: fname, gps: 0, error: e.message });
    }
    await sleep(3000);
  }
  allResults.multiBenchmark = benchResults;

  // ── TEST 3: Throughput Stability ────────────────────────────────────────
  console.log('\n' + '═'.repeat(64));
  console.log('  TEST 3: THROUGHPUT STABILITY (60s, Rastrigin POP=4096)');
  console.log('═'.repeat(64));

  serverPages['bench'] = generateBenchmarkHTML(4096, 2000, 'rastrigin');
  // Override duration to 60s
  serverPages['bench'] = serverPages['bench'].replace('const DURATION = 10000;', 'const DURATION = 60000;');
  try {
    const r = await runBenchPage(browser, `http://127.0.0.1:${PORT}/bench`, 120000);
    const s = stats(r.samples);
    console.log(`  Mean: ${s.mean.toFixed(1)} gen/s`);
    console.log(`  Std:  ${s.std.toFixed(1)} gen/s`);
    console.log(`  CV:   ${s.cv.toFixed(1)}%`);
    console.log(`  Min:  ${s.min.toFixed(1)}, Max: ${s.max.toFixed(1)}`);
    console.log(`  Samples: ${r.samples.length} (1/sec)`);
    allResults.stabilityTest = { ...s, samples: r.samples, elapsed: r.elapsed };
  } catch (e) {
    console.log(`  FAILED: ${e.message}`);
  }
  await sleep(3000);

  // ── TEST 4: Genome Size Scaling ─────────────────────────────────────────
  console.log('\n' + '═'.repeat(64));
  console.log('  TEST 4: GENOME SIZE SCALING (POP=4096, Rastrigin)');
  console.log('═'.repeat(64));

  const dims = [100, 500, 1000, 2000, 4000];
  const dimResults = [];

  for (const dim of dims) {
    serverPages['bench'] = generateBenchmarkHTML(4096, dim, 'rastrigin');
    process.stdout.write(`  DIM=${String(dim).padStart(5)} ... `);
    try {
      const r = await runBenchPage(browser, `http://127.0.0.1:${PORT}/bench`, 60000);
      console.log(`${r.gps.toFixed(1)} gen/s  (mem: ${r.memMB} MB)`);
      dimResults.push({ dim, gps: r.gps, memMB: parseFloat(r.memMB) });
    } catch (e) {
      console.log(`FAILED: ${e.message}`);
      dimResults.push({ dim, gps: 0, error: e.message });
    }
    await sleep(3000);
  }
  allResults.genomeScaling = dimResults;

  // ── TEST 5: Multi-Tab GPU Contention ────────────────────────────────────
  console.log('\n' + '═'.repeat(64));
  console.log('  TEST 5: MULTI-TAB GPU CONTENTION (Rastrigin POP=4096)');
  console.log('═'.repeat(64));

  const tabCounts = [1, 2, 4, 8];
  const tabResults = [];
  serverPages['bench'] = generateBenchmarkHTML(4096, 2000, 'rastrigin');

  for (const nTabs of tabCounts) {
    process.stdout.write(`  ${nTabs} tab(s) ... `);
    const pages = [];
    try {
      for (let i = 0; i < nTabs; i++) {
        const page = await browser.newPage();
        await page.goto(`http://127.0.0.1:${PORT}/bench?t=${Date.now()}_${i}`, { waitUntil: 'domcontentloaded' });
        pages.push(page);
      }

      // Wait for all to finish
      const results = [];
      for (const page of pages) {
        const deadline = Date.now() + 60000;
        while (Date.now() < deadline) {
          const text = await page.$eval('#log', el => el.textContent).catch(() => 'Loading...');
          if (text.startsWith('{')) { results.push(JSON.parse(text)); break; }
          if (text.startsWith('ERROR')) { results.push({ gps: 0, error: text }); break; }
          await sleep(500);
        }
      }

      const gpsValues = results.map(r => r.gps || 0);
      const totalGps = gpsValues.reduce((a, b) => a + b, 0);
      const perTab = totalGps / nTabs;
      console.log(`per-tab: ${perTab.toFixed(1)} gen/s  total: ${totalGps.toFixed(1)} gen/s`);
      tabResults.push({ tabs: nTabs, perTab, totalGps, individual: gpsValues });
    } catch (e) {
      console.log(`FAILED: ${e.message}`);
      tabResults.push({ tabs: nTabs, perTab: 0, totalGps: 0, error: e.message });
    }
    for (const p of pages) await p.close().catch(() => {});
    await sleep(5000);
  }
  allResults.tabContention = tabResults;

  // ── Save & Print ────────────────────────────────────────────────────────
  await browser.close();
  srv.close();

  console.log('\n' + '═'.repeat(64));
  console.log('  COMPREHENSIVE RESULTS');
  console.log('═'.repeat(64));

  // Population scaling table
  console.log('\n  Population Scaling (DIM=2000):');
  console.log('  POP    | gen/s   | VRAM (MB)');
  console.log('  -------+---------+----------');
  for (const r of popResults) {
    console.log(`  ${String(r.pop).padStart(5)}  | ${r.gps.toFixed(1).padStart(7)} | ${String(r.memMB).padStart(8)}`);
  }

  // Multi-benchmark table
  console.log('\n  Multi-Benchmark (POP=4096, DIM=2000):');
  console.log('  Function    | gen/s');
  console.log('  ------------+-------');
  for (const r of benchResults) {
    console.log(`  ${r.name.padEnd(12)} | ${r.gps.toFixed(1)}`);
  }

  // Genome scaling table
  console.log('\n  Genome Scaling (POP=4096):');
  console.log('  DIM    | gen/s   | VRAM (MB)');
  console.log('  -------+---------+----------');
  for (const r of dimResults) {
    console.log(`  ${String(r.dim).padStart(5)}  | ${r.gps.toFixed(1).padStart(7)} | ${String(r.memMB).padStart(8)}`);
  }

  // Tab contention table
  console.log('\n  Multi-Tab Contention:');
  console.log('  Tabs | per-tab g/s | total g/s | efficiency');
  console.log('  -----+-------------+-----------+-----------');
  const singleTab = tabResults[0]?.perTab || 1;
  for (const r of tabResults) {
    const eff = (r.perTab / singleTab * 100).toFixed(0);
    console.log(`  ${String(r.tabs).padStart(4)} | ${r.perTab.toFixed(1).padStart(11)} | ${r.totalGps.toFixed(1).padStart(9)} | ${eff}%`);
  }

  // Save JSON
  const outPath = path.join(__dirname, 'comprehensive_results.json');
  fs.writeFileSync(outPath, JSON.stringify(allResults, null, 2));
  console.log(`\n  Saved to ${outPath}`);
  console.log('═'.repeat(64));
}

main().catch(err => {
  console.error('Fatal:', err);
  process.exit(1);
});
