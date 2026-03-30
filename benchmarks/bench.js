#!/usr/bin/env node
/**
 * bench.js — Centralized Benchmark Harness
 *
 * Single entry point for ALL benchmarks. Stores results with git hash,
 * timestamp, and hardware info. Supports selective runs and comparison.
 *
 * Usage:
 *   node bench.js                     # run all benchmarks
 *   node bench.js --suite=perf        # only performance benchmarks
 *   node bench.js --suite=correctness # only correctness tests
 *   node bench.js --suite=baselines   # only Python/JAX/PyTorch baselines
 *   node bench.js --compare           # compare last 2 runs
 *   node bench.js --history           # show all past runs
 *   node bench.js --runs=5            # N runs per benchmark (default 3)
 *
 * Results saved to: benchmarks/results/<timestamp>_<githash>.json
 *
 * Prerequisites:
 *   cd poc/v37_webgpu && python3 serve.py   (port 8081)
 */

const puppeteer = require('puppeteer-core');
const { execSync } = require('child_process');
const http = require('http');
const fs = require('fs');
const path = require('path');
const { ThermalMonitor, snapshot, getChipModel } = require('./thermal_monitor');

// ═══════════════════════════════════════════════════════════════════════════
//  CONFIG
// ═══════════════════════════════════════════════════════════════════════════

const CHROME_PATH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const CHROME_ARGS = [
  '--ignore-gpu-blocklist', '--enable-features=WebGPU',
  '--no-sandbox', '--disable-setuid-sandbox',
];
const V37_PORT = 8081;
const BENCH_PORT = 9877;
const RESULTS_DIR = path.join(__dirname, 'results');

// CLI
const args = Object.fromEntries(
  process.argv.slice(2).filter(a => a.startsWith('--')).map(a => {
    const [k, v] = a.slice(2).split('=');
    return [k, v === undefined ? true : v];
  })
);
const N_RUNS = parseInt(args.runs || '10');
const SUITE = args.suite || 'all';
const WARMUP_MS = 15000;
const SAMPLE_INTERVAL = 2000;
const N_SAMPLES = 8;
const COOLDOWN_MS = 10000;

// ═══════════════════════════════════════════════════════════════════════════
//  UTILITIES
// ═══════════════════════════════════════════════════════════════════════════

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

function stats(arr) {
  const v = arr.filter(x => !isNaN(x) && isFinite(x));
  if (!v.length) return { mean: NaN, std: NaN, min: NaN, max: NaN, n: 0, cv: NaN, ci95: [NaN, NaN] };
  const mean = v.reduce((s, x) => s + x, 0) / v.length;
  const std = Math.sqrt(v.reduce((s, x) => s + (x - mean) ** 2, 0) / v.length);
  // 95% CI using t-distribution critical values (two-tailed)
  const tCrit = { 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 15: 2.145, 20: 2.086, 30: 2.042 };
  const df = v.length - 1;
  const t = tCrit[v.length] || (df > 30 ? 1.96 : 2.228); // fallback to z=1.96 for large n
  const se = std / Math.sqrt(v.length);
  const ci95 = [mean - t * se, mean + t * se];
  return { mean, std, min: Math.min(...v), max: Math.max(...v), n: v.length, cv: std / mean * 100, ci95 };
}

// Welch's t-test (unequal variance, two-tailed)
function welchTTest(a, b) {
  const sa = stats(a), sb = stats(b);
  if (sa.n < 2 || sb.n < 2) return { t: NaN, df: NaN, p: NaN, significant: false };
  const va = sa.std ** 2, vb = sb.std ** 2;
  const na = sa.n, nb = sb.n;
  const t = (sa.mean - sb.mean) / Math.sqrt(va / na + vb / nb);
  const num = (va / na + vb / nb) ** 2;
  const den = (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1);
  const df = num / den;
  // Approximate p-value using normal distribution for df > 30, else conservative
  const absT = Math.abs(t);
  let p;
  if (df > 30) {
    // Normal approximation
    p = 2 * (1 - normalCDF(absT));
  } else {
    // Conservative: use t-table thresholds
    p = absT > 3.5 ? 0.001 : absT > 2.5 ? 0.02 : absT > 2.0 ? 0.05 : absT > 1.5 ? 0.15 : 0.5;
  }
  return { t, df, p, significant: p < 0.05, meanDiff: sa.mean - sb.mean };
}

function normalCDF(x) {
  // Abramowitz & Stegun approximation
  const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741, a4 = -1.453152027, a5 = 1.061405429;
  const p = 0.3275911;
  const sign = x < 0 ? -1 : 1;
  x = Math.abs(x) / Math.SQRT2;
  const t = 1.0 / (1.0 + p * x);
  const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
  return 0.5 * (1.0 + sign * y);
}

function parseFirst(text) {
  const m = text.match(/[\d.]+/);
  return m ? parseFloat(m[0]) : NaN;
}

function getGitInfo() {
  try {
    const hash = execSync('git rev-parse --short HEAD 2>/dev/null', { encoding: 'utf-8' }).trim();
    const dirty = execSync('git status --porcelain 2>/dev/null', { encoding: 'utf-8' }).trim();
    const branch = execSync('git branch --show-current 2>/dev/null', { encoding: 'utf-8' }).trim();
    return { hash: hash + (dirty ? '-dirty' : ''), branch };
  } catch { return { hash: 'unknown', branch: 'unknown' }; }
}

function runPython(script, timeout = 300000) {
  try {
    const out = execSync(`python3 ${script}`, {
      encoding: 'utf-8', timeout, cwd: path.join(__dirname, '..'),
      stdio: ['pipe', 'pipe', 'pipe']
    });
    return out;
  } catch (e) {
    return e.stdout || e.message;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
//  BROWSER MANAGEMENT — FULL ISOLATION
//
//  Each benchmark gets a FRESH Chrome process to prevent:
//    - GPU memory leaks from previous tests
//    - Thermal carryover (heat from test N affecting test N+1)
//    - Browser-level caching or JIT warmth
//    - Shared GPU context scheduling artifacts
// ═══════════════════════════════════════════════════════════════════════════

async function freshBrowser() {
  return puppeteer.launch({
    headless: false, defaultViewport: null,
    args: CHROME_ARGS, executablePath: CHROME_PATH,
  });
}

async function closeBrowser(browser) {
  if (browser) { try { await browser.close(); } catch {} }
}

async function thermalGate(label) {
  // Wait until GPU is cool before next benchmark
  const MAX_WAIT = 60000;
  const TARGET_TEMP = 35; // °C
  const start = Date.now();
  let s = snapshot();
  const temp = s.batteryTempC || 0;
  const gpuUtil = s.gpu?.device || 0;

  if (temp > TARGET_TEMP || gpuUtil > 20) {
    process.stdout.write(`  [thermal gate] temp=${temp}°C gpu=${gpuUtil}% — cooling`);
    while (Date.now() - start < MAX_WAIT) {
      await sleep(3000);
      s = snapshot();
      process.stdout.write('.');
      if ((s.batteryTempC || 0) <= TARGET_TEMP && (s.gpu?.device || 0) <= 20) break;
    }
    const final = snapshot();
    console.log(` → ${final.batteryTempC || '?'}°C gpu=${final.gpu?.device || 0}%`);
  } else {
    // Minimum cooldown even if temp is fine — let GPU fully idle
    await sleep(COOLDOWN_MS);
  }
}

async function waitForEvolution(page, timeout = 60000) {
  const dl = Date.now() + timeout;
  while (Date.now() < dl) {
    const disabled = await page.$eval('#startBtn', el => el.disabled).catch(() => true);
    if (!disabled) break;
    await sleep(300);
  }
  await page.$eval('#startBtn', el => el.click()).catch(() => {});
  await sleep(500);
  const dl2 = Date.now() + 30000;
  while (Date.now() < dl2) {
    const gen = await page.$eval('#gen', el => parseInt(el.textContent) || 0).catch(() => 0);
    if (gen > 0) return;
    await sleep(300);
  }
  throw new Error('Evolution did not start');
}

// ═══════════════════════════════════════════════════════════════════════════
//  DYNAMIC BENCHMARK PAGE (for scaling tests)
// ═══════════════════════════════════════════════════════════════════════════

let _benchServer = null;
let _benchPages = {};

function generateBenchHTML(pop, dim, fitnessName = 'rastrigin') {
  const fitFns = {
    rastrigin: `var val = 10.0 * f32(DIM); for (var d = 0u; d < DIM; d++) { let x = pop[idx * DIM + d]; val += x * x - 10.0 * cos(2.0 * 3.14159265 * x); } fitnesses[idx] = -val;`,
    sphere: `var val = 0.0; for (var d = 0u; d < DIM; d++) { let x = pop[idx * DIM + d]; val += x * x; } fitnesses[idx] = -val;`,
    ackley: `var s1 = 0.0; var s2 = 0.0; for (var d = 0u; d < DIM; d++) { let x = pop[idx * DIM + d]; s1 += x*x; s2 += cos(2.0*3.14159265*x); } let n = f32(DIM); fitnesses[idx] = -(20.0*(1.0-exp(-0.2*sqrt(s1/n)))+2.71828-exp(s2/n));`,
    schwefel: `var val = 418.9829*f32(DIM); for (var d = 0u; d < DIM; d++) { let x = pop[idx*DIM+d]*100.0; let ax = abs(x); val -= select(x*sin(sqrt(ax)),0.0,ax<0.001); } fitnesses[idx] = -val;`,
    griewank: `var ss = 0.0; var pc = 1.0; for (var d = 0u; d < DIM; d++) { let x = pop[idx*DIM+d]; ss += x*x; pc *= cos(x/sqrt(f32(d+1u))); } fitnesses[idx] = -(ss/4000.0-pc+1.0);`,
  };
  const fit = fitFns[fitnessName] || fitFns.rastrigin;
  // Returns self-contained HTML that auto-runs and outputs JSON to #log
  return `<!DOCTYPE html><html><head><meta charset="UTF-8"><title>B</title></head><body><pre id="log">Loading...</pre><script>
const POP=${pop},DIM=${dim},WG=64;async function run(){if(!navigator.gpu){document.getElementById('log').textContent='NO_WEBGPU';return;}
const a=await navigator.gpu.requestAdapter({powerPreference:'high-performance'});if(!a){document.getElementById('log').textContent='NO_ADAPTER';return;}
const d=await a.requestDevice({requiredLimits:{maxStorageBufferBindingSize:a.limits.maxStorageBufferBindingSize}});
const sh=d.createShaderModule({code:\`struct U{generation:u32,seed:u32,p1:f32,p2:f32}@group(0)@binding(0)var<uniform>u:U;@group(0)@binding(1)var<storage,read_write>pop:array<f32>;@group(0)@binding(2)var<storage,read_write>fitnesses:array<f32>;@group(0)@binding(3)var<storage,read_write>next_pop:array<f32>;fn rng(s:ptr<function,u32>)->f32{*s=*s*1103515245u+12345u;return f32((*s>>16u)&0x7FFFu)/32767.0;}const POP:u32=\${POP}u;const DIM:u32=\${DIM}u;@compute@workgroup_size(\${WG})fn fitness(@builtin(global_invocation_id)gid:vec3u){let idx=gid.x;if(idx>=POP){return;}${fit}}@compute@workgroup_size(\${WG})fn evolve(@builtin(global_invocation_id)gid:vec3u){let idx=gid.x;if(idx>=POP){return;}var seed=idx+u.generation*POP+u.seed;if(idx==0u){var bi=0u;var bf=fitnesses[0];for(var i=1u;i<POP;i++){if(fitnesses[i]>bf){bf=fitnesses[i];bi=i;}}for(var dd=0u;dd<DIM;dd++){next_pop[dd]=pop[bi*DIM+dd];}return;}var p1=u32(rng(&seed)*f32(POP));for(var t=0u;t<4u;t++){let c=u32(rng(&seed)*f32(POP));if(fitnesses[c]>fitnesses[p1]){p1=c;}}var p2=u32(rng(&seed)*f32(POP));for(var t=0u;t<4u;t++){let c=u32(rng(&seed)*f32(POP));if(fitnesses[c]>fitnesses[p2]){p2=c;}}for(var dd=0u;dd<DIM;dd++){var g=select(pop[p2*DIM+dd],pop[p1*DIM+dd],rng(&seed)<0.5);g+=(rng(&seed)*2.0-1.0)*0.3;next_pop[idx*DIM+dd]=g;}}\`});
const pB=d.createBuffer({size:POP*DIM*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,mappedAtCreation:true});const ar=new Float32Array(pB.getMappedRange());for(let i=0;i<ar.length;i++)ar[i]=(Math.random()*10.24)-5.12;pB.unmap();
const p2=d.createBuffer({size:POP*DIM*4,usage:GPUBufferUsage.STORAGE});const fB=d.createBuffer({size:POP*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});const rB=d.createBuffer({size:POP*4,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});const uB=d.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
const fP=d.createComputePipeline({layout:'auto',compute:{module:sh,entryPoint:'fitness'}});const eP=d.createComputePipeline({layout:'auto',compute:{module:sh,entryPoint:'evolve'}});
const bg=d.createBindGroup({layout:fP.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:uB}},{binding:1,resource:{buffer:pB}},{binding:2,resource:{buffer:fB}},{binding:3,resource:{buffer:p2}}]});
const bg2=d.createBindGroup({layout:fP.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:uB}},{binding:1,resource:{buffer:p2}},{binding:2,resource:{buffer:fB}},{binding:3,resource:{buffer:pB}}]});
const disp=Math.ceil(POP/WG);let gen=0;let useA=true;for(let i=0;i<200;i++){d.queue.writeBuffer(uB,0,new Uint32Array([gen,Math.random()*0xFFFFFFFF,0,0]));const e=d.createCommandEncoder();let p=e.beginComputePass();p.setPipeline(fP);p.setBindGroup(0,useA?bg:bg2);p.dispatchWorkgroups(disp);p.end();p=e.beginComputePass();p.setPipeline(eP);p.setBindGroup(0,useA?bg:bg2);p.dispatchWorkgroups(disp);p.end();d.queue.submit([e.finish()]);useA=!useA;gen++;}await d.queue.onSubmittedWorkDone();
const t0=performance.now();const DUR=10000;let gc=0;const smp=[];let ls=t0;while(performance.now()-t0<DUR){for(let b=0;b<50;b++){d.queue.writeBuffer(uB,0,new Uint32Array([gen,Math.random()*0xFFFFFFFF,0,0]));const e=d.createCommandEncoder();let p=e.beginComputePass();p.setPipeline(fP);p.setBindGroup(0,useA?bg:bg2);p.dispatchWorkgroups(disp);p.end();p=e.beginComputePass();p.setPipeline(eP);p.setBindGroup(0,useA?bg:bg2);p.dispatchWorkgroups(disp);p.end();d.queue.submit([e.finish()]);useA=!useA;gen++;gc++;}await d.queue.onSubmittedWorkDone();const n=performance.now();if(n-ls>=1000){smp.push(gc/((n-t0)/1000));ls=n;}}await d.queue.onSubmittedWorkDone();
const el=(performance.now()-t0)/1000;const gps=gc/el;const e2=d.createCommandEncoder();e2.copyBufferToBuffer(fB,0,rB,0,POP*4);d.queue.submit([e2.finish()]);await rB.mapAsync(GPUMapMode.READ);const fits=new Float32Array(rB.getMappedRange());let bf=fits[0];for(let i=1;i<POP;i++)if(fits[i]>bf)bf=fits[i];rB.unmap();
const mem=((POP*DIM*4*2)+(POP*4*2)+16)/(1024*1024);document.getElementById('log').textContent=JSON.stringify({pop:POP,dim:DIM,fitness:'${fitnessName}',gps,elapsed:el,gens:gc,bestFit:bf,memMB:mem.toFixed(1),samples:smp});}
run().catch(e=>{document.getElementById('log').textContent='ERROR:'+e.message;});</script></body></html>`;
}

async function startBenchServer() {
  return new Promise(resolve => {
    _benchServer = http.createServer((req, res) => {
      const key = new URL(req.url, 'http://localhost').pathname.replace('/', '') || 'index';
      const html = _benchPages[key];
      if (html) { res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' }); res.end(html); }
      else { res.writeHead(404); res.end('Not found'); }
    });
    _benchServer.listen(BENCH_PORT, () => resolve());
  });
}

async function runBenchPage(browser, url, timeout = 60000) {
  const page = await browser.newPage();
  await page.goto(url, { waitUntil: 'domcontentloaded' });
  const dl = Date.now() + timeout;
  while (Date.now() < dl) {
    const text = await page.$eval('#log', el => el.textContent).catch(() => 'Loading...');
    if (text.startsWith('{') || text.startsWith('ERROR') || text === 'NO_WEBGPU') {
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
//  BENCHMARK SUITES
// ═══════════════════════════════════════════════════════════════════════════

// ── PERF: Rastrigin throughput (p2p_demo) ─────────────────────────────────
async function benchRastriginThroughput() {
  const runs = [];
  for (let r = 0; r < N_RUNS; r++) {
    await thermalGate('rastrigin_' + r);
    const browser = await freshBrowser();
    try {
      const page = await browser.newPage();
      await page.goto(`http://127.0.0.1:${V37_PORT}/p2p_demo.html`, { waitUntil: 'domcontentloaded' });
      await waitForEvolution(page);
      process.stdout.write(`    run ${r + 1}/${N_RUNS}  warmup...`);
      await sleep(WARMUP_MS);
      const samples = [];
      for (let s = 0; s < N_SAMPLES; s++) {
        samples.push(parseFirst(await page.$eval('#speed', el => el.textContent).catch(() => '')));
        await sleep(SAMPLE_INTERVAL);
      }
      const s = stats(samples);
      runs.push(s.mean);
      process.stdout.write(` ${s.mean.toFixed(1)} gen/s\n`);
    } catch (e) {
      process.stdout.write(` FAILED: ${e.message.slice(0, 40)}\n`);
      runs.push(NaN);
    } finally {
      await closeBrowser(browser);
    }
  }
  return { id: 'rastrigin_throughput', label: 'Rastrigin (p2p_demo)', unit: 'gen/s', ...stats(runs), raw: runs };
}

// ── PERF: Financial throughput (index.html) ───────────────────────────────
async function benchFinancialThroughput() {
  const runs = [];
  for (let r = 0; r < N_RUNS; r++) {
    await thermalGate('financial_' + r);
    const browser = await freshBrowser();
    try {
      const page = await browser.newPage();
      await page.goto(`http://127.0.0.1:${V37_PORT}/index.html`, { waitUntil: 'domcontentloaded' });
      await sleep(1000);
      await page.evaluate(() => window.localStorage.clear());
      await page.$eval('#consentBtn', el => el.click()).catch(() => {});
      await sleep(1000);
      await page.evaluate(() => { if (typeof init === 'function') init(); });
      await sleep(3000);
      await waitForEvolution(page);
      process.stdout.write(`    run ${r + 1}/${N_RUNS}  warmup...`);
      await sleep(WARMUP_MS);
      const samples = [];
      for (let s = 0; s < N_SAMPLES; s++) {
        samples.push(parseFirst(await page.$eval('#speed', el => el.textContent).catch(() => '')));
        await sleep(SAMPLE_INTERVAL);
      }
      const s = stats(samples);
      runs.push(s.mean);
      process.stdout.write(` ${s.mean.toFixed(1)} gen/s\n`);
    } catch (e) {
      process.stdout.write(` FAILED: ${e.message.slice(0, 40)}\n`);
      runs.push(NaN);
    } finally {
      await closeBrowser(browser);
    }
  }
  return { id: 'financial_throughput', label: 'Financial (index.html)', unit: 'gen/s', ...stats(runs), raw: runs };
}

// ── PERF: Population scaling ──────────────────────────────────────────────
async function benchPopScaling() {
  const pops = [512, 1024, 2048, 4096, 8192, 16384, 32768];
  const results = [];
  for (const pop of pops) {
    await thermalGate('pop_' + pop);
    const browser = await freshBrowser();
    try {
      _benchPages['bench'] = generateBenchHTML(pop, 2000, 'rastrigin');
      process.stdout.write(`    POP=${String(pop).padStart(5)}...`);
      const r = await runBenchPage(browser, `http://127.0.0.1:${BENCH_PORT}/bench`);
      console.log(` ${r.gps.toFixed(1)} gen/s (${r.memMB} MB)`);
      results.push({ pop, gps: r.gps, memMB: parseFloat(r.memMB) });
    } catch (e) {
      console.log(` FAILED`);
      results.push({ pop, gps: 0, error: e.message });
    } finally {
      await closeBrowser(browser);
    }
  }
  return { id: 'pop_scaling', label: 'Population Scaling', results };
}

// ── PERF: Multi-benchmark suite ───────────────────────────────────────────
async function benchMultiFitness() {
  const fns = ['sphere', 'rastrigin', 'ackley', 'schwefel', 'griewank'];
  const results = [];
  for (const fn of fns) {
    await thermalGate('fitness_' + fn);
    const browser = await freshBrowser();
    try {
      _benchPages['bench'] = generateBenchHTML(4096, 2000, fn);
      process.stdout.write(`    ${fn.padEnd(12)}...`);
      const r = await runBenchPage(browser, `http://127.0.0.1:${BENCH_PORT}/bench`);
      console.log(` ${r.gps.toFixed(1)} gen/s`);
      results.push({ name: fn, gps: r.gps, bestFit: r.bestFit });
    } catch (e) {
      console.log(` FAILED`);
      results.push({ name: fn, gps: 0, error: e.message });
    } finally {
      await closeBrowser(browser);
    }
  }
  return { id: 'multi_fitness', label: 'Multi-Benchmark Suite', results };
}

// ── PERF: Genome size scaling ─────────────────────────────────────────────
async function benchGenomeScaling() {
  const dims = [100, 500, 1000, 2000, 4000];
  const results = [];
  for (const dim of dims) {
    await thermalGate('genome_' + dim);
    const browser = await freshBrowser();
    try {
      _benchPages['bench'] = generateBenchHTML(4096, dim, 'rastrigin');
      process.stdout.write(`    DIM=${String(dim).padStart(5)}...`);
      const r = await runBenchPage(browser, `http://127.0.0.1:${BENCH_PORT}/bench`);
      console.log(` ${r.gps.toFixed(1)} gen/s (${r.memMB} MB)`);
      results.push({ dim, gps: r.gps, memMB: parseFloat(r.memMB) });
    } catch (e) {
      console.log(` FAILED`);
      results.push({ dim, gps: 0, error: e.message });
    } finally {
      await closeBrowser(browser);
    }
  }
  return { id: 'genome_scaling', label: 'Genome Size Scaling', results };
}

// ── PERF: Multi-tab contention ────────────────────────────────────────────
async function benchTabContention() {
  const tabCounts = [1, 2, 4, 8];
  const results = [];
  _benchPages['bench'] = generateBenchHTML(4096, 2000, 'rastrigin');

  for (const n of tabCounts) {
    await thermalGate('tabs_' + n);
    const browser = await freshBrowser();
    try {
      process.stdout.write(`    ${n} tab(s)...`);
      const pages = [];
      for (let i = 0; i < n; i++) {
        const p = await browser.newPage();
        await p.goto(`http://127.0.0.1:${BENCH_PORT}/bench?t=${Date.now()}_${i}`, { waitUntil: 'domcontentloaded' });
        pages.push(p);
      }
      const data = [];
      for (const p of pages) {
        const dl = Date.now() + 60000;
        while (Date.now() < dl) {
          const t = await p.$eval('#log', el => el.textContent).catch(() => 'Loading...');
          if (t.startsWith('{')) { data.push(JSON.parse(t)); break; }
          if (t.startsWith('ERROR')) { data.push({ gps: 0 }); break; }
          await sleep(500);
        }
      }
      const gps = data.map(d => d.gps || 0);
      const total = gps.reduce((a, b) => a + b, 0);
      const perTab = total / n;
      console.log(` per-tab: ${perTab.toFixed(1)}, total: ${total.toFixed(1)} gen/s`);
      results.push({ tabs: n, perTab, totalGps: total });
    } catch (e) {
      console.log(` FAILED`);
      results.push({ tabs: n, perTab: 0, totalGps: 0, error: e.message });
    } finally {
      await closeBrowser(browser);
    }
  }
  return { id: 'tab_contention', label: 'Multi-Tab Contention', results };
}

// ── PERF: GPU utilization ─────────────────────────────────────────────────
async function benchGPUUtil() {
  // Idle — measure before launching Chrome
  await thermalGate('gpu_util_idle');
  const idle = [];
  for (let i = 0; i < 10; i++) { idle.push(snapshot().gpu.device || 0); await sleep(1000); }

  // Load — fresh browser
  await thermalGate('gpu_util_load');
  const browser = await freshBrowser();
  try {
    const page = await browser.newPage();
    await page.goto(`http://127.0.0.1:${V37_PORT}/p2p_demo.html`, { waitUntil: 'domcontentloaded' });
    await waitForEvolution(page);
    await sleep(5000);
    const load = []; const speeds = [];
    for (let i = 0; i < 20; i++) {
      load.push(snapshot().gpu.device || 0);
      speeds.push(parseFirst(await page.$eval('#speed', el => el.textContent).catch(() => '')));
      await sleep(1000);
    }

    return {
      id: 'gpu_util', label: 'GPU Utilization',
      idle: stats(idle), load: stats(load), speed: stats(speeds),
      temp: snapshot().batteryTempC,
    };
  } finally {
    await closeBrowser(browser);
  }
}

// ── PERF: Convergence curve (best fitness vs wall-clock time) ─────────────
function generateConvergenceHTML(pop, dim, durationSec = 60) {
  return `<!DOCTYPE html><html><head><meta charset="UTF-8"><title>Conv</title></head><body><pre id="log">Loading...</pre><script>
const POP=${pop},DIM=${dim},WG=64,DUR=${durationSec * 1000};
async function run(){if(!navigator.gpu){document.getElementById('log').textContent='NO_WEBGPU';return;}
const a=await navigator.gpu.requestAdapter({powerPreference:'high-performance'});if(!a){document.getElementById('log').textContent='NO_ADAPTER';return;}
const d=await a.requestDevice({requiredLimits:{maxStorageBufferBindingSize:a.limits.maxStorageBufferBindingSize}});
const sh=d.createShaderModule({code:\`struct U{generation:u32,seed:u32,p1:f32,p2:f32}@group(0)@binding(0)var<uniform>u:U;@group(0)@binding(1)var<storage,read_write>pop:array<f32>;@group(0)@binding(2)var<storage,read_write>fitnesses:array<f32>;@group(0)@binding(3)var<storage,read_write>next_pop:array<f32>;fn rng(s:ptr<function,u32>)->f32{*s=*s*1103515245u+12345u;return f32((*s>>16u)&0x7FFFu)/32767.0;}const POP:u32=\${POP}u;const DIM:u32=\${DIM}u;@compute@workgroup_size(\${WG})fn fitness(@builtin(global_invocation_id)gid:vec3u){let idx=gid.x;if(idx>=POP){return;}var val=10.0*f32(DIM);for(var dd=0u;dd<DIM;dd++){let x=pop[idx*DIM+dd];val+=x*x-10.0*cos(2.0*3.14159265*x);}fitnesses[idx]=-val;}@compute@workgroup_size(\${WG})fn evolve(@builtin(global_invocation_id)gid:vec3u){let idx=gid.x;if(idx>=POP){return;}var seed=idx+u.generation*POP+u.seed;if(idx==0u){var bi=0u;var bf=fitnesses[0];for(var i=1u;i<POP;i++){if(fitnesses[i]>bf){bf=fitnesses[i];bi=i;}}for(var dd=0u;dd<DIM;dd++){next_pop[dd]=pop[bi*DIM+dd];}return;}var p1=u32(rng(&seed)*f32(POP));for(var t=0u;t<4u;t++){let c=u32(rng(&seed)*f32(POP));if(fitnesses[c]>fitnesses[p1]){p1=c;}}var p2=u32(rng(&seed)*f32(POP));for(var t=0u;t<4u;t++){let c=u32(rng(&seed)*f32(POP));if(fitnesses[c]>fitnesses[p2]){p2=c;}}for(var dd=0u;dd<DIM;dd++){var g=select(pop[p2*DIM+dd],pop[p1*DIM+dd],rng(&seed)<0.5);g+=(rng(&seed)*2.0-1.0)*0.3;next_pop[idx*DIM+dd]=g;}}\`});
const pB=d.createBuffer({size:POP*DIM*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,mappedAtCreation:true});const ar=new Float32Array(pB.getMappedRange());for(let i=0;i<ar.length;i++)ar[i]=(Math.random()*10.24)-5.12;pB.unmap();
const p2=d.createBuffer({size:POP*DIM*4,usage:GPUBufferUsage.STORAGE});const fB=d.createBuffer({size:POP*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});const rB=d.createBuffer({size:POP*4,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});const uB=d.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
const fP=d.createComputePipeline({layout:'auto',compute:{module:sh,entryPoint:'fitness'}});const eP=d.createComputePipeline({layout:'auto',compute:{module:sh,entryPoint:'evolve'}});
const bg=d.createBindGroup({layout:fP.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:uB}},{binding:1,resource:{buffer:pB}},{binding:2,resource:{buffer:fB}},{binding:3,resource:{buffer:p2}}]});
const bg2=d.createBindGroup({layout:fP.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:uB}},{binding:1,resource:{buffer:p2}},{binding:2,resource:{buffer:fB}},{binding:3,resource:{buffer:pB}}]});
const disp=Math.ceil(POP/WG);let gen=0;let useA=true;
// Warmup 200 gens
for(let i=0;i<200;i++){d.queue.writeBuffer(uB,0,new Uint32Array([gen,Math.random()*0xFFFFFFFF,0,0]));const e=d.createCommandEncoder();let p=e.beginComputePass();p.setPipeline(fP);p.setBindGroup(0,useA?bg:bg2);p.dispatchWorkgroups(disp);p.end();p=e.beginComputePass();p.setPipeline(eP);p.setBindGroup(0,useA?bg:bg2);p.dispatchWorkgroups(disp);p.end();d.queue.submit([e.finish()]);useA=!useA;gen++;}
await d.queue.onSubmittedWorkDone();
// Timed convergence run
const curve=[];const t0=performance.now();let gc=0;
while(performance.now()-t0<DUR){
  for(let b=0;b<20;b++){d.queue.writeBuffer(uB,0,new Uint32Array([gen,Math.random()*0xFFFFFFFF,0,0]));const e=d.createCommandEncoder();let p=e.beginComputePass();p.setPipeline(fP);p.setBindGroup(0,useA?bg:bg2);p.dispatchWorkgroups(disp);p.end();p=e.beginComputePass();p.setPipeline(eP);p.setBindGroup(0,useA?bg:bg2);p.dispatchWorkgroups(disp);p.end();d.queue.submit([e.finish()]);useA=!useA;gen++;gc++;}
  await d.queue.onSubmittedWorkDone();
  // Read best fitness
  const e2=d.createCommandEncoder();e2.copyBufferToBuffer(fB,0,rB,0,POP*4);d.queue.submit([e2.finish()]);await rB.mapAsync(GPUMapMode.READ);const fits=new Float32Array(rB.getMappedRange());let bf=fits[0];for(let i=1;i<POP;i++)if(fits[i]>bf)bf=fits[i];rB.unmap();
  curve.push({t:((performance.now()-t0)/1000).toFixed(2),gen:gc,bestFit:bf});
}
document.getElementById('log').textContent=JSON.stringify({pop:POP,dim:DIM,totalGens:gc,elapsed:((performance.now()-t0)/1000).toFixed(2),curve});}
run().catch(e=>{document.getElementById('log').textContent='ERROR:'+e.message;});</script></body></html>`;
}

async function benchConvergenceCurve() {
  const DURATION_SEC = 60;
  const N_CONV_RUNS = Math.min(N_RUNS, 5); // cap convergence runs (each is 60s)
  const allCurves = [];

  for (let r = 0; r < N_CONV_RUNS; r++) {
    await thermalGate('convergence_' + r);
    const browser = await freshBrowser();
    try {
      _benchPages['conv'] = generateConvergenceHTML(4096, 100, DURATION_SEC);
      process.stdout.write(`    run ${r + 1}/${N_CONV_RUNS} (${DURATION_SEC}s)...`);
      const result = await runBenchPage(browser, `http://127.0.0.1:${BENCH_PORT}/conv`, (DURATION_SEC + 30) * 1000);
      const finalFit = result.curve[result.curve.length - 1]?.bestFit ?? NaN;
      console.log(` ${result.totalGens} gens, best=${finalFit.toFixed(4)}`);
      allCurves.push(result.curve);
    } catch (e) {
      console.log(` FAILED: ${e.message.slice(0, 40)}`);
      allCurves.push([]);
    } finally {
      await closeBrowser(browser);
    }
  }

  return { id: 'convergence_curve', label: 'Convergence Curve (Rastrigin D=100)', durationSec: DURATION_SEC, runs: allCurves };
}

// ── PERF: Wall-clock quality comparison (WebGPU vs PyTorch, same time budget)
async function benchQualityComparison() {
  const DURATION_SEC = 60;
  const DIM = 100;
  const POP = 4096;

  // WebGPU run
  process.stdout.write(`    WebGPU (${DURATION_SEC}s)...`);
  await thermalGate('quality_webgpu');
  let webgpuBest = NaN, webgpuGens = 0;
  const browser = await freshBrowser();
  try {
    _benchPages['quality'] = generateConvergenceHTML(POP, DIM, DURATION_SEC);
    const result = await runBenchPage(browser, `http://127.0.0.1:${BENCH_PORT}/quality`, (DURATION_SEC + 30) * 1000);
    webgpuGens = result.totalGens;
    webgpuBest = result.curve[result.curve.length - 1]?.bestFit ?? NaN;
    console.log(` ${webgpuGens} gens, best=${webgpuBest.toFixed(4)}`);
  } catch (e) {
    console.log(` FAILED: ${e.message.slice(0, 40)}`);
  } finally {
    await closeBrowser(browser);
  }

  // PyTorch MPS run (same time budget, same problem)
  process.stdout.write(`    PyTorch MPS (${DURATION_SEC}s)...`);
  await thermalGate('quality_pytorch');
  const pyScript = `
import torch, time, math
D=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
POP=${POP};DIM=${DIM};PI=math.pi;DUR=${DURATION_SEC}
pop=(torch.rand(POP,DIM,device=D)*10.24-5.12).float()
def fit(x):return -(10*DIM+(x**2-10*torch.cos(2*PI*x)).sum(1))
def evolve(pop,f):
  idx=torch.randint(0,POP,(POP,4),device=D)
  tf=f[idx];best_of_4=tf.argmax(1)
  p1_idx=idx[torch.arange(POP,device=D),best_of_4]
  idx2=torch.randint(0,POP,(POP,4),device=D)
  tf2=f[idx2];best_of_4b=tf2.argmax(1)
  p2_idx=idx2[torch.arange(POP,device=D),best_of_4b]
  p1=pop[p1_idx];p2=pop[p2_idx]
  mask=(torch.rand(POP,DIM,device=D)<0.5).float()
  child=p1*mask+p2*(1-mask)+(torch.randn(POP,DIM,device=D)*0.3)
  child[0]=pop[f.argmax()]
  return child
for _ in range(50):f=fit(pop);pop=evolve(pop,f)
if D.type=='mps':torch.mps.synchronize()
t0=time.perf_counter();gc=0;curve=[]
while time.perf_counter()-t0<DUR:
  for _ in range(20):f=fit(pop);pop=evolve(pop,f);gc+=1
  if D.type=='mps':torch.mps.synchronize()
  curve.append(f'{time.perf_counter()-t0:.2f},{gc},{f.max().item():.6f}')
if D.type=='mps':torch.mps.synchronize()
elapsed=time.perf_counter()-t0
bf=fit(pop).max().item()
print(f'GENS={gc} BEST={bf:.6f} ELAPSED={elapsed:.2f}')
for c in curve:print(f'CURVE:{c}')
`;
  let pytorchBest = NaN, pytorchGens = 0, pytorchCurve = [];
  const tmpPy = path.join(require('os').tmpdir(), '_swarm_quality_cmp.py');
  fs.writeFileSync(tmpPy, pyScript);
  try {
    const out = execSync(`python3 "${tmpPy}"`, {
      encoding: 'utf-8', timeout: (DURATION_SEC + 60) * 1000,
      cwd: path.join(__dirname, '..'), stdio: ['pipe', 'pipe', 'pipe'],
    });
    const gensM = out.match(/GENS=(\d+)/);
    const bestM = out.match(/BEST=([-\d.]+)/);
    if (gensM) pytorchGens = parseInt(gensM[1]);
    if (bestM) pytorchBest = parseFloat(bestM[1]);
    pytorchCurve = out.split('\n').filter(l => l.startsWith('CURVE:')).map(l => {
      const [t, g, f] = l.slice(6).split(',');
      return { t: parseFloat(t), gen: parseInt(g), bestFit: parseFloat(f) };
    });
    console.log(` ${pytorchGens} gens, best=${pytorchBest.toFixed(4)}`);
  } catch (e) {
    console.log(` FAILED: ${(e.stderr || e.message).slice(0, 60)}`);
  }

  return {
    id: 'quality_comparison', label: `Wall-Clock Quality (${DURATION_SEC}s, Rastrigin D=${DIM})`,
    durationSec: DURATION_SEC, pop: POP, dim: DIM,
    webgpu: { gens: webgpuGens, bestFit: webgpuBest },
    pytorch: { gens: pytorchGens, bestFit: pytorchBest, curve: pytorchCurve },
    winner: (isNaN(webgpuBest) && isNaN(pytorchBest)) ? 'tie' : isNaN(pytorchBest) ? 'webgpu' : isNaN(webgpuBest) ? 'pytorch' : webgpuBest > pytorchBest ? 'webgpu' : 'pytorch',
  };
}

// ── CORRECTNESS: GPU/CPU parity ───────────────────────────────────────────
async function testGPUParity() {
  process.stdout.write('    Python CPU parity...');
  const pyOut = runPython('python3 tests/test_gpu_cpu_parity.py');
  const pyPass = (pyOut.match(/PASSED/g) || []).length;
  const pyFail = (pyOut.match(/FAILED/g) || []).length;
  console.log(` ${pyPass} passed, ${pyFail} failed`);

  process.stdout.write('    WebGPU GPU parity...');
  let gpuPass = 0, gpuFail = 0;
  try {
    const gpuOut = execSync('node tests/test_gpu_parity_puppeteer.js', {
      encoding: 'utf-8', timeout: 60000, cwd: path.join(__dirname, '..'),
    });
    gpuPass = (gpuOut.match(/PASSED|✅/g) || []).length;
    gpuFail = (gpuOut.match(/FAILED|❌/g) || []).length;
    console.log(` ${gpuPass} passed, ${gpuFail} failed`);
  } catch (e) {
    console.log(` ERROR: ${e.message.slice(0, 40)}`);
  }

  return { id: 'gpu_parity', label: 'GPU/CPU Parity', python: { passed: pyPass, failed: pyFail }, gpu: { passed: gpuPass, failed: gpuFail } };
}

// ── CORRECTNESS: f32 precision ────────────────────────────────────────────
async function testF32Precision() {
  process.stdout.write('    f32 precision...');
  const out = runPython('python3 benchmarks/f32_precision_analysis.py');
  const rho = out.match(/Ranking correlation.*?([\d.]+)/);
  const agreement = out.match(/Tournament agreement.*?([\d.]+)/);
  const nnErr = out.match(/NN forward max error.*?([\d.e+-]+)/);
  console.log(` rho=${rho?.[1]||'?'}, agree=${agreement?.[1]||'?'}%`);
  return {
    id: 'f32_precision', label: 'f32 Precision',
    rho: rho ? parseFloat(rho[1]) : null,
    tournament_agreement: agreement ? parseFloat(agreement[1]) : null,
    nn_max_error: nnErr?.[1] || null,
  };
}

// ── BASELINES: Python/JAX/PyTorch ─────────────────────────────────────────
async function benchBaselines() {
  const results = {};

  process.stdout.write('    NumPy baseline...');
  const npOut = runPython('python3 benchmarks/python_baseline.py', 600000);
  const npGps = npOut.match(/Python gen\/sec:\s*([\d.]+)/);
  results.numpy = npGps ? parseFloat(npGps[1]) : null;
  console.log(` ${results.numpy || 'FAILED'} gen/s`);

  process.stdout.write('    JAX baseline...');
  const jaxOut = runPython('python3 benchmarks/jax_baseline.py', 600000);
  const jaxRast = jaxOut.match(/Rastrigin gen\/sec:\s*([\d.]+)/);
  const jaxFin = jaxOut.match(/Financial gen\/sec:\s*([\d.]+)/);
  results.jax_rastrigin = jaxRast ? parseFloat(jaxRast[1]) : null;
  results.jax_financial = jaxFin ? parseFloat(jaxFin[1]) : null;
  console.log(` rast=${results.jax_rastrigin}, fin=${results.jax_financial}`);

  process.stdout.write('    PyTorch baseline (eager only)...');
  // Only run eager mode to avoid torch.compile OOM
  const ptOut = runPython('python3 -c "' +
    "import torch,time;D=torch.device('mps' if torch.backends.mps.is_available() else 'cpu');" +
    "POP=4096;DIM=2000;pop=(torch.rand(POP,DIM,device=D)*10.24-5.12).float();" +
    "import math;PI=math.pi;" +
    "def f(x):return -(10*DIM+(x**2-10*torch.cos(2*PI*x)).sum(1));" +
    "[f(pop) for _ in range(10)];torch.mps.synchronize() if D.type=='mps' else None;" +
    "t=time.perf_counter();[f(pop) for _ in range(100)];torch.mps.synchronize() if D.type=='mps' else None;" +
    "e=time.perf_counter()-t;print(f'{100/e:.1f}')" + '"', 120000);
  const ptGps = ptOut.match(/([\d.]+)/);
  results.pytorch_rastrigin = ptGps ? parseFloat(ptGps[1]) : null;
  console.log(` ${results.pytorch_rastrigin || 'FAILED'} gen/s`);

  // Published external baselines (for paper context, not measured here)
  results.external_references = {
    evojax: {
      note: 'EvoJAX (Tang et al., 2022) — JAX-based, GPU-accelerated ES',
      hardware: 'NVIDIA V100',
      rastrigin_gps: 'Not directly reported (focuses on RL tasks)',
      url: 'https://github.com/google/evojax',
    },
    evotorch: {
      note: 'EvoTorch (Toklu et al., 2023) — PyTorch-based, multi-GPU ES',
      hardware: 'NVIDIA A100',
      rastrigin_gps: 'Not directly reported (focuses on scalability)',
      url: 'https://github.com/nnaisense/evotorch',
    },
    openai_es: {
      note: 'OpenAI ES (Salimans et al., 2017) — distributed CPU ES',
      hardware: '720 CPUs',
      throughput: '~1B params/hour (not gen/s comparable)',
    },
    note: 'Direct gen/s comparison with these frameworks is not possible due to different problem encodings, hardware, and focus areas. Our baselines (NumPy, JAX CPU, PyTorch MPS) provide same-hardware apples-to-apples comparison.',
  };

  return { id: 'baselines', label: 'CPU/GPU Baselines', ...results };
}

// ═══════════════════════════════════════════════════════════════════════════
//  SUITE DEFINITIONS
// ═══════════════════════════════════════════════════════════════════════════

const SUITES = {
  perf: {
    label: 'Performance',
    tests: [
      { name: 'Rastrigin Throughput', fn: benchRastriginThroughput },
      { name: 'Financial Throughput', fn: benchFinancialThroughput },
      { name: 'Population Scaling', fn: benchPopScaling },
      { name: 'Multi-Fitness', fn: benchMultiFitness },
      { name: 'Genome Scaling', fn: benchGenomeScaling },
      { name: 'Tab Contention', fn: benchTabContention },
      { name: 'GPU Utilization', fn: benchGPUUtil },
      { name: 'Convergence Curve', fn: benchConvergenceCurve },
      { name: 'Quality Comparison', fn: benchQualityComparison },
    ],
  },
  correctness: {
    label: 'Correctness',
    tests: [
      { name: 'GPU/CPU Parity', fn: testGPUParity },
      { name: 'f32 Precision', fn: testF32Precision },
    ],
  },
  baselines: {
    label: 'Baselines (CPU/GPU)',
    tests: [
      { name: 'Baselines', fn: benchBaselines },
    ],
  },
};

// ═══════════════════════════════════════════════════════════════════════════
//  COMPARE & HISTORY
// ═══════════════════════════════════════════════════════════════════════════

function getResultFiles() {
  if (!fs.existsSync(RESULTS_DIR)) return [];
  return fs.readdirSync(RESULTS_DIR)
    .filter(f => f.endsWith('.json'))
    .sort()
    .map(f => path.join(RESULTS_DIR, f));
}

function showHistory() {
  const files = getResultFiles();
  if (!files.length) { console.log('No results yet.'); return; }
  console.log('\n  Run History:');
  console.log('  ' + '─'.repeat(70));
  for (const f of files) {
    const data = JSON.parse(fs.readFileSync(f, 'utf-8'));
    const tests = Object.keys(data.results || {}).length;
    console.log(`  ${path.basename(f, '.json').padEnd(40)} | ${data.git?.hash || '?'} | ${tests} tests`);
  }
}

function compareRuns() {
  const files = getResultFiles();
  if (files.length < 2) { console.log('Need at least 2 runs to compare.'); return; }
  const a = JSON.parse(fs.readFileSync(files[files.length - 2], 'utf-8'));
  const b = JSON.parse(fs.readFileSync(files[files.length - 1], 'utf-8'));

  console.log('\n  ═══ COMPARISON ═══');
  console.log(`  A: ${a.git?.hash || '?'} (${a.timestamp})`);
  console.log(`  B: ${b.git?.hash || '?'} (${b.timestamp})`);
  console.log('  ' + '─'.repeat(60));

  // Compare key metrics
  const metrics = [
    ['Rastrigin g/s', 'rastrigin_throughput', 'mean'],
    ['Financial g/s', 'financial_throughput', 'mean'],
  ];
  for (const [label, key, field] of metrics) {
    const va = a.results?.[key]?.[field];
    const vb = b.results?.[key]?.[field];
    if (va != null && vb != null) {
      const pct = ((vb - va) / va * 100).toFixed(1);
      const arrow = vb > va ? '▲' : vb < va ? '▼' : '=';
      console.log(`  ${label.padEnd(20)} ${String(va.toFixed(1)).padStart(8)} → ${String(vb.toFixed(1)).padStart(8)}  ${arrow} ${pct}%`);
    }
  }
}

// ═══════════════════════════════════════════════════════════════════════════
//  MAIN
// ═══════════════════════════════════════════════════════════════════════════

async function main() {
  if (args.history) { showHistory(); return; }
  if (args.compare) { compareRuns(); return; }

  const git = getGitInfo();
  const chip = getChipModel();
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);

  console.log('');
  console.log('  ╔══════════════════════════════════════════════════════╗');
  console.log('  ║           SWARM BENCHMARK HARNESS                   ║');
  console.log('  ╚══════════════════════════════════════════════════════╝');
  console.log(`  Git:      ${git.hash} (${git.branch})`);
  console.log(`  Hardware: ${chip}`);
  console.log(`  Suite:    ${SUITE}`);
  console.log(`  Runs:     ${N_RUNS} per benchmark`);
  console.log(`  Time:     ${timestamp}`);
  console.log('');

  // Determine which suites to run
  const suitesToRun = SUITE === 'all'
    ? Object.values(SUITES)
    : SUITES[SUITE] ? [SUITES[SUITE]] : [];

  if (!suitesToRun.length) {
    console.log(`  Unknown suite: ${SUITE}`);
    console.log(`  Available: all, ${Object.keys(SUITES).join(', ')}`);
    return;
  }

  await startBenchServer();
  const thermal = new ThermalMonitor(5000);
  thermal.start();
  const allResults = {};

  for (const suite of suitesToRun) {
    console.log(`\n  ── ${suite.label} ${'─'.repeat(50 - suite.label.length)}`);
    for (const test of suite.tests) {
      console.log(`\n  ${test.name}:`);
      try {
        const result = await test.fn();
        allResults[result.id] = result;
      } catch (e) {
        console.log(`    ERROR: ${e.message}`);
        allResults[test.name] = { error: e.message };
      }
    }
  }

  const thermalReport = thermal.stop();
  if (_benchServer) _benchServer.close();

  // Save
  if (!fs.existsSync(RESULTS_DIR)) fs.mkdirSync(RESULTS_DIR, { recursive: true });
  const output = {
    timestamp: new Date().toISOString(),
    git,
    hardware: chip,
    suite: SUITE,
    n_runs: N_RUNS,
    results: allResults,
    thermal: {
      chipModel: thermalReport.chipModel,
      temperature: thermalReport.temperature,
      gpuUtilization: thermalReport.gpuUtilization,
      thermalThrottled: thermalReport.thermalThrottled,
    },
  };

  const filename = `${timestamp}_${git.hash}.json`;
  const outPath = path.join(RESULTS_DIR, filename);
  fs.writeFileSync(outPath, JSON.stringify(output, null, 2));

  // Print summary
  console.log('\n  ╔══════════════════════════════════════════════════════╗');
  console.log('  ║                    SUMMARY                          ║');
  console.log('  ╚══════════════════════════════════════════════════════╝');

  for (const [key, val] of Object.entries(allResults)) {
    if (val.mean != null) {
      const ci = val.ci95 || [NaN, NaN];
      console.log(`  ${(val.label || key).padEnd(30)} ${val.mean.toFixed(1)} ± ${(val.std || 0).toFixed(1)} ${val.unit || ''}  [95% CI: ${ci[0].toFixed(1)}–${ci[1].toFixed(1)}]  (n=${val.n})`);
    } else if (val.results) {
      console.log(`  ${(val.label || key)}:`);
      for (const r of val.results) {
        const label = r.name || (r.pop != null ? `POP=${r.pop}` : r.dim != null ? `DIM=${r.dim}` : r.tabs != null ? `${r.tabs} tabs` : '?');
        console.log(`    ${String(label).padEnd(12)} ${(r.gps || r.perTab || 0).toFixed(1)} gen/s`);
      }
    } else if (val.id === 'quality_comparison') {
      console.log(`  ${val.label}:`);
      console.log(`    WebGPU:  ${val.webgpu.gens} gens, best=${val.webgpu.bestFit?.toFixed(4) || 'N/A'}`);
      console.log(`    PyTorch: ${val.pytorch.gens} gens, best=${val.pytorch.bestFit?.toFixed(4) || 'N/A'}`);
      console.log(`    Winner:  ${val.winner}`);
    } else if (val.id === 'convergence_curve') {
      console.log(`  ${val.label}: ${val.runs.length} runs, ${val.durationSec}s each`);
    } else if (val.id === 'gpu_util') {
      console.log(`  ${val.label}:`);
      console.log(`    Idle:  ${val.idle.mean.toFixed(1)}% ± ${val.idle.std.toFixed(1)}%`);
      console.log(`    Load:  ${val.load.mean.toFixed(1)}% ± ${val.load.std.toFixed(1)}%  (${val.speed.mean.toFixed(1)} gen/s)`);
      console.log(`    Temp:  ${val.temp || 'N/A'}°C`);
    }
  }

  // Statistical comparison (WebGPU vs PyTorch throughput)
  const rast = allResults.rastrigin_throughput;
  const baselines = allResults.baselines;
  if (rast?.raw && baselines?.pytorch_rastrigin) {
    const ptRaw = Array(rast.raw.length).fill(baselines.pytorch_rastrigin); // single-point baseline
    const test = welchTTest(rast.raw, ptRaw);
    console.log(`\n  ── Statistical Significance ──`);
    console.log(`  WebGPU vs PyTorch (Rastrigin): t=${test.t.toFixed(2)}, p≈${test.p < 0.001 ? '<0.001' : test.p.toFixed(3)}, ${test.significant ? 'SIGNIFICANT' : 'not significant'}`);
  }

  // Seed control notice
  console.log(`\n  Note: WebGPU RNG is non-deterministic (Math.random seed).`);
  console.log(`  Results are stochastic; N=${N_RUNS} runs with CI reported for reproducibility.`);

  console.log(`\n  Saved: ${outPath}`);
  console.log(`  Compare: node bench.js --compare`);
  console.log(`  History: node bench.js --history`);
  console.log('');
}

main().catch(err => {
  console.error('Fatal:', err.message);
  process.exit(1);
});
