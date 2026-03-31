/**
 * Paper Arithmetic Verification
 *
 * Every derived number (ratio, comparison, percentage) in the paper
 * must be traceable to the raw data in the tables. This test catches
 * copy-paste errors, stale numbers, and hallucinated calculations.
 *
 * Run: node tests/paper_arithmetic.test.js
 */

let failures = 0;
let passes = 0;

function check(name, actual, expected, tolerance = 0.05) {
  const ratio = Math.abs(actual - expected) / Math.max(Math.abs(expected), 0.001);
  if (ratio > tolerance) {
    console.error(`FAIL: ${name}`);
    console.error(`  Expected: ${expected}, Got: ${actual}, Off by: ${(ratio * 100).toFixed(1)}%`);
    failures++;
  } else {
    console.log(`  OK: ${name} = ${actual} (expected ${expected})`);
    passes++;
  }
}

function checkExact(name, actual, expected) {
  if (actual !== expected) {
    console.error(`FAIL: ${name}`);
    console.error(`  Expected: ${expected}, Got: ${actual}`);
    failures++;
  } else {
    console.log(`  OK: ${name} = ${actual}`);
    passes++;
  }
}

// ═══════════════════════════════════════════
// RAW DATA FROM TABLES (source of truth)
// ═══════════════════════════════════════════

const TABLE1 = {
  numpy:      3.9,
  jax_cpu:    20.6,
  pytorch_mps: 160.5,
  webgpu:     170.3,
  wgpu_native: 326.5,
  pytorch_cuda: 311.1,
  jax_gpu:    1163.9,
};

const TABLE2 = {
  numpy:      0.056,
  jax_cpu:    0.157,
  pytorch_mps: 0.29,
  pytorch_cuda: 0.49,
  jax_gpu:    6.43,
  webgpu:     46.2,
};

const TABLE3 = {
  pytorch_cuda: 0.61,
  pytorch_mps:  2.52,
  webgpu_unfused: 62.3,
  jax_gpu:    105.1,
  webgpu_fused: 135.9,
};

const TABLE6 = {
  pop_512:   12948,
  pop_1024:  12791,
  pop_2048:  12842,
  pop_4096:  12685,
  pop_8192:  12714,
  pop_16384: 12741,
  pop_32768: 12702,
};

const TABLE10 = {
  tabs_1_per: 11607, tabs_1_total: 11607,
  tabs_2_per: 11296, tabs_2_total: 22593,
  tabs_4_per:  9727, tabs_4_total: 38909,
  tabs_8_per:  9314, tabs_8_total: 74512,
};

const TABLE11 = {
  pytorch_mps: 18.7,
  webgpu: 1258.8,
};

// ═══════════════════════════════════════════
// ABSTRACT CLAIMS
// ═══════════════════════════════════════════

console.log("\n=== ABSTRACT ===");

check("Abstract: WebGPU 7.2x over JAX GPU (financial)",
  TABLE2.webgpu / TABLE2.jax_gpu, 7.2, 0.05);

check("Abstract: WebGPU 94x over PyTorch CUDA (financial)",
  TABLE2.webgpu / TABLE2.pytorch_cuda, 94, 0.05);

check("Abstract: Acrobot 1.29x over JAX GPU",
  TABLE3.webgpu_fused / TABLE3.jax_gpu, 1.29, 0.05);

check("Abstract: JAX GPU 1164 vs WebGPU 170 on Rastrigin",
  TABLE1.jax_gpu / TABLE1.webgpu, 6.8, 0.05);

check("Abstract: native Metal 1.92x over WebGPU",
  TABLE1.wgpu_native / TABLE1.webgpu, 1.92, 0.02);

// ═══════════════════════════════════════════
// CONTRIBUTIONS (C1)
// ═══════════════════════════════════════════

console.log("\n=== CONTRIBUTIONS ===");

check("C1: 159x over PyTorch MPS (financial)",
  TABLE2.webgpu / TABLE2.pytorch_mps, 159, 0.05);

check("C1: 94x over PyTorch CUDA (financial)",
  TABLE2.webgpu / TABLE2.pytorch_cuda, 94, 0.05);

check("C1: 223x over CUDA (Acrobot)",
  TABLE3.webgpu_fused / TABLE3.pytorch_cuda, 223, 0.05);

// ═══════════════════════════════════════════
// TABLE 1 (Rastrigin) — "vs NumPy" column
// ═══════════════════════════════════════════

console.log("\n=== TABLE 1: vs NumPy ===");

check("Table 1: JAX CPU 5.3x over NumPy",
  TABLE1.jax_cpu / TABLE1.numpy, 5.3, 0.05);

check("Table 1: PyTorch MPS 41x over NumPy",
  TABLE1.pytorch_mps / TABLE1.numpy, 41, 0.05);

check("Table 1: WebGPU 44x over NumPy",
  TABLE1.webgpu / TABLE1.numpy, 44, 0.05);

check("Table 1: wgpu-native 84x over NumPy",
  TABLE1.wgpu_native / TABLE1.numpy, 84, 0.05);

check("Table 1: PyTorch CUDA 80x over NumPy",
  TABLE1.pytorch_cuda / TABLE1.numpy, 80, 0.05);

check("Table 1: JAX GPU 298x over NumPy",
  TABLE1.jax_gpu / TABLE1.numpy, 298, 0.05);

// ═══════════════════════════════════════════
// TABLE 1 — inline claims
// ═══════════════════════════════════════════

console.log("\n=== TABLE 1: Inline claims ===");

check("WebGPU 1.06x parity with PyTorch MPS",
  TABLE1.webgpu / TABLE1.pytorch_mps, 1.06, 0.02);

check("JAX GPU 6.8x over WebGPU",
  TABLE1.jax_gpu / TABLE1.webgpu, 6.8, 0.05);

check("PyTorch CUDA 1.83x over WebGPU",
  TABLE1.pytorch_cuda / TABLE1.webgpu, 1.83, 0.05);

check("WebGPU 0.55x of PyTorch CUDA",
  TABLE1.webgpu / TABLE1.pytorch_cuda, 0.55, 0.05);

check("Browser overhead 48% (1 - webgpu/native)",
  (1 - TABLE1.webgpu / TABLE1.wgpu_native) * 100, 48, 0.05);

// ═══════════════════════════════════════════
// TABLE 2 (Financial) — "vs" columns
// ═══════════════════════════════════════════

console.log("\n=== TABLE 2: Financial ratios ===");

check("Table 2: JAX CPU 2.8x over NumPy",
  TABLE2.jax_cpu / TABLE2.numpy, 2.8, 0.05);

check("Table 2: PyTorch MPS 5.2x over NumPy",
  TABLE2.pytorch_mps / TABLE2.numpy, 5.2, 0.05);

check("Table 2: PyTorch CUDA 8.8x over NumPy",
  TABLE2.pytorch_cuda / TABLE2.numpy, 8.8, 0.05);

check("Table 2: JAX GPU 115x over NumPy",
  TABLE2.jax_gpu / TABLE2.numpy, 115, 0.05);

check("Table 2: WebGPU 825x over NumPy",
  TABLE2.webgpu / TABLE2.numpy, 825, 0.05);

check("Table 2: WebGPU 159x over PyTorch MPS",
  TABLE2.webgpu / TABLE2.pytorch_mps, 159, 0.05);

check("Table 2: WebGPU 94x over PyTorch CUDA",
  TABLE2.webgpu / TABLE2.pytorch_cuda, 94, 0.05);

check("Table 2: PyTorch CUDA 1.7x over PyTorch MPS",
  TABLE2.pytorch_cuda / TABLE2.pytorch_mps, 1.7, 0.05);

check("Table 2: JAX GPU 22x over PyTorch MPS",
  TABLE2.jax_gpu / TABLE2.pytorch_mps, 22, 0.05);

check("Table 2: JAX GPU 13x over PyTorch CUDA",
  TABLE2.jax_gpu / TABLE2.pytorch_cuda, 13, 0.05);

check("Table 2: WebGPU 7.2x over JAX GPU",
  TABLE2.webgpu / TABLE2.jax_gpu, 7.2, 0.05);

// ═══════════════════════════════════════════
// TABLE 3 (Acrobot) — ratios
// ═══════════════════════════════════════════

console.log("\n=== TABLE 3: Acrobot ratios ===");

check("Table 3: PyTorch MPS 4.1x over PyTorch CUDA",
  TABLE3.pytorch_mps / TABLE3.pytorch_cuda, 4.1, 0.05);

check("Table 3: WebGPU unfused 24.7x over PyTorch MPS",
  TABLE3.webgpu_unfused / TABLE3.pytorch_mps, 24.7, 0.05);

check("Table 3: WebGPU unfused 102x over PyTorch CUDA",
  TABLE3.webgpu_unfused / TABLE3.pytorch_cuda, 102, 0.05);

check("Table 3: JAX GPU 41.7x over PyTorch MPS",
  TABLE3.jax_gpu / TABLE3.pytorch_mps, 41.7, 0.05);

check("Table 3: JAX GPU 172x over PyTorch CUDA",
  TABLE3.jax_gpu / TABLE3.pytorch_cuda, 172, 0.05);

check("Table 3: WebGPU fused 54x over PyTorch MPS",
  TABLE3.webgpu_fused / TABLE3.pytorch_mps, 54, 0.05);

check("Table 3: WebGPU fused 223x over PyTorch CUDA",
  TABLE3.webgpu_fused / TABLE3.pytorch_cuda, 223, 0.05);

check("Ablation: fusion provides 2.18x on top of unfused",
  TABLE3.webgpu_fused / TABLE3.webgpu_unfused, 2.18, 0.05);

check("JAX GPU 1.29x slower than fused WebGPU (Acrobot)",
  TABLE3.webgpu_fused / TABLE3.jax_gpu, 1.29, 0.05);

// ═══════════════════════════════════════════
// TABLE 5 (Population scaling) — <2.1% variation
// ═══════════════════════════════════════════

console.log("\n=== TABLE 6: Population scaling ===");

const popValues = Object.values(TABLE6);
const popMin = Math.min(...popValues);
const popMax = Math.max(...popValues);
const popVariation = ((popMax - popMin) / popMax) * 100;

check("Population scaling <2.1% variation",
  popVariation, 2.0, 0.10);

// ═══════════════════════════════════════════
// TABLE 10 (Multi-tab) — efficiency claims
// ═══════════════════════════════════════════

console.log("\n=== TABLE 10: Multi-tab ===");

check("2-tab efficiency 97%",
  (TABLE10.tabs_2_per / TABLE10.tabs_1_per) * 100, 97, 0.02);

check("4-tab efficiency 84%",
  (TABLE10.tabs_4_per / TABLE10.tabs_1_per) * 100, 84, 0.02);

check("8-tab efficiency 80%",
  (TABLE10.tabs_8_per / TABLE10.tabs_1_per) * 100, 80, 0.02);

check("8-tab total throughput 6.4x",
  TABLE10.tabs_8_total / TABLE10.tabs_1_total, 6.4, 0.05);

check("4-tab total throughput 3.4x",
  TABLE10.tabs_4_total / TABLE10.tabs_1_total, 3.4, 0.05);

// ═══════════════════════════════════════════
// TABLE 11 (MountainCar)
// ═══════════════════════════════════════════

console.log("\n=== TABLE 11: MountainCar ===");

check("MountainCar: WebGPU 67x over PyTorch MPS",
  TABLE11.webgpu / TABLE11.pytorch_mps, 67, 0.05);

// ═══════════════════════════════════════════
// CONCLUSION claims
// ═══════════════════════════════════════════

console.log("\n=== CONCLUSION ===");

check("Conclusion: 7.2x over JAX GPU (financial)",
  TABLE2.webgpu / TABLE2.jax_gpu, 7.2, 0.05);

check("Conclusion: 94x over PyTorch CUDA (financial)",
  TABLE2.webgpu / TABLE2.pytorch_cuda, 94, 0.05);

check("Conclusion: 1.29x over JAX GPU (Acrobot)",
  TABLE3.webgpu_fused / TABLE3.jax_gpu, 1.29, 0.05);

check("Conclusion: JAX GPU 6.8x over WebGPU (Rastrigin)",
  TABLE1.jax_gpu / TABLE1.webgpu, 6.8, 0.05);

check("Conclusion: native Metal 1.92x",
  TABLE1.wgpu_native / TABLE1.webgpu, 1.92, 0.02);

// ═══════════════════════════════════════════
// CROSS-TABLE CONSISTENCY
// ═══════════════════════════════════════════

console.log("\n=== CROSS-TABLE CONSISTENCY ===");

// Abstract says "46.2 gen/s" — matches Table 2
checkExact("Abstract WebGPU financial matches Table 2", TABLE2.webgpu, 46.2);

// Abstract says "6.43 gen/s" for JAX GPU — matches Table 2
checkExact("Abstract JAX GPU financial matches Table 2", TABLE2.jax_gpu, 6.43);

// Abstract says "1,164" for JAX Rastrigin — check rounding
check("Abstract JAX GPU Rastrigin rounds to 1164",
  Math.round(TABLE1.jax_gpu), 1164, 0.001);

// ═══════════════════════════════════════════
// SUMMARY
// ═══════════════════════════════════════════

console.log(`\n${"=".repeat(50)}`);
console.log(`RESULTS: ${passes} passed, ${failures} failed`);
console.log(`${"=".repeat(50)}`);

if (failures > 0) {
  console.error(`\n${failures} ARITHMETIC ERROR(S) IN PAPER — FIX BEFORE SUBMISSION`);
  process.exit(1);
} else {
  console.log("\nAll paper arithmetic verified. Safe to submit.");
}
