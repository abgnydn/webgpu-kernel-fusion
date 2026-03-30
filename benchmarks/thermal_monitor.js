#!/usr/bin/env node
/**
 * Thermal & GPU Monitor — captures hardware metrics during benchmarks.
 * No sudo required. Works on Apple Silicon Macs.
 *
 * Captures:
 *   - Battery temperature (proxy for SoC temp)
 *   - GPU utilization (Device, Tiler, Renderer %)
 *   - Thermal warning state (pmset)
 *   - Timestamps for correlation with benchmark events
 */

const { execSync } = require('child_process');

function getBatteryTemp() {
  try {
    const out = execSync('ioreg -r -n AppleSmartBattery 2>/dev/null', { encoding: 'utf-8' });
    const m = out.match(/"Temperature"\s*=\s*(\d+)/);
    return m ? parseInt(m[1]) / 100 : null;  // centi-degrees → °C
  } catch { return null; }
}

function getGPUUtilization() {
  try {
    const out = execSync('ioreg -r -d 1 -c IOAccelerator 2>/dev/null', { encoding: 'utf-8' });
    const device   = out.match(/"Device Utilization %"\s*=\s*(\d+)/);
    const tiler    = out.match(/"Tiler Utilization %"\s*=\s*(\d+)/);
    const renderer = out.match(/"Renderer Utilization %"\s*=\s*(\d+)/);
    return {
      device:   device   ? parseInt(device[1])   : null,
      tiler:    tiler    ? parseInt(tiler[1])     : null,
      renderer: renderer ? parseInt(renderer[1])  : null,
    };
  } catch { return { device: null, tiler: null, renderer: null }; }
}

function getThermalPressure() {
  try {
    const out = execSync('pmset -g therm 2>/dev/null', { encoding: 'utf-8' });
    if (out.includes('No thermal warning')) return 'nominal';
    if (out.includes('moderate'))           return 'moderate';
    if (out.includes('heavy'))              return 'heavy';
    if (out.includes('trapping'))           return 'trapping';
    if (out.includes('sleeping'))           return 'sleeping';
    return 'unknown';
  } catch { return 'unknown'; }
}

function getChipModel() {
  try {
    const out = execSync('sysctl -n machdep.cpu.brand_string 2>/dev/null', { encoding: 'utf-8' }).trim();
    if (out) return out;
    // Fallback: system_profiler
    const sp = execSync("system_profiler SPHardwareDataType 2>/dev/null | grep 'Chip'", { encoding: 'utf-8' }).trim();
    return sp.replace(/.*:\s*/, '') || 'unknown';
  } catch { return 'unknown'; }
}

function getMemoryPressure() {
  try {
    const out = execSync('memory_pressure 2>/dev/null | head -1', { encoding: 'utf-8' }).trim();
    const m = out.match(/(\d+)%/);
    return m ? parseInt(m[1]) : null;
  } catch { return null; }
}

function snapshot() {
  return {
    timestamp:    Date.now(),
    iso:          new Date().toISOString(),
    batteryTempC: getBatteryTemp(),
    gpu:          getGPUUtilization(),
    thermal:      getThermalPressure(),
  };
}

class ThermalMonitor {
  constructor(intervalMs = 2000) {
    this.intervalMs = intervalMs;
    this.samples = [];
    this._timer = null;
    this.chipModel = getChipModel();
  }

  start() {
    this.samples = [];
    this._sample(); // immediate first sample
    this._timer = setInterval(() => this._sample(), this.intervalMs);
  }

  stop() {
    if (this._timer) clearInterval(this._timer);
    this._timer = null;
    this._sample(); // final sample
    return this.report();
  }

  _sample() {
    this.samples.push(snapshot());
  }

  report() {
    if (!this.samples.length) return null;

    const temps = this.samples.map(s => s.batteryTempC).filter(t => t !== null);
    const gpuDevice = this.samples.map(s => s.gpu.device).filter(v => v !== null);
    const thermalStates = this.samples.map(s => s.thermal);
    const throttled = thermalStates.some(t => t !== 'nominal');

    return {
      chipModel:     this.chipModel,
      durationMs:    this.samples[this.samples.length - 1].timestamp - this.samples[0].timestamp,
      nSamples:      this.samples.length,
      intervalMs:    this.intervalMs,
      temperature: {
        unit:    '°C (battery, SoC proxy)',
        min:     temps.length ? Math.min(...temps) : null,
        max:     temps.length ? Math.max(...temps) : null,
        mean:    temps.length ? +(temps.reduce((a, b) => a + b, 0) / temps.length).toFixed(1) : null,
        start:   temps[0] || null,
        end:     temps[temps.length - 1] || null,
      },
      gpuUtilization: {
        unit:    '% (Device Utilization)',
        min:     gpuDevice.length ? Math.min(...gpuDevice) : null,
        max:     gpuDevice.length ? Math.max(...gpuDevice) : null,
        mean:    gpuDevice.length ? +(gpuDevice.reduce((a, b) => a + b, 0) / gpuDevice.length).toFixed(1) : null,
      },
      thermalThrottled: throttled,
      thermalStates:    [...new Set(thermalStates)],
      samples:          this.samples,
    };
  }
}

module.exports = { ThermalMonitor, snapshot, getChipModel, getBatteryTemp, getGPUUtilization, getThermalPressure };

// CLI: run standalone to check current state
if (require.main === module) {
  console.log('Chip:', getChipModel());
  console.log('Snapshot:', JSON.stringify(snapshot(), null, 2));
}
