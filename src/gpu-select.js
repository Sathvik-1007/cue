// Pick the best Vulkan device for whisper.cpp. Machines with a discrete GPU
// plus an integrated one (very common laptops) enumerate the iGPU as device 0,
// and ggml uses device 0 by default — measured: large-v3-turbo on the Intel iGPU
// took 29s per 11s of audio vs 1.4s on the NVIDIA GPU beside it. So we prefer
// a discrete GPU. Pure logic here; the caller feeds it `vulkaninfo --summary`
// text (or ggml's own "ggml_vulkan: N = <name>" lines) and gets a device index.
const { execFileSync } = require('child_process');

const DISCRETE_RE = /nvidia|geforce|quadro|rtx|gtx|radeon rx|radeon pro|amd radeon (?!graphics)|arc a\d|arc b\d/i;
const INTEGRATED_RE = /intel\(r\) (uhd|hd|iris)|uhd graphics|iris|amd radeon graphics|radeon vega|integrated|llvmpipe|swiftshader/i;

// Parse device names, in index order, from either source format.
function parseVulkanDevices(text) {
  const names = [];
  const src = String(text || '');
  // ggml format: "ggml_vulkan: 0 = NVIDIA GeForce GTX 1650 Ti (NVIDIA) | uma: 0 ..."
  for (const m of src.matchAll(/ggml_vulkan:\s*(\d+)\s*=\s*([^|\n]+?)\s*(?:\||\n|$)/g)) names[+m[1]] = m[2].trim();
  if (names.length) return names.filter(Boolean);
  // vulkaninfo --summary format: "deviceName = NVIDIA GeForce GTX 1650 Ti"
  for (const m of src.matchAll(/deviceName\s*=\s*(.+)/g)) names.push(m[1].trim());
  return names;
}

// Return the index to expose via GGML_VK_VISIBLE_DEVICES, or null to leave the
// default. Prefers: discrete GPU > anything not integrated/software > device 0.
function pickDeviceIndex(names) {
  if (!names || names.length < 2) return null; // one device: nothing to choose
  const disc = names.findIndex((n) => DISCRETE_RE.test(n) && !INTEGRATED_RE.test(n));
  if (disc >= 0) return disc;
  const notIgpu = names.findIndex((n) => !INTEGRATED_RE.test(n));
  return notIgpu >= 0 ? notIgpu : null;
}

let cached; // probe once per process
function probeVulkanDevices() {
  if (cached !== undefined) return cached;
  try {
    const out = execFileSync('vulkaninfo', ['--summary'], { encoding: 'utf8', timeout: 5000, stdio: ['ignore', 'pipe', 'ignore'] });
    cached = parseVulkanDevices(out);
  } catch (_) { cached = []; }
  return cached;
}

// Env additions for the whisper-server child, e.g. { GGML_VK_VISIBLE_DEVICES: '1' }.
function vulkanEnvForBestDevice() {
  const idx = pickDeviceIndex(probeVulkanDevices());
  return idx === null ? {} : { GGML_VK_VISIBLE_DEVICES: String(idx) };
}

module.exports = { parseVulkanDevices, pickDeviceIndex, probeVulkanDevices, vulkanEnvForBestDevice };
