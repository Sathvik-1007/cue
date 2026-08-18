const assert = require('node:assert/strict');
const test = require('node:test');

const builder = require('../electron-builder.cjs');
const pkg = require('../package.json');
const { parseSourcesShort, pickMonitorSource, pickNonBluetoothMic } = require('../src/linux-audio');
const kwin = require('../src/kwin-capture');
const hypr = require('../src/hypr-capture');

test('defines explicit Linux package targets', () => {
  assert.equal(pkg.scripts['pack:linux'], 'electron-builder --linux --dir');
  assert.equal(pkg.scripts['dist:linux'], 'electron-builder --linux AppImage --x64');
  assert.equal(pkg.scripts['dist:linux:arm64'], 'electron-builder --linux AppImage --arm64');
  assert.deepEqual(builder.linux.target, [{ target: 'AppImage', arch: ['x64', 'arm64'] }]);
});

test('parses pactl list sources short output', () => {
  const out = '51\talsa_output.pci-0000_00_1f.3.analog-stereo.monitor\tPipeWire\ts32le 2ch 48000Hz\tSUSPENDED\n' +
    '52\talsa_input.pci-0000_00_1f.3.analog-stereo\tPipeWire\ts32le 2ch 48000Hz\tSUSPENDED\n';
  const sources = parseSourcesShort(out);
  assert.equal(sources.length, 2);
  assert.equal(sources[0].name, 'alsa_output.pci-0000_00_1f.3.analog-stereo.monitor');
  assert.equal(sources[0].monitor, true);
  assert.equal(sources[1].monitor, false);
  assert.deepEqual(parseSourcesShort(''), []);
  assert.deepEqual(parseSourcesShort(null), []);
});

test('picks the monitor of the default sink', () => {
  const mon = { id: '51', name: 'alsa_output.internal.monitor', monitor: true };
  const usb = { id: '60', name: 'alsa_output.usb-dac.monitor', monitor: true };
  const mic = { id: '52', name: 'alsa_input.internal', monitor: false };
  // default sink wins over list order
  assert.equal(pickMonitorSource([mon, usb, mic], '', 'alsa_output.usb-dac'), usb);
  // stored preference wins over the default sink
  assert.equal(pickMonitorSource([mon, usb, mic], 'alsa_output.internal.monitor', 'alsa_output.usb-dac'), mon);
  // stored source unplugged → fall back to the default sink's monitor
  assert.equal(pickMonitorSource([mon, mic], 'alsa_output.usb-dac.monitor', 'alsa_output.internal'), mon);
  // no default-sink match → any monitor; a bare mic never qualifies
  assert.equal(pickMonitorSource([mic, usb], '', 'alsa_output.internal'), usb);
  assert.equal(pickMonitorSource([mic], '', ''), null);
  assert.equal(pickMonitorSource([], '', ''), null);
  assert.equal(pickMonitorSource(undefined, '', ''), null);
});

test('kwin-capture builds a correct exclusion rule', () => {
  const r = kwin.buildRule('MicrosoftEdgeUpdate');
  // Force-type rule, exact case-sensitive class match — verified live on Plasma 6.7.3.
  assert.equal(r.wmclass, 'MicrosoftEdgeUpdate');
  assert.equal(r.wmclassmatch, '1');            // exact
  assert.equal(r.excludefromcapture, 'true');
  assert.equal(r.excludefromcapturerule, '2');  // Force
});

test('kwin-capture merges the rules list without duplication', () => {
  const id = kwin.RULE_ID;
  assert.equal(kwin.mergeRulesList('', true), id);
  assert.equal(kwin.mergeRulesList('uuid-a,uuid-b', true), 'uuid-a,uuid-b,' + id);
  assert.equal(kwin.mergeRulesList('uuid-a,' + id, true), 'uuid-a,' + id); // idempotent
  assert.equal(kwin.mergeRulesList('uuid-a,' + id + ',uuid-b', false), 'uuid-a,uuid-b');
  assert.equal(kwin.mergeRulesList(id, false), '');
});

test('kwin-capture version floor is Plasma 6.6', () => {
  assert.equal(kwin.versionMeetsFloor(kwin.parseVersion('kwin 6.7.3')), true);
  assert.equal(kwin.versionMeetsFloor(kwin.parseVersion('6.6.0')), true);
  assert.equal(kwin.versionMeetsFloor(kwin.parseVersion('6.5.5')), false);
  assert.equal(kwin.versionMeetsFloor(kwin.parseVersion('7.0.0')), true);
  assert.equal(kwin.versionMeetsFloor(null), false);
  assert.equal(kwin.parseVersion('no version here'), null);
});

test('hypr-capture builds a noscreenshare rule and gates on 0.50', () => {
  assert.equal(hypr.buildRule('MicrosoftEdgeUpdate'), 'noscreenshare, class:^(MicrosoftEdgeUpdate)$');
  // noscreenshare landed in Hyprland 0.50.0
  assert.equal(hypr.versionMeetsFloor(hypr.parseVersion('Hyprland, built from ... tag: v0.50.1')), true);
  assert.equal(hypr.versionMeetsFloor(hypr.parseVersion('v0.50.0')), true);
  assert.equal(hypr.versionMeetsFloor(hypr.parseVersion('v0.49.0')), false);
  assert.equal(hypr.versionMeetsFloor(hypr.parseVersion('v1.0.0')), true);
  assert.equal(hypr.versionMeetsFloor(null), false);
  assert.equal(hypr.parseVersion('no version'), null);
});

test('pickNonBluetoothMic avoids flipping a Bluetooth headset to HFP', () => {
  const bt = { id: '1', name: 'bluez_input.88:0E:85:66:81:97', monitor: false };
  const wired = { id: '2', name: 'alsa_input.pci-0000_00_1f.3.analog-stereo', monitor: false };
  const mon = { id: '3', name: 'alsa_output.foo.monitor', monitor: true };
  // default is the BT headset -> use the wired mic instead
  assert.equal(pickNonBluetoothMic([bt, wired, mon], bt.name), wired.name);
  // default is already wired -> nothing to change
  assert.equal(pickNonBluetoothMic([bt, wired], wired.name), null);
  // BT default but no wired mic exists -> null (renderer keeps the default)
  assert.equal(pickNonBluetoothMic([bt, mon], bt.name), null);
  assert.equal(pickNonBluetoothMic([], ''), null);
});

test('gpu-select prefers a discrete GPU over the integrated one', () => {
  const g = require('../src/gpu-select');
  // ggml's own enumeration format (this machine): iGPU is device 0
  const ggml = 'ggml_vulkan: Found 2 Vulkan devices:\nggml_vulkan: 0 = Intel(R) UHD Graphics (CML GT2) (Intel open-source Mesa driver) | uma: 1 | fp16: 1\nggml_vulkan: 1 = NVIDIA GeForce GTX 1650 Ti (NVIDIA) | uma: 0 | fp16: 1\n';
  const names = g.parseVulkanDevices(ggml);
  assert.deepEqual(names, ['Intel(R) UHD Graphics (CML GT2) (Intel open-source Mesa driver)', 'NVIDIA GeForce GTX 1650 Ti (NVIDIA)']);
  assert.equal(g.pickDeviceIndex(names), 1, 'picks the NVIDIA, not device 0');
  // vulkaninfo --summary format
  const vi = 'deviceName         = Intel(R) UHD Graphics (CML GT2)\n...\ndeviceName         = NVIDIA GeForce GTX 1650 Ti\n';
  assert.equal(g.pickDeviceIndex(g.parseVulkanDevices(vi)), 1);
  // single device -> leave default; AMD discrete beats AMD integrated
  assert.equal(g.pickDeviceIndex(['NVIDIA GeForce RTX 4070']), null);
  assert.equal(g.pickDeviceIndex(['AMD Radeon Graphics', 'AMD Radeon RX 7800 XT']), 1);
  // only integrated + software -> null (don't force anything)
  assert.equal(g.pickDeviceIndex(['Intel(R) Iris Xe', 'llvmpipe']), null);
});

test('local transcriber drops speaker bleed on "you" but keeps a real voice', () => {
  const { LocalWhisperTranscriber } = require('../src/local-whisper-transcriber');
  const t = new LocalWhisperTranscriber({ sessionOptions: {}, sessionFactory: () => ({}), segmenterFactory: () => ({}) });
  const mk = (amp) => { const b = Buffer.alloc(16000 * 2); const v = new Int16Array(b.buffer); for (let i = 0; i < v.length; i++) v[i] = Math.round(Math.sin(i / 3) * amp); return b; };
  const loud = mk(9000), quiet = mk(2500);
  const rms = (buf) => { const v = new Int16Array(buf.buffer, buf.byteOffset, buf.length / 2); let a = 0; for (let i = 0; i < v.length; i += 8) a += v[i] * v[i]; return Math.sqrt(a / (v.length / 8)); };
  t._lastThemRms = rms(loud); t._trackSpeech('them', true);
  assert.equal(t._isSpeakerBleed('you', quiet), true, 'quiet mic while speaker loud = echo, dropped');
  assert.equal(t._isSpeakerBleed('you', loud), false, 'loud mic while speaker loud = real voice, kept');
  t._trackSpeech('them', false); t._speech.themLastEnd = 0;
  assert.equal(t._isSpeakerBleed('you', quiet), false, 'speaker silent = everything kept');
  assert.equal(t._isSpeakerBleed('them', quiet), false, 'guard never touches the them channel');
});
