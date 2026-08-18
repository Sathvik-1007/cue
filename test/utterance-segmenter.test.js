const assert = require('node:assert/strict');
const test = require('node:test');
const { UtteranceSegmenter } = require('../src/utterance-segmenter');

const FRAME_SAMPLES = 480;
const FRAME_BYTES = FRAME_SAMPLES * 2;

function frame(amplitude) {
  const buffer = Buffer.alloc(FRAME_BYTES);
  for (let offset = 0; offset < buffer.length; offset += 2) buffer.writeInt16LE(amplitude, offset);
  return buffer;
}

function pushFrames(segmenter, amplitude, count) {
  for (let index = 0; index < count; index += 1) segmenter.push(frame(amplitude));
}

test('includes pre-roll and emits after the configured trailing silence', () => {
  const utterances = [];
  const speechStates = [];
  const segmenter = new UtteranceSegmenter({
    channel: 'you',
    onSpeechState: (_channel, speaking) => speechStates.push(speaking),
    onUtterance: (_channel, pcm) => utterances.push(pcm)
  });

  pushFrames(segmenter, 0, 10);
  pushFrames(segmenter, 1200, 8);
  pushFrames(segmenter, 0, 18);

  assert.equal(utterances.length, 1);
  assert.ok(utterances[0].length >= FRAME_BYTES * 30);
  assert.deepEqual(speechStates, [true, false]);
});

test('flushes a final active utterance when capture stops', () => {
  const utterances = [];
  const segmenter = new UtteranceSegmenter({
    channel: 'them',
    onUtterance: (_channel, pcm) => utterances.push(pcm)
  });
  pushFrames(segmenter, 1000, 8);
  segmenter.stop();
  assert.equal(utterances.length, 1);
  assert.ok(utterances[0].length >= FRAME_BYTES * 8);
});

test('splits long speech into bounded segments with overlap', () => {
  const utterances = [];
  const segmenter = new UtteranceSegmenter({
    channel: 'you',
    preRollMs: 60,
    minUtteranceMs: 30,
    maxUtteranceMs: 300,
    overlapMs: 60,
    onUtterance: (_channel, pcm) => utterances.push(pcm)
  });
  pushFrames(segmenter, 1000, 24);
  segmenter.stop();

  assert.ok(utterances.length >= 3);
  assert.equal(utterances[0].length, FRAME_BYTES * 10);
  assert.equal(utterances[1].length, FRAME_BYTES * 10);
  assert.ok(utterances.every((pcm) => pcm.length <= FRAME_BYTES * 10));
});

// ---- boundary-quality options (soft cut / quiet-frame hard cut / min speech / abort)

test('softCutMs: long speech ends at the first brief pause, not mid-word', () => {
  const utterances = [];
  const segmenter = new UtteranceSegmenter({
    channel: 'them', preRollMs: 0, maxUtteranceMs: 30000, overlapMs: 0,
    softCutMs: 1500, softPauseMs: 150, minSpeechMs: 0,
    vadOptions: { silenceFrames: 20 },
    onUtterance: (_c, pcm) => utterances.push(pcm)
  });
  pushFrames(segmenter, 1200, 60);   // 1.8s speech (past softCut)
  pushFrames(segmenter, 0, 6);       // 180ms pause -> soft cut here (VAD still trailing)
  pushFrames(segmenter, 1200, 20);   // speech resumes -> second utterance
  pushFrames(segmenter, 0, 25);      // real end
  assert.equal(utterances.length, 2);
  // first chunk = speech + the pause frames it was cut in, nothing from the resumed speech
  assert.ok(utterances[0].length >= FRAME_BYTES * 60 && utterances[0].length <= FRAME_BYTES * 67, String(utterances[0].length));
});

test('cutSearchMs: the hard cap cuts at the quietest recent frame, no overlap', () => {
  const utterances = [];
  const segmenter = new UtteranceSegmenter({
    channel: 'them', preRollMs: 0, maxUtteranceMs: 900, overlapMs: 300,
    cutSearchMs: 300, minSpeechMs: 0,
    vadOptions: { silenceFrames: 10 },
    onUtterance: (_c, pcm) => utterances.push(pcm)
  });
  pushFrames(segmenter, 1200, 22);   // 660ms loud (first frame is the onset frame: pre-roll, not appended)
  pushFrames(segmenter, 300, 1);     // one quiet-ish frame between words (still above offset 150)
  pushFrames(segmenter, 1200, 12);   // cap (30 frames = 900ms) hits here
  pushFrames(segmenter, 0, 15);
  assert.equal(utterances.length, 2);
  // cut lands right after the quiet frame, not at the 30-frame cap, and nothing is duplicated
  assert.equal(utterances[0].length, FRAME_BYTES * 22);
  assert.equal(utterances[0].length + utterances[1].length, FRAME_BYTES * (21 + 1 + 12 + 10));
});

test('minSpeechMs: a click / breath is never emitted; a real short word is', () => {
  const utterances = [];
  const states = [];
  const mk = (minSpeechMs) => new UtteranceSegmenter({
    channel: 'you', preRollMs: 0, minSpeechMs, vadOptions: { silenceFrames: 5, minSpeechFrames: 4 },
    onSpeechState: (_c, speaking) => states.push(speaking),
    onUtterance: (_c, pcm) => utterances.push(pcm)
  });
  const s1 = mk(200);
  pushFrames(s1, 1200, 2);   // 60ms click: VAD aborts (below minSpeechFrames) -> nothing collected
  pushFrames(s1, 0, 10);
  assert.equal(utterances.length, 0);
  assert.deepEqual(states, [true, false]); // abort still closes the speech state
  const s2 = mk(200);
  pushFrames(s2, 1200, 5);   // 150ms burst: VAD accepts, but < 200ms of speech energy -> dropped
  pushFrames(s2, 0, 10);
  assert.equal(utterances.length, 0);
  const s3 = mk(200);
  pushFrames(s3, 1200, 9);   // 270ms "yes" -> emitted
  pushFrames(s3, 0, 10);
  assert.equal(utterances.length, 1);
});
