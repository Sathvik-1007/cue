const test = require('node:test');
const assert = require('node:assert');
const { looksLikeHallucination, buildVocabPrompt } = require('../src/stt');
const { DeepgramStreamingSTT } = require('../src/stt-streaming');

test('looksLikeHallucination drops Whisper silence artifacts', () => {
  ['', '   ', 'Thank you for watching.', 'thanks for watching', 'Bye-bye!', '👍👍'].forEach((s) => {
    assert.equal(looksLikeHallucination(s), true, JSON.stringify(s));
  });
});

test('looksLikeHallucination keeps real speech', () => {
  ['Tell me about your experience with Kubernetes.', 'You know, I led the migration.'].forEach((s) => {
    assert.equal(looksLikeHallucination(s), false, JSON.stringify(s));
  });
});

test('buildVocabPrompt seeds base vocab and resume proper nouns, capped', () => {
  const p = buildVocabPrompt({ resumeText: 'Optum EKS Terraform', jobDescription: 'AWS SRE' });
  assert.ok(p.includes('Kubernetes'));
  assert.ok(p.includes('Optum'));
  assert.ok(p.length <= 850);
  assert.ok(buildVocabPrompt(undefined).length > 0);
  assert.ok(buildVocabPrompt({ resumeText: 'Xyzzy '.repeat(4000) }).length <= 850);
});

test('Deepgram accumulates is_final segments into one turn at speech_final', () => {
  const finals = [];
  const d = new DeepgramStreamingSTT('k', { onTranscript: (t) => finals.push(t) });
  d._handleMessage({ type: 'Results', is_final: true, channel: { alternatives: [{ transcript: 'Tell me about' }] } });
  d._handleMessage({ type: 'Results', is_final: true, channel: { alternatives: [{ transcript: 'your experience' }] } });
  d._handleMessage({ type: 'Results', is_final: true, speech_final: true, channel: { alternatives: [{ transcript: 'with Kubernetes.' }] } });
  assert.deepEqual(finals, ['Tell me about your experience with Kubernetes.']);
});

test('Deepgram flushes pending segments on UtteranceEnd when speech_final never arrives', () => {
  const finals = [];
  const d = new DeepgramStreamingSTT('k', { onTranscript: (t) => finals.push(t) });
  d._handleMessage({ type: 'Results', is_final: true, channel: { alternatives: [{ transcript: 'hello there' }] } });
  d._handleMessage({ type: 'UtteranceEnd' });
  assert.deepEqual(finals, ['hello there']);
  d._handleMessage({ type: 'UtteranceEnd' });
  assert.deepEqual(finals, ['hello there'], 'no duplicate emit on a second UtteranceEnd');
});

test('Deepgram drops hallucinated finals', () => {
  const finals = [];
  const d = new DeepgramStreamingSTT('k', { onTranscript: (t) => finals.push(t) });
  d._handleMessage({ type: 'Results', is_final: true, speech_final: true, channel: { alternatives: [{ transcript: 'Thank you.' }] } });
  assert.deepEqual(finals, []);
});

const { collapseRepeats, keepSegments } = require('../src/stt');
test('collapseRepeats folds whisper decoder loops, keeps real speech', () => {
  assert.equal(collapseRepeats('I think, I think, I think we should go.'), 'I think, we should go.');
  assert.equal(collapseRepeats('ask not what you can do, ask what you can do, ask what you can do, ask what you can do.'), 'ask not what you can do, ask what you can do.');
  assert.equal(collapseRepeats('Tell me about your experience with Kubernetes.'), 'Tell me about your experience with Kubernetes.');
  assert.equal(collapseRepeats('no no'), 'no no');
});
test('keepSegments drops low-confidence / looping / past-the-end segments', () => {
  const ok = { text: ' And so, my fellow Americans.', end: 10.4, avg_logprob: -0.2, compression_ratio: 1.3, no_speech_prob: 0 };
  assert.equal(keepSegments({ duration: 11, segments: [ok] }), 'And so, my fellow Americans.');
  assert.equal(keepSegments({ duration: 1, segments: [{ ...ok, text: ' Thank you.', end: 29.98 }] }), '');
  assert.equal(keepSegments({ duration: 5, segments: [{ ...ok, avg_logprob: -1.4 }] }), '');
  assert.equal(keepSegments({ duration: 5, segments: [{ ...ok, compression_ratio: 3.1 }] }), '');
  assert.equal(keepSegments({ duration: 5, segments: [{ ...ok, no_speech_prob: 0.9 }] }), '');
  assert.equal(keepSegments({ text: 'plain json fallback' }), 'plain json fallback');
});
test('looksLikeHallucination catches URLs, tags, loops', () => {
  ['www.pens.com.au', '[BLANK_AUDIO]', '(music)', 'S3, EC2, EC2, EC2, EC2, EC2', 'Subtitles by the Amara.org community'].forEach((s) => assert.equal(looksLikeHallucination(s), true, s));
  ['Yes.', 'I led the migration to EKS.', 'We use S3 and EC2 a lot.'].forEach((s) => assert.equal(looksLikeHallucination(s), false, s));
});
