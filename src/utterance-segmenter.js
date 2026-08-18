const { AdaptiveVAD, AudioRingBuffer } = require('./vad');

const DEFAULT_SAMPLE_RATE = 16000;
const PCM_BYTES_PER_SAMPLE = 2;
const DEFAULT_PRE_ROLL_MS = 300;
const DEFAULT_MIN_UTTERANCE_MS = 180;
// Cap each utterance at 5s: during continuous speech a chunk is transcribed
// every ~5s, so text streams in near-realtime instead of waiting for a pause
// (a lecture or a non-stop talker used to produce nothing for 12s+). Short
// pauses still finalize sooner via silence detection. overlapMs re-feeds the
// last 300ms into the next chunk so a word straddling the cut isn't lost.
const DEFAULT_MAX_UTTERANCE_MS = 5000;
const DEFAULT_OVERLAP_MS = 300;

class UtteranceSegmenter {
  /** Segment one PCM channel into bounded utterances without writing audio to disk. */
  constructor({
    channel,
    sampleRate = DEFAULT_SAMPLE_RATE,
    preRollMs = DEFAULT_PRE_ROLL_MS,
    minUtteranceMs = DEFAULT_MIN_UTTERANCE_MS,
    maxUtteranceMs = DEFAULT_MAX_UTTERANCE_MS,
    overlapMs = DEFAULT_OVERLAP_MS,
    vadOptions = {},
    onSpeechState = () => {},
    onUtterance = () => {},
    // Optional: while an utterance is open, hand out the audio collected so
    // far every partialIntervalMs so a caller can transcribe a growing window
    // and show words before the utterance ends (streaming over batch STT).
    partialIntervalMs = 0,
    onPartial = () => {},
    // Boundary quality (all optional, off by default):
    //  softCutMs   after this much audio, end the utterance at the next brief
    //              pause (softPauseMs of low energy) instead of waiting for
    //              the full end-of-speech silence -> bounded latency without
    //              cutting mid-word.
    //  cutSearchMs when the hard cap hits, cut at the quietest 30ms frame in
    //              the last cutSearchMs instead of at an arbitrary sample.
    //  minSpeechMs drop utterances with less than this much above-onset
    //              energy (a click, a breath) - whisper hallucinates on them.
    softCutMs = 0,
    softPauseMs = 150,
    cutSearchMs = 0,
    minSpeechMs = 0
  }) {
    if (!channel) throw new Error('UtteranceSegmenter requires a channel.');
    this.channel = channel;
    this.sampleRate = sampleRate;
    this.minimumBytes = this._millisecondsToBytes(minUtteranceMs);
    this.maximumBytes = this._millisecondsToBytes(maxUtteranceMs);
    this.overlapBytes = this._millisecondsToBytes(overlapMs);
    this.onSpeechState = onSpeechState;
    this.onUtterance = onUtterance;
    this.partialBytes = this._millisecondsToBytes(partialIntervalMs);
    this.onPartial = onPartial;
    this.bytesSincePartial = 0;
    this.frameMs = vadOptions.frameDurationMs || 30;
    this.frameBytes = Math.floor(sampleRate * this.frameMs / 1000) * PCM_BYTES_PER_SAMPLE;
    this.softCutBytes = this._millisecondsToBytes(softCutMs);
    this.softPauseFrames = Math.max(1, Math.round(softPauseMs / this.frameMs));
    this.searchFrames = Math.round(cutSearchMs / this.frameMs);
    this.minSpeechFrames = Math.round(minSpeechMs / this.frameMs);
    this._resetFrameState();
    this.ringBuffer = new AudioRingBuffer(preRollMs, sampleRate);
    this.utteranceChunks = [];
    this.utteranceBytes = 0;
    this.collecting = false;
    this.startedDuringPush = false;
    this.endedDuringPush = false;

    this.vad = new AdaptiveVAD({
      sampleRate,
      ...vadOptions,
      onSpeechStart: () => this._beginUtterance(),
      onSpeechEnd: (durationMs) => this._requestUtteranceEnd(durationMs),
      onSpeechAbort: () => this._abortUtterance(),
      onFrame: (energy, isSpeech, isSilence) => this._onFrame(energy, isSpeech, isSilence)
    });
  }

  _resetFrameState() {
    this.speechFramesSinceCut = 0;
    this.lowRun = 0;
    this.recentFrames = [];
    this.pendingCut = 0;
    this._chunkFrameIdx = 0;
    this._chunkAppends = false;
  }

  // Called by the VAD for every 30ms frame, BEFORE the chunk is appended.
  _onFrame(energy, isSpeech, isSilence) {
    if (!this.collecting) return;
    if (isSpeech) this.speechFramesSinceCut++;
    if (isSilence) this.lowRun++; else this.lowRun = 0;
    if (!this._chunkAppends) return; // positions only make sense for appended chunks
    this._chunkFrameIdx++;
    const pos = this.utteranceBytes + this._chunkFrameIdx * this.frameBytes; // end of this frame
    if (this.searchFrames) {
      this.recentFrames.push({ pos, energy });
      if (this.recentFrames.length > this.searchFrames) this.recentFrames.shift();
    }
    if (this.softCutBytes && !this.pendingCut && pos >= this.softCutBytes &&
        this.lowRun >= this.softPauseFrames && this.speechFramesSinceCut >= this.minSpeechFrames) {
      this.pendingCut = pos;
    }
  }

  push(pcm) {
    const chunk = Buffer.from(pcm || []);
    if (chunk.length < PCM_BYTES_PER_SAMPLE) return;
    if (chunk.length % PCM_BYTES_PER_SAMPLE !== 0) {
      throw new Error('PCM chunks must contain complete 16-bit samples.');
    }

    this.startedDuringPush = false;
    this.endedDuringPush = false;
    const wasCollecting = this.collecting;
    this._chunkAppends = wasCollecting;
    this._chunkFrameIdx = 0;

    if (!wasCollecting) this.ringBuffer.write(chunk);
    this.vad.processChunk(chunk);

    // A chunk that triggered speech start is already present in the pre-roll.
    if (wasCollecting && this.collecting) this._appendChunk(chunk);
    if (this.endedDuringPush) this._finalizeUtterance();
    else if (this.collecting && this.pendingCut && this.pendingCut <= this.utteranceBytes) this._cutAt(this.pendingCut);
    else if (this.collecting && this.partialBytes && this.bytesSincePartial >= this.partialBytes) {
      this.bytesSincePartial = 0;
      this.onPartial(this.channel, Buffer.concat(this.utteranceChunks, this.utteranceBytes));
    }
  }

  stop() {
    if (this.collecting) this._finalizeUtterance();
    this.reset();
  }

  reset() {
    this.collecting = false;
    this.startedDuringPush = false;
    this.endedDuringPush = false;
    this.utteranceChunks = [];
    this.utteranceBytes = 0;
    this._resetFrameState();
    this.ringBuffer.clear();
    this.vad.reset();
  }

  // Too short to be speech (VAD never confirmed it): drop what was collected.
  _abortUtterance() {
    if (!this.collecting) return;
    this.collecting = false;
    this.utteranceChunks = [];
    this.utteranceBytes = 0;
    this._resetFrameState();
    this.ringBuffer.clear();
    this.onSpeechState(this.channel, false, 0);
  }

  // Emit [0, pos) and keep the rest as the start of the next utterance.
  _cutAt(pos) {
    const combined = Buffer.concat(this.utteranceChunks, this.utteranceBytes);
    const head = combined.subarray(0, pos);
    if (head.length >= this.minimumBytes) this._emit(head);
    const remainder = Buffer.from(combined.subarray(pos));
    this.utteranceChunks = remainder.length ? [remainder] : [];
    this.utteranceBytes = remainder.length;
    this.bytesSincePartial = 0;
    this._resetFrameState();
    this._chunkAppends = true;
  }

  _beginUtterance() {
    this.collecting = true;
    this.startedDuringPush = true;
    const preRoll = this.ringBuffer.read();
    this.utteranceChunks = preRoll.length ? [Buffer.from(preRoll)] : [];
    this.utteranceBytes = preRoll.length;
    this.bytesSincePartial = 0;
    this._resetFrameState();
    this.ringBuffer.clear();
    this.onSpeechState(this.channel, true);
  }

  _requestUtteranceEnd(durationMs) {
    this.endedDuringPush = true;
    this.onSpeechState(this.channel, false, durationMs);
  }

  // Time O(n), space O(n) at each 25-second boundary; regular pushes are O(1).
  _appendChunk(chunk) {
    this.utteranceChunks.push(chunk);
    this.utteranceBytes += chunk.length;
    this.bytesSincePartial += chunk.length;

    while (this.utteranceBytes >= this.maximumBytes) {
      if (this.searchFrames && this.recentFrames.length) {
        // Hard cap: cut at the quietest recent frame (between words), no overlap.
        let best = this.recentFrames[0];
        for (const f of this.recentFrames) if (f.energy < best.energy) best = f;
        this._cutAt(Math.min(best.pos, this.utteranceBytes));
        continue;
      }
      const combined = Buffer.concat(this.utteranceChunks, this.utteranceBytes);
      this._emit(combined.subarray(0, this.maximumBytes));
      const nextStart = Math.max(0, this.maximumBytes - this.overlapBytes);
      const remainder = Buffer.from(combined.subarray(nextStart));
      this.utteranceChunks = remainder.length ? [remainder] : [];
      this.utteranceBytes = remainder.length;
      this.bytesSincePartial = 0;
    }
  }

  _finalizeUtterance() {
    if (!this.collecting) return;
    const utterance = Buffer.concat(this.utteranceChunks, this.utteranceBytes);
    const enoughSpeech = !this.minSpeechFrames || this.speechFramesSinceCut >= this.minSpeechFrames;
    if (utterance.length >= this.minimumBytes && enoughSpeech) this._emit(utterance);
    this.collecting = false;
    this.endedDuringPush = false;
    this.utteranceChunks = [];
    this.utteranceBytes = 0;
    this._resetFrameState();
    this.ringBuffer.clear();
  }

  _emit(pcm) {
    this.onUtterance(this.channel, Buffer.from(pcm));
  }

  _millisecondsToBytes(durationMs) {
    return Math.floor(this.sampleRate * PCM_BYTES_PER_SAMPLE * durationMs / 1000);
  }
}

module.exports = {
  UtteranceSegmenter,
  DEFAULT_SAMPLE_RATE,
  DEFAULT_PRE_ROLL_MS,
  DEFAULT_MIN_UTTERANCE_MS,
  DEFAULT_MAX_UTTERANCE_MS,
  DEFAULT_OVERLAP_MS
};
