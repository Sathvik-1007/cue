const { UtteranceSegmenter } = require('./utterance-segmenter');
const { WhisperServerSession } = require('./whisper-server-session');

const CHANNELS = Object.freeze(['you', 'them']);
// On stop, keep transcribing whatever is still queued rather than discarding it
// (previously anything not finished within 15s was thrown away, which on a slow
// CPU model silently lost the tail of every long session). Generous cap only
// as a safety net against a hung model.
const DEFAULT_DRAIN_TIMEOUT_MS = 180000;

class LocalWhisperTranscriber {
  /** Coordinate two audio channels through one sequential, persistent model session. */
  constructor({
    sessionOptions,
    sessionFactory = (options) => new WhisperServerSession(options),
    segmenterFactory = (options) => new UtteranceSegmenter(options),
    drainTimeoutMs = DEFAULT_DRAIN_TIMEOUT_MS,
    onTranscript = () => {},
    onSpeechState = () => {},
    onStatus = () => {},
    onError = () => {}
  }) {
    this.session = sessionFactory({ ...sessionOptions, onState: onStatus });
    this.segmenterFactory = segmenterFactory;
    this.drainTimeoutMs = drainTimeoutMs;
    this.onTranscript = onTranscript;
    this.onSpeechState = onSpeechState;
    this.onStatus = onStatus;
    this.onError = onError;
    this.segmenters = new Map();
    this.queueTail = Promise.resolve();
    this.pendingJobs = 0;
    this.acceptingAudio = false;
    this.discardPendingJobs = false;
  }

  async start() {
    this.discardPendingJobs = false;
    await this.session.start();
    this._buildSegmenters();
    this.acceptingAudio = true;
  }

  _buildSegmenters() {
    this.segmenters.clear();
    for (const channel of CHANNELS) {
      const isRemoteAudio = channel === 'them';
      this.segmenters.set(channel, this.segmenterFactory({
        channel,
        vadOptions: {
          onsetThreshold: isRemoteAudio ? 200 : 220,
          offsetThreshold: isRemoteAudio ? 120 : 130,
          silenceFrames: isRemoteAudio ? 20 : 18
        },
        onSpeechState: (speechChannel, speaking, durationMs) => {
          this._trackSpeech(speechChannel, speaking);
          this.onSpeechState(speechChannel, speaking, durationMs);
        },
        onUtterance: (utteranceChannel, pcm) => {
          if (this._isSpeakerBleed(utteranceChannel, pcm)) return; // drop echo of "them"
          this._enqueue(utteranceChannel, pcm);
        }
      }));
    }
  }

  // ---- speaker-bleed guard --------------------------------------------------
  // The laptop mic hears the speakers, so "them" audio leaks into "you" and gets
  // transcribed twice (once correctly as them, once garbled as you). We can't
  // use echo cancellation (it makes the audio server touch other streams). So:
  // a "you" utterance that was captured while "them" was speaking AND is much
  // quieter than what "them" is producing is treated as bleed and dropped. A
  // real person talking over the speaker is louder at the mic and gets through.
  _trackSpeech(channel, speaking) {
    if (!this._speech) this._speech = { them: false, themLastEnd: 0 };
    if (channel === 'them') {
      this._speech.them = speaking;
      if (!speaking) this._speech.themLastEnd = Date.now();
    }
  }
  _isSpeakerBleed(channel, pcm) {
    if (channel !== 'you' || !this._speech) return false;
    const themActive = this._speech.them || (Date.now() - this._speech.themLastEnd) < 800;
    if (!themActive) return false;
    // RMS of this utterance vs the recent "them" level
    const rms = (buf) => { const v = new Int16Array(buf.buffer, buf.byteOffset, Math.floor(buf.length / 2)); let a = 0; for (let i = 0; i < v.length; i += 8) a += v[i] * v[i]; return Math.sqrt(a / Math.max(1, v.length / 8)); };
    const youRms = rms(pcm);
    const themRms = this._lastThemRms || 0;
    // bleed sits well below the source; a real voice at the mic is comparable or louder
    return themRms > 0 && youRms < themRms * 0.55;
  }

  push(channel, pcm) {
    if (!this.acceptingAudio) return;
    if (channel === 'them' && pcm && pcm.length >= 640) {
      // running loudness of the speaker output, for the bleed guard
      const v = new Int16Array(pcm.buffer, pcm.byteOffset, Math.floor(pcm.length / 2));
      let a = 0; for (let i = 0; i < v.length; i += 8) a += v[i] * v[i];
      const r = Math.sqrt(a / Math.max(1, v.length / 8));
      this._lastThemRms = this._lastThemRms ? this._lastThemRms * 0.7 + r * 0.3 : r;
    }
    const segmenter = this.segmenters.get(channel);
    if (!segmenter) throw new Error(`Unknown local Whisper channel: ${channel}`);
    segmenter.push(pcm);
  }

  // stop({ keepSession }) — with keepSession the whisper-server (and the model
  // loaded on the GPU) stays resident so the NEXT listen starts instantly
  // instead of paying the ~9s model load + ~7s shader warmup again. The
  // caller releases the session on model change or app quit.
  async stop({ keepSession = false } = {}) {
    this.acceptingAudio = false;
    for (const segmenter of this.segmenters.values()) segmenter.stop();

    const drained = await this._drainQueue();
    if (!drained) {
      this.discardPendingJobs = true;
      this.session.abortInferences();
    }
    if (!keepSession) await this.session.stop({ force: !drained });
    this.segmenters.clear();
    this.onStatus({ status: keepSession ? 'idle' : 'off', message: keepSession ? 'Local Whisper idle (model kept loaded).' : 'Local Whisper stopped.' });
  }

  // Re-arm segmenters on an already-running session (instant restart).
  async resume() {
    if (!this.session.isRunning || !this.session.isRunning()) return this.start();
    this.discardPendingJobs = false;
    this._buildSegmenters();
    this.acceptingAudio = true;
    this.onStatus({ status: 'ready', message: 'Local Whisper is ready.' });
  }

  // Warm the GPU shader pipeline once with a short silent clip so the first
  // real chunk isn't slow (Vulkan compiles shaders lazily on first inference).
  async warmup() {
    try {
      const silent = Buffer.alloc(16000 * 2); // 1s of 16k mono silence
      await this.session.transcribe(silent);
    } catch (_) { /* best effort */ }
  }

  forceStop() {
    this.acceptingAudio = false;
    this.discardPendingJobs = true;
    this.session.abortInferences();
    return this.session.stop({ force: true });
  }

  _enqueue(channel, pcm) {
    this.pendingJobs += 1;
    this.onStatus({ status: 'transcribing', channel, pending: this.pendingJobs });

    const job = this.queueTail.then(async () => {
      if (this.discardPendingJobs) return;
      const text = await this.session.transcribe(pcm);
      if (text) this.onTranscript(channel, text);
    });

    this.queueTail = job
      .catch((error) => {
        if (!this.discardPendingJobs) this.onError(error);
      })
      .finally(() => {
        this.pendingJobs -= 1;
        if (this.acceptingAudio && this.pendingJobs === 0) {
          this.onStatus({ status: 'ready', message: 'Local Whisper is ready.' });
        }
      });
    return job;
  }

  async _drainQueue() {
    let timeout = null;
    try {
      return await Promise.race([
        this.queueTail.then(() => true),
        new Promise((resolve) => {
          timeout = setTimeout(() => resolve(false), this.drainTimeoutMs);
        })
      ]);
    } finally {
      if (timeout) clearTimeout(timeout);
    }
  }
}

module.exports = { LocalWhisperTranscriber, CHANNELS, DEFAULT_DRAIN_TIMEOUT_MS };
