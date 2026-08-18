// Speech-to-text factory. Decoupled from the LLM provider because Anthropic has
// no audio API — we transcribe with whatever audio-capable key is available, and
// fall back across providers. Returns { text, provider } or { text:'', error }.
const { pcmToWav } = require('./wav');
const { formatProviderErrorMessage, isQuotaError, normalizeAzureBaseURL, CURRENT_GEMINI_DEFAULT } = require('./llm');

const BASE_VOCAB = 'CI/CD, Docker, Kubernetes, Terraform, Jenkins, AWS, Azure, GCP, ' +
  'CodeCommit, CodePipeline, CodeBuild, CodeDeploy, DevOps, SRE, microservices, deployment, ' +
  'pipeline, container, orchestration, Ansible, Prometheus, Grafana, Helm, EKS, ECS, Lambda, ' +
  'S3, EC2, IAM, GitHub Actions, GitLab, Kafka, PostgreSQL, Redis, MongoDB, REST API, gRPC';

// Whisper's classic non-speech outputs: closing phrases from its YouTube
// training data, subtitle credits, URLs, bracketed sound tags, and token loops
// ("EC2, EC2, EC2, ..."). Anything here on its own is noise, not the speaker.
function looksLikeHallucination(raw) {
  const trimmed = (raw || '').trim();
  if (!trimmed) return true;
  if (/^[\p{Emoji_Presentation}\p{Extended_Pictographic}\s]+$/u.test(trimmed)) return true;
  if (/^[\[\(♪*][^]*[\]\)♪*]$/.test(trimmed)) return true;                 // [BLANK_AUDIO], (music), ♪ ♪
  if (/(^|\s)(www\.|https?:\/\/)|\.(com|org|net|au|co\.uk)\b/i.test(trimmed)) return true; // URLs
  if (/subtitles?\s+by|amara\.org|transcri(bed|ption)\s+by|copyright/i.test(trimmed)) return true;
  const words = trimmed.toLowerCase().replace(/[^\p{L}\p{N}\s']/gu, ' ').split(/\s+/).filter(Boolean);
  if (words.length >= 4) {                                                   // token loop
    let run = 1;
    for (let i = 1; i < words.length; i++) {
      run = words[i] === words[i - 1] ? run + 1 : 1;
      if (run >= 4) return true;
    }
    if (new Set(words).size <= Math.max(1, Math.floor(words.length / 4))) return true;
  }
  const t = trimmed.replace(/[.,!?…]+$/g, '').trim().toLowerCase();
  const artifacts = new Set([
    'thank you', 'thank you very much', 'thank you for watching', 'thanks for watching',
    'thank you so much', 'thanks', 'please subscribe', 'like and subscribe', 'see you next time',
    'see you in the next video', 'bye-bye', 'bye bye', 'bye', 'goodbye', 'you', 'okay', 'oh', 'hmm', 'mm', 'uh', 'um',
    'the end', 'so', 'yeah', 'and', 'the', 'i'
  ]);
  return artifacts.has(t);
}

// Whisper decoder loops: "ask what you can do, ask what you can do, ask what
// you can do, ..." — a phrase of 1-8 words repeated 3+ times back to back is
// never real speech. Collapse it to one occurrence (keeps the sentence, drops
// the loop) instead of throwing the whole result away.
function collapseRepeats(text) {
  const words = String(text || '').split(/\s+/).filter(Boolean);
  if (words.length < 3) return words.join(' ');
  const norm = words.map((w) => w.toLowerCase().replace(/[^\p{L}\p{N}']/gu, ''));
  for (let n = 1; n <= 8; n++) {
    for (let i = 0; i + 3 * n <= words.length; i++) {
      let reps = 1;
      while (i + (reps + 1) * n <= words.length && norm.slice(i, i + n).join(' ') === norm.slice(i + reps * n, i + (reps + 1) * n).join(' ')) reps++;
      if (reps >= 3) {
        const kept = words.slice(0, i + n).concat(words.slice(i + reps * n));
        return collapseRepeats(kept.join(' '));
      }
    }
  }
  return words.join(' ');
}

// verbose_json segment filter — the standard whisper heuristics: a segment is
// silence/hallucination if the model itself is unsure (avg_logprob), it looped
// (compression_ratio), it says no-speech, or it claims to run far past the end
// of the audio (a "Thank you." stamped 0.0-29.98s on a 1s clip).
function keepSegments(data) {
  const segs = Array.isArray(data && data.segments) ? data.segments : null;
  if (!segs) return String((data && data.text) || '').trim();
  const dur = Number(data.duration) || 0;
  const kept = segs.filter((sg) => {
    if (!sg || !String(sg.text || '').trim()) return false;
    if (Number(sg.no_speech_prob) > 0.6) return false;
    if (Number(sg.avg_logprob) < -1.0) return false;
    if (Number(sg.compression_ratio) > 2.4) return false;
    if (dur && Number(sg.end) > dur + 1.5) return false;
    return !looksLikeHallucination(sg.text);
  });
  return collapseRepeats(kept.map((sg) => String(sg.text).trim()).join(' '));
}

function buildVocabPrompt(settings) {
  const s = settings || {};
  const text = (s.resumeText || '') + ' ' + (s.jobDescription || '');
  const proper = Array.from(new Set(text.match(/\b([A-Z][a-zA-Z0-9+.#]{2,}|[A-Z]{2,6})\b/g) || []));
  let prompt = BASE_VOCAB + (proper.length ? ', ' + proper.slice(0, 60).join(', ') : '');
  if (prompt.length > 850) prompt = prompt.slice(0, 850);
  return prompt;
}

// Whisper conditions on the prompt as "what was said just before". Feeding the
// tail of the previous transcript keeps casing/terms/sentence flow consistent
// across chunk boundaries (what whisper_streaming does). The prompt window is
// ~224 tokens and whisper keeps the END of it, so context goes last and the
// vocab list is trimmed to leave room.
function buildPrompt(settings, context) {
  const vocab = buildVocabPrompt(settings);
  const ctx = String(context || '').replace(/\s+/g, ' ').trim();
  if (!ctx) return vocab;
  return vocab.slice(0, 300) + '\n' + ctx.slice(-200);
}

async function transcribeOpenAI(apiKey, wav, model, baseURL, prompt) {
  const OpenAI = require('openai');
  const toFile = OpenAI.toFile || require('openai/uploads').toFile;
  const client = new OpenAI({ apiKey, baseURL });
  const file = await toFile(wav, 'audio.wav', { type: 'audio/wav' });
  const m = model || 'whisper-1';
  const verbose = /whisper/i.test(m); // gpt-4o-transcribe only does json/text
  const res = await client.audio.transcriptions.create({
    file,
    model: m,
    language: 'en',
    temperature: 0,
    prompt: prompt || '',
    ...(verbose ? { response_format: 'verbose_json' } : {})
  });
  return verbose ? keepSegments(res) : (res.text || '').trim();
}

// Azure transcription. Needs the resource endpoint + an *audio* deployment name
// (e.g. a whisper or gpt-4o-transcribe deployment) — the chat deployment won't
// work here. Mirrors streamAzure's auth: AzureOpenAI SDK for *.openai.azure.com,
// else the OpenAI-compatible base with the api-key header.
async function transcribeAzure(apiKey, wav, deployment, endpoint, prompt) {
  const OpenAI = require('openai');
  const url = normalizeAzureBaseURL(endpoint);
  if (!url) throw new Error('Add your Azure endpoint in Settings to transcribe with Azure.');
  if (!deployment) throw new Error('Set your Azure audio deployment name in Settings → Audio.');
  const toFile = OpenAI.toFile || require('openai/uploads').toFile;
  const file = await toFile(wav, 'audio.wav', { type: 'audio/wav' });
  let client;
  if (/\.openai\.azure\.com/i.test(url)) {
    client = new OpenAI.AzureOpenAI({ endpoint: url, apiKey, apiVersion: '2024-10-21' });
  } else {
    const azureFetch = async (input, init) => {
      const headers = new Headers(init && init.headers);
      headers.set('api-key', apiKey);
      headers.delete('authorization');
      return fetch(input, { ...init, headers });
    };
    client = new OpenAI({ baseURL: url, apiKey, fetch: azureFetch });
  }
  const res = await client.audio.transcriptions.create({
    file, model: deployment, language: 'en', temperature: 0, prompt: prompt || ''
  });
  return (res.text || '').trim();
}

// Custom OpenAI-compatible transcription endpoint (POST multipart to
// .../audio/transcriptions). Raw fetch, no SDK: one small request per
// utterance with a keep-alive connection, so it's as fast as the endpoint
// allows. Accepts either the full .../audio/transcriptions URL or a base URL.
function customSttEndpoint(url) {
  const u = String(url || '').trim().replace(/\/+$/, '');
  if (!u) return '';
  return /\/audio\/transcriptions$/i.test(u) ? u : u.replace(/\/v1$/i, '') + '/v1/audio/transcriptions';
}
let customVerbose = true; // flips off if the endpoint rejects verbose_json
async function transcribeCustom(cfg, wav, prompt, signal) {
  const endpoint = customSttEndpoint(cfg.url);
  if (!endpoint) throw new Error('Set the Custom transcription URL in Settings → Audio.');
  const form = new FormData();
  form.append('file', new Blob([wav], { type: 'audio/wav' }), 'audio.wav');
  form.append('model', cfg.model || 'whisper-1');
  form.append('response_format', customVerbose ? 'verbose_json' : 'json');
  form.append('temperature', '0');
  form.append('language', 'en');
  if (prompt) form.append('prompt', prompt);
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 20000);
  if (signal) signal.addEventListener('abort', () => controller.abort(), { once: true });
  try {
    const res = await fetch(endpoint, {
      method: 'POST',
      headers: cfg.apiKey ? { Authorization: `Bearer ${cfg.apiKey}` } : {},
      body: form,
      signal: controller.signal,
      keepalive: true
    });
    if (!res.ok) {
      const body = await res.text().catch(() => '');
      if (customVerbose && res.status === 400 && /response_format|verbose/i.test(body)) {
        customVerbose = false;
        return transcribeCustom(cfg, wav, prompt, signal);
      }
      const err = new Error(`Custom STT ${res.status}: ${body.slice(0, 200) || res.statusText}`);
      err.status = res.status;
      throw err;
    }
    const data = await res.json();
    return keepSegments(data);
  } finally { clearTimeout(timer); }
}

async function transcribeGemini(apiKey, wav) {
  const { GoogleGenAI } = require('@google/genai');
  const ai = new GoogleGenAI({ apiKey });
  const res = await ai.models.generateContent({
    model: CURRENT_GEMINI_DEFAULT,
    contents: [{ role: 'user', parts: [
      { text: 'Transcribe this audio verbatim. Return only the spoken words with no commentary. If there is no clear speech, return an empty response.' },
      { inlineData: { mimeType: 'audio/wav', data: wav.toString('base64') } }
    ] }]
  });
  return ((res && res.text) || '').trim();
}

function createSTT(settings) {
  const keys = settings.apiKeys || {};
  const selectedProvider = settings.sttProvider || 'auto';
  const chain = [];
  const custom = settings.customStt || {};
  if ((selectedProvider === 'auto' || selectedProvider === 'custom') && custom.url) {
    chain.push({ p: 'custom', fn: (wav, prompt, signal) => transcribeCustom(custom, wav, prompt, signal) });
  }
  if ((selectedProvider === 'auto' || selectedProvider === 'openai') && keys.openai) {
    chain.push({ p: 'openai', fn: (wav, prompt) => transcribeOpenAI(keys.openai, wav, settings.sttModel, undefined, prompt) });
  }
  if ((selectedProvider === 'auto' || selectedProvider === 'groq') && keys.groq) {
    chain.push({ p: 'groq', fn: (wav, prompt) => transcribeOpenAI(keys.groq, wav, 'whisper-large-v3-turbo', 'https://api.groq.com/openai/v1', prompt) });
  }
  if ((selectedProvider === 'auto' || selectedProvider === 'gemini') && keys.gemini) {
    chain.push({ p: 'gemini', fn: (wav) => transcribeGemini(keys.gemini, wav) });
  }
  if ((selectedProvider === 'auto' || selectedProvider === 'azure') && keys.azure && settings.azureEndpoint && settings.azureSttDeployment) {
    chain.push({ p: 'azure', fn: (wav, prompt) => transcribeAzure(keys.azure, wav, settings.azureSttDeployment, settings.azureEndpoint, prompt) });
  }
  // In auto mode prefer OpenAI when present; an explicit selection is already
  // the only entry, so this never overrides what the user picked.
  if (selectedProvider === 'auto' && keys.openai && chain.length > 1) chain.unshift(chain.splice(chain.findIndex((c) => c.p === 'openai'), 1)[0]);

  let disabledUntil = 0;
  let lastProvider = null;

  return {
    available: chain.length > 0,
    providers: chain.map((c) => c.p),
    // opts.context: tail of the previous transcript on this channel (prompt
    // conditioning for continuity across chunks). opts.signal: AbortSignal to
    // cancel a request whose result is no longer wanted (a superseded partial).
    async transcribe(pcm, opts = {}) {
      if (!chain.length || !pcm || pcm.length < 3200) return { text: '' };
      const now = Date.now();
      if (disabledUntil && now < disabledUntil) return { text: '', error: { provider: lastProvider, message: `Temporary ${lastProvider || 'provider'} quota or rate-limit; waiting 30s before retrying.` } };
      const wav = pcmToWav(pcm, 16000, 1);
      const prompt = buildPrompt(settings, opts.context);
      let lastErr = null;
      for (const c of chain) {
        try {
          const text = await c.fn(wav, prompt, opts.signal);
          disabledUntil = 0;
          lastProvider = c.p;
          if (looksLikeHallucination(text)) return { text: '', provider: c.p };
          return { text, provider: c.p };
        } catch (e) {
          // Shares detection/wording with the LLM error path (src/llm.js) so a
          // 404 (dead/misspelled model) or 429 (quota) reads the same whether it
          // came from a chat request or a transcription request.
          const quota = isQuotaError(e);
          const message = formatProviderErrorMessage(e, c.p);
          lastErr = { status: e && e.status, code: e && e.code, message, provider: c.p };
          if (quota) {
            lastProvider = c.p;
            disabledUntil = now + 30000;
            break;
          }
        }
      }
      return { text: '', error: lastErr };
    }
  };
}

module.exports = { createSTT, looksLikeHallucination, collapseRepeats, buildVocabPrompt, buildPrompt, keepSegments, customSttEndpoint };
