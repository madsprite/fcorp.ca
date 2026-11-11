/* F Corp senior associate: Qwen2.5-0.5B-Instruct via WebLLM (WebGPU), in a module worker.
   Loaded only when the visitor chooses to connect it. Messages never leave the device;
   the model files are fetched once from Hugging Face and kept in the browser's cache. */
import { CreateMLCEngine } from 'https://cdn.jsdelivr.net/npm/@mlc-ai/web-llm@0.2.86/+esm';
import { SENIOR_MODEL, TRANSFER_LINES, buildMessages, finish, guard } from './senior.js';

let engine = null, blocked = null, history = [];

async function load(f16) {
  blocked = new Set(await (await fetch(new URL('blocked.json', import.meta.url))).json());
  engine = await CreateMLCEngine(f16 ? SENIOR_MODEL.id : SENIOR_MODEL.idF32, {
    initProgressCallback: r => postMessage({ type: 'progress', progress: r.progress, text: r.text }),
  });
  postMessage({ type: 'ready' });
}

async function reply(text, forceTransfer) {
  let r = forceTransfer ? { text: TRANSFER_LINES[0], transfer: true } : guard(text);
  // one retry when the model copies template text ("Your Company Name"); then a safe line
  for (let attempt = 0; !r || (r.retry && attempt < 2); attempt++) {
    const out = await engine.chat.completions.create({
      messages: buildMessages(history, text),
      max_tokens: 40,
      temperature: 0.6,
      top_p: 0.9,
      frequency_penalty: 0.2,
    });
    r = finish(out.choices[0].message.content, blocked);
  }
  history.push({ role: 'user', content: text }, { role: 'assistant', content: r.text });
  return r;
}

onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === 'load') { if (!engine) await load(!!m.f16); else postMessage({ type: 'ready' }); }
    if (m.type === 'new') history = [];
    if (m.type === 'reply') {
      const r = await reply(m.text, !!m.forceTransfer);
      postMessage({ type: 'reply', id: m.id, text: r.text, transfer: r.transfer });
    }
  } catch (err) {
    postMessage({ type: 'error', id: m.id, message: String(err && err.message || err) });
  }
};
