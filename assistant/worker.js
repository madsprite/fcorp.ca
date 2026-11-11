/* F Corp associate: runs Pebble-10M-Chat entirely inside this worker, on the visitor's device. */
importScripts('pebble.js');

let model = null, tok = null, lex = null;
const here = p => new URL(p, self.location).href;

async function fetchWithProgress(url, total) {
  const res = await fetch(url);
  if (!res.ok) throw new Error('HTTP ' + res.status);
  const len = Number(res.headers.get('content-length')) || total;
  if (!res.body || !len) return res.arrayBuffer();
  const reader = res.body.getReader();
  const out = new Uint8Array(len);
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    out.set(value, got); got += value.length;
    postMessage({ type: 'progress', got, total: len });
  }
  return out.buffer.slice(0, got);
}

async function load() {
  const manifest = await (await fetch(here('pebble-10m.json'))).json();
  const tj = await (await fetch(here('tokenizer.json'))).json();
  const words = (await (await fetch(here('words.txt'))).text()).split('\n').filter(Boolean);
  const buf = await fetchWithProgress(here('pebble-10m-f16.bin'), manifest.bytes);
  const W = PebbleLM.loadWeights(manifest, buf);
  model = new PebbleLM.Pebble(W);
  tok = new PebbleLM.Tokenizer(tj);
  lex = new PebbleLM.Lexicon(words, tok);
  postMessage({ type: 'ready' });
}

onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === 'load') { if (!model) await load(); else postMessage({ type: 'ready' }); }
    if (m.type === 'reply') {
      const text = PebbleLM.reply(model, tok, m.text, null, partial => postMessage({ type: 'partial', id: m.id, text: partial }), lex);
      postMessage({ type: 'reply', id: m.id, text });
    }
  } catch (err) {
    postMessage({ type: 'error', message: String(err && err.message || err) });
  }
};
