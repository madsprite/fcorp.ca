/* F Corp associate: runs Pebble-10M entirely inside this worker, on the visitor's device. */
importScripts('pebble.js');

let chat = null;
const here = p => new URL(p, self.location).href;

// total = the real (decompressed) size from the manifest. Content-Length can't be used:
// the host gzips the file, so the header gives the smaller compressed size.
async function fetchWithProgress(url, total) {
  const res = await fetch(url);
  if (!res.ok) throw new Error('HTTP ' + res.status);
  if (!res.body) return res.arrayBuffer();
  const reader = res.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value); got += value.length;
    postMessage({ type: 'progress', got: Math.min(got, total), total });
  }
  const out = new Uint8Array(got);
  let at = 0;
  for (const c of chunks) { out.set(c, at); at += c.length; }
  if (got !== total) throw new Error(`model download incomplete: ${got} of ${total} bytes`);
  return out.buffer;
}

async function load() {
  const manifest = await (await fetch(here('pebble-10m-base.json'))).json();
  const tj = await (await fetch(here('tokenizer.json'))).json();
  const words = (await (await fetch(here('words.txt'))).text()).split('\n').filter(Boolean);
  const buf = await fetchWithProgress(here('pebble-10m-base-f16.bin'), manifest.bytes);
  const model = new PebbleLM.Pebble(PebbleLM.loadWeights(manifest, buf));
  const tok = new PebbleLM.Tokenizer(tj);
  chat = new PebbleLM.ChatSession(model, tok, new PebbleLM.Lexicon(words, tok)); // runs the role-play header once
  postMessage({ type: 'ready' });
}

onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === 'load') { if (!chat) await load(); else postMessage({ type: 'ready' }); }
    if (m.type === 'new') chat.reset();
    if (m.type === 'reply') {
      const r = chat.reply(m.text, null, { forceTransfer: !!m.forceTransfer });
      postMessage({ type: 'reply', id: m.id, text: r.text, transfer: r.transfer });
    }
  } catch (err) {
    postMessage({ type: 'error', message: String(err && err.message || err) });
  }
};
