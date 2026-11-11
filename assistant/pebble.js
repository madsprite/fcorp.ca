/*
 * Pebble-10M, token-by-token inference in plain JavaScript.
 * Model: basically-ai/Pebble-10M (Apache-2.0). Hybrid Mamba2 + attention,
 * 8 blocks (m m m a m m m a), d_model 384, vocab 2048, context 512.
 *
 * Runs in a Web Worker (importScripts) and in Node (require) with the same code.
 * Recurrent mode: each Mamba2 block keeps a conv window and SSM state; each
 * attention block keeps a KV cache. One call to step() = one token.
 */
(function (root) {
  'use strict';

  const D = 384, V = 2048, CTX = 512;
  const D_INNER = 768, D_STATE = 128, N_HEADS_M = 8, HEADDIM = 96, D_CONV = 4;
  const CONV_DIM = D_INNER + 2 * D_STATE;               // 1024
  const IN_PROJ = 2 * D_INNER + 2 * D_STATE + N_HEADS_M; // 1800
  const N_HEADS_A = 6, HD = 64, FF = 1536;
  const LAYOUT = ['m', 'm', 'm', 'a', 'm', 'm', 'm', 'a'];

  // ---------- weights ----------
  function f16ToF32Array(u16) {
    const out = new Float32Array(u16.length);
    for (let i = 0; i < u16.length; i++) {
      const h = u16[i], s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
      out[i] = e === 0 ? s * f * 5.960464477539063e-8
             : e === 31 ? (f ? NaN : s * Infinity)
             : s * Math.pow(2, e - 15) * (1 + f / 1024);
    }
    return out;
  }

  // manifest: { tensors: [{name, shape, offset, length, dtype:'f16'|'f32'}] }, buf: ArrayBuffer
  function loadWeights(manifest, buf) {
    const W = {};
    for (const t of manifest.tensors) {
      W[t.name] = t.dtype === 'f16'
        ? f16ToF32Array(new Uint16Array(buf, t.offset, t.length))
        : new Float32Array(buf.slice(t.offset, t.offset + 4 * t.length));
    }
    if (!W['lm_head.weight']) W['lm_head.weight'] = W['wte.weight'];
    return W;
  }

  // ---------- math ----------
  // out[o] = sum_i W[o*n + i] * x[i]
  function matvec(W, x, out, rows, n) {
    for (let o = 0; o < rows; o++) {
      let acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
      const base = o * n;
      let i = 0;
      for (; i + 3 < n; i += 4) {
        acc0 += W[base + i] * x[i];
        acc1 += W[base + i + 1] * x[i + 1];
        acc2 += W[base + i + 2] * x[i + 2];
        acc3 += W[base + i + 3] * x[i + 3];
      }
      for (; i < n; i++) acc0 += W[base + i] * x[i];
      out[o] = acc0 + acc1 + acc2 + acc3;
    }
  }
  function rmsnorm(x, w, out, n, eps) {
    let ss = 0;
    for (let i = 0; i < n; i++) ss += x[i] * x[i];
    const r = 1 / Math.sqrt(ss / n + eps);
    for (let i = 0; i < n; i++) out[i] = w[i] * (x[i] * r);
  }
  const silu = v => v / (1 + Math.exp(-v));
  const softplus = v => v > 20 ? v : Math.log1p(Math.exp(v));
  function erf(x) { // Abramowitz-Stegun 7.1.26, |err| < 1.5e-7
    const s = x < 0 ? -1 : 1; x = Math.abs(x);
    const t = 1 / (1 + 0.3275911 * x);
    const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
    return s * y;
  }
  const gelu = v => 0.5 * v * (1 + erf(v / Math.SQRT2));

  // ---------- model ----------
  class Pebble {
    constructor(W) {
      this.W = W;
      this.invf = new Float32Array(HD / 2);
      for (let i = 0; i < HD / 2; i++) this.invf[i] = 1 / Math.pow(10000, (2 * i) / HD);
      // scratch
      this.x = new Float32Array(D); this.h = new Float32Array(D); this.t = new Float32Array(D);
      this.zx = new Float32Array(IN_PROJ); this.xbc = new Float32Array(CONV_DIM);
      this.y = new Float32Array(D_INNER);
      this.qkv = new Float32Array(3 * D); this.att = new Float32Array(D); this.sc = new Float32Array(CTX);
      this.ff = new Float32Array(FF); this.logits = new Float32Array(V);
      this.reset();
    }
    reset() {
      this.pos = 0;
      this.state = LAYOUT.map(k => k === 'm'
        ? { conv: new Float32Array(CONV_DIM * D_CONV), ssm: new Float32Array(N_HEADS_M * HEADDIM * D_STATE) }
        : { k: new Float32Array(CTX * D), v: new Float32Array(CTX * D) });
    }

    mamba(li) {
      const W = this.W, p = `blocks.${li}.`, st = this.state[li];
      const x = this.x, h = this.h, zx = this.zx, xbc = this.xbc, y = this.y;
      rmsnorm(x, W[p + 'ln.weight'], h, D, 1e-6);
      matvec(W[p + 'mixer.in_proj.weight'], h, zx, IN_PROJ, D);
      // zx = [z (768) | xBC (1024) | dt (8)]
      const cw = W[p + 'mixer.conv1d.weight'], cb = W[p + 'mixer.conv1d.bias'], conv = st.conv;
      for (let c = 0; c < CONV_DIM; c++) {
        const b = c * D_CONV;
        conv[b] = conv[b + 1]; conv[b + 1] = conv[b + 2]; conv[b + 2] = conv[b + 3];
        conv[b + 3] = zx[D_INNER + c];
        const v = conv[b] * cw[b] + conv[b + 1] * cw[b + 1] + conv[b + 2] * cw[b + 2] + conv[b + 3] * cw[b + 3] + cb[c];
        xbc[c] = silu(v);
      }
      const Alog = W[p + 'mixer.A_log'], dtb = W[p + 'mixer.dt_bias'], Dp = W[p + 'mixer.D'], ssm = st.ssm;
      const B0 = D_INNER, C0 = D_INNER + D_STATE;
      for (let hh = 0; hh < N_HEADS_M; hh++) {
        const dt = softplus(zx[D_INNER + CONV_DIM + hh] + dtb[hh]);
        const dA = Math.exp(-Math.exp(Alog[hh]) * dt);
        for (let pp = 0; pp < HEADDIM; pp++) {
          const xi = xbc[hh * HEADDIM + pp], k = dt * xi;
          const sb = (hh * HEADDIM + pp) * D_STATE;
          let acc = 0;
          for (let n = 0; n < D_STATE; n++) {
            const s = ssm[sb + n] * dA + k * xbc[B0 + n];
            ssm[sb + n] = s;
            acc += s * xbc[C0 + n];
          }
          y[hh * HEADDIM + pp] = acc + Dp[hh] * xi;
        }
      }
      // gated RMSNorm (norm_before_gate = false): norm(y * silu(z))
      for (let i = 0; i < D_INNER; i++) y[i] *= silu(zx[i]);
      rmsnorm(y, W[p + 'mixer.norm.weight'], y, D_INNER, 1e-5);
      matvec(W[p + 'mixer.out_proj.weight'], y, this.t, D, D_INNER);
      for (let i = 0; i < D; i++) x[i] += this.t[i];
    }

    attention(li) {
      const W = this.W, p = `blocks.${li}.`, st = this.state[li], pos = this.pos;
      const x = this.x, h = this.h, qkv = this.qkv, att = this.att, sc = this.sc;
      rmsnorm(x, W[p + 'ln1.weight'], h, D, 1e-6);
      matvec(W[p + 'wqkv.weight'], h, qkv, 3 * D, D);
      // rotary on q and k (rotate-half), heads laid out [3][6][64]
      const half = HD / 2;
      for (let s = 0; s < 2; s++) for (let hh = 0; hh < N_HEADS_A; hh++) {
        const b = s * D + hh * HD;
        for (let i = 0; i < half; i++) {
          const a = pos * this.invf[i], c = Math.cos(a), sn = Math.sin(a);
          const v1 = qkv[b + i], v2 = qkv[b + half + i];
          qkv[b + i] = v1 * c - v2 * sn;
          qkv[b + half + i] = v1 * sn + v2 * c;
        }
      }
      st.k.set(qkv.subarray(D, 2 * D), pos * D);
      st.v.set(qkv.subarray(2 * D, 3 * D), pos * D);
      const scale = 1 / Math.sqrt(HD);
      for (let hh = 0; hh < N_HEADS_A; hh++) {
        const qo = hh * HD;
        let mx = -Infinity;
        for (let t = 0; t <= pos; t++) {
          let d = 0; const ko = t * D + qo;
          for (let i = 0; i < HD; i++) d += qkv[qo + i] * st.k[ko + i];
          d *= scale; sc[t] = d; if (d > mx) mx = d;
        }
        let sum = 0;
        for (let t = 0; t <= pos; t++) { sc[t] = Math.exp(sc[t] - mx); sum += sc[t]; }
        for (let i = 0; i < HD; i++) att[qo + i] = 0;
        for (let t = 0; t <= pos; t++) {
          const wgt = sc[t] / sum, vo = t * D + qo;
          for (let i = 0; i < HD; i++) att[qo + i] += wgt * st.v[vo + i];
        }
      }
      matvec(W[p + 'wo.weight'], att, this.t, D, D);
      for (let i = 0; i < D; i++) x[i] += this.t[i];
      rmsnorm(x, W[p + 'ln2.weight'], h, D, 1e-6);
      matvec(W[p + 'fc1.weight'], h, this.ff, FF, D);
      for (let i = 0; i < FF; i++) this.ff[i] = gelu(this.ff[i]);
      matvec(W[p + 'fc2.weight'], this.ff, this.t, D, FF);
      for (let i = 0; i < D; i++) x[i] += this.t[i];
    }

    // copy of the recurrent state, so a fixed prompt prefix is only processed once
    snapshot() {
      return {
        pos: this.pos,
        state: this.state.map(s => s.conv
          ? { conv: s.conv.slice(), ssm: s.ssm.slice() }
          : { k: s.k.slice(0, this.pos * D), v: s.v.slice(0, this.pos * D) }),
        logits: this.logits.slice(),
      };
    }
    restore(snap) {
      this.pos = snap.pos;
      snap.state.forEach((s, i) => {
        const t = this.state[i];
        if (s.conv) { t.conv.set(s.conv); t.ssm.set(s.ssm); }
        else { t.k.set(s.k); t.v.set(s.v); }
      });
      this.logits.set(snap.logits);
    }

    // feed one token, return logits for the next one
    step(tok) {
      if (this.pos >= CTX) throw new Error('context full');
      this.x.set(this.W['wte.weight'].subarray(tok * D, tok * D + D));
      for (let li = 0; li < LAYOUT.length; li++) {
        if (LAYOUT[li] === 'm') this.mamba(li); else this.attention(li);
      }
      rmsnorm(this.x, this.W['lnf.weight'], this.h, D, 1e-6);
      matvec(this.W['lm_head.weight'], this.h, this.logits, V, D);
      this.pos++;
      return this.logits;
    }
  }

  // ---------- sampling ----------
  function sample(logits, recent, opt) {
    const { temperature = 0.7, topK = 50, topP = 0.95, repetitionPenalty = 1.2, rand = Math.random } = opt || {};
    const l = Float32Array.from(logits);
    for (const t of recent) l[t] = l[t] > 0 ? l[t] / repetitionPenalty : l[t] * repetitionPenalty;
    if (temperature <= 0) { let b = 0; for (let i = 1; i < l.length; i++) if (l[i] > l[b]) b = i; return b; }
    const idx = Array.from(l.keys()).filter(i => l[i] > -Infinity).sort((a, b) => l[b] - l[a]).slice(0, topK);
    if (!idx.length) return 0;
    const mx = l[idx[0]];
    let probs = idx.map(i => Math.exp((l[i] - mx) / temperature));
    let s = probs.reduce((a, b) => a + b, 0); probs = probs.map(p => p / s);
    let cum = 0, keep = 0;
    while (keep < probs.length) { cum += probs[keep++]; if (cum >= topP) break; }
    const kp = probs.slice(0, keep), ks = kp.reduce((a, b) => a + b, 0);
    let r = rand() * ks;
    for (let i = 0; i < keep; i++) { r -= kp[i]; if (r <= 0) return idx[i]; }
    return idx[keep - 1];
  }

  // ---------- tokenizer (GPT-2 style byte-level BPE) ----------
  class Tokenizer {
    constructor(tj) {
      this.vocab = tj.model.vocab;
      this.inv = [];
      for (const [k, v] of Object.entries(this.vocab)) this.inv[v] = k;
      for (const a of tj.added_tokens || []) { this.vocab[a.content] = a.id; this.inv[a.id] = a.content; }
      this.ranks = new Map();
      tj.model.merges.forEach((m, i) => this.ranks.set(Array.isArray(m) ? m[0] + ' ' + m[1] : m, i));
      // byte <-> unicode table
      const bs = [];
      for (let i = 33; i <= 126; i++) bs.push(i);
      for (let i = 161; i <= 172; i++) bs.push(i);
      for (let i = 174; i <= 255; i++) bs.push(i);
      const cs = bs.slice(); let n = 0;
      for (let b = 0; b < 256; b++) if (!bs.includes(b)) { bs.push(b); cs.push(256 + n++); }
      this.b2u = []; this.u2b = {};
      bs.forEach((b, i) => { const ch = String.fromCharCode(cs[i]); this.b2u[b] = ch; this.u2b[ch] = b; });
      this.pat = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;
      this.enc = new TextEncoder(); this.dec = new TextDecoder();
      this.cache = new Map();
    }
    bpe(word) {
      if (this.cache.has(word)) return this.cache.get(word);
      let parts = Array.from(word);
      while (parts.length > 1) {
        let best = -1, bestRank = Infinity;
        for (let i = 0; i < parts.length - 1; i++) {
          const r = this.ranks.get(parts[i] + ' ' + parts[i + 1]);
          if (r !== undefined && r < bestRank) { bestRank = r; best = i; }
        }
        if (best < 0) break;
        parts.splice(best, 2, parts[best] + parts[best + 1]);
      }
      const ids = parts.map(p => this.vocab[p]).filter(v => v !== undefined);
      this.cache.set(word, ids);
      return ids;
    }
    encode(text) {
      const out = [];
      for (const m of text.matchAll(this.pat)) {
        const bytes = this.enc.encode(m[0]);
        out.push(...this.bpe(Array.from(bytes, b => this.b2u[b]).join('')));
      }
      return out;
    }
    decode(ids) {
      const bytes = [];
      for (const id of ids) {
        const s = this.inv[id];
        if (s === undefined || s === '<|eos|>') continue;
        for (const ch of s) bytes.push(this.u2b[ch]);
      }
      return this.dec.decode(new Uint8Array(bytes)).replace(/�/g, '');
    }
  }

  // ---------- lexicon: dictionary-guided decoding ----------
  // A token may only be chosen if every word it touches stays on the way to a real
  // word, and a word may only end once it is complete. The model still ranks tokens;
  // the lexicon only removes the ones that would build non-words.
  class Lexicon {
    constructor(words, tok) {
      this.words = new Set(); this.prefixes = new Set();
      for (const w of words) this.add(w);
      // decoded text of every single token; null when it is not valid text on its own
      const strict = new TextDecoder('utf-8', { fatal: true });
      this.text = tok.inv.map((s, id) => {
        if (id === 0 || s === undefined) return null;
        try { return strict.decode(new Uint8Array(Array.from(s, ch => tok.u2b[ch]))); } catch (e) { return null; }
      });
    }
    add(w) {
      w = w.toLowerCase();
      if (!w) return;
      this.words.add(w);
      for (let i = 1; i <= w.length; i++) this.prefixes.add(w.slice(0, i));
    }
    // Returns the new state after appending text, or null if the text would build a
    // non-word, use a disallowed character, or produce messy punctuation.
    // state = { w: partial word, last: last character emitted }
    advance(state, text, extra) {
      let w = state.w, last = state.last;
      for (const ch of text) {
        if (/[\p{L}']/u.test(ch)) {
          if (/[,.?!]/.test(last)) return null;          // a space must follow punctuation
          if (w && ch !== ch.toLowerCase()) return null; // no capitals inside a word
          if (ch === "'" && !w) return null;             // apostrophe only inside a word
          w += ch.toLowerCase();
          if (!this.prefixes.has(w) && !(extra && extra.prefixes.has(w))) return null;
        } else if (ch === ' ') {
          if (w && !this.complete(w, extra)) return null;
          if (last === ' ') return null;                 // no double spaces
          w = '';
        } else if (/[,.?!]/.test(ch)) {
          if (w && !this.complete(w, extra)) return null;
          if (!w && !/[\p{L}]/u.test(last)) return null; // punctuation only right after a word
          w = '';
        } else {
          return null;                                   // digits, hyphens, symbols, newlines
        }
        last = ch;
      }
      return { w, last };
    }
    complete(w, extra) { return this.words.has(w) || !!(extra && extra.words.has(w)); }
  }

  // ---------- chat session, base model ----------
  // Pebble-10M (base) predicts the next token normally but is not chat-tuned, so it is
  // steered with a role-play header and one example chat. The live chat follows it.
  // History is kept as the exact tokens the model has already processed, so a new turn
  // only feeds the new message; when the 512-token context fills, the window slides and
  // the recent turns are replayed once from a snapshot taken after the header.
  const PERSONA =
    'F Corp Customer Care. Live chat.\n' +
    'The associate is calm, warm and brief, keeps the customer on the subject of their account, ' +
    'and transfers them to a specialist when it cannot help.\n\n' +
    'Customer: Hello? Is anyone there?\n' +
    'Associate: Hello. I am here for you. How can I help you today?\n' +
    'Customer: My bill is higher than last month.\n' +
    'Associate: I understand. Let me take a look at your account for you.\n' +
    'Customer: What is your favourite movie?\n' +
    'Associate: I am here to help with your account. Is there anything else about your bill?\n' +
    'Customer: I want to cancel everything.\n' +
    'Associate: Of course. I will transfer you to a specialist who can help with that.\n\n' +
    'Customer: Hi.\n' +
    'Associate: Welcome to F Corp. What brings you here today?\n';
  const TRANSFER = /\b(transfer\w*|escalat\w*)\b/i;
  const TRANSFER_LINES = [
    'I will transfer you to a specialist who can help you with that.',
    'Thank you for your patience. I will transfer you to a specialist now.',
    'I understand. Let me transfer you to a colleague who can take this further.',
  ];

  class ChatSession {
    constructor(model, tok, lex, opt) {
      this.model = model; this.tok = tok; this.lex = lex;
      this.o = Object.assign({ maxTokens: 24, maxUserTokens: 60, minWords: 4, candidates: 3, temperature: 0.45, topK: 40, topP: 0.9, repetitionPenalty: 1.2 }, opt);
      this.nl = tok.encode('\n');
      this.assoc = tok.encode('Associate:');
      model.reset();
      this.head = [0, ...tok.encode(PERSONA)];
      for (const t of this.head) model.step(t);
      this.base = model.snapshot();
      this.reset();
    }
    reset() { this.lines = []; this.said = []; this.heard = []; this.fed = this.head.slice(); this.model.restore(this.base); }

    // make the model's state equal to `tokens`, feeding only what is new
    sync(tokens) {
      let common = 0;
      while (common < this.fed.length && common < tokens.length && this.fed[common] === tokens[common]) common++;
      if (common < this.fed.length) { this.model.restore(this.base); this.fed = this.head.slice(); common = this.head.length; }
      let logits = null;
      for (let i = common; i < tokens.length; i++) { logits = this.model.step(tokens[i]); this.fed.push(tokens[i]); }
      return Float32Array.from(logits || this.model.logits);
    }

    // one sampling pass from the current state; seed = tokens the reply must start with
    generate(logits, seed, penalize, temperature, onText) {
      const { tok, lex, o } = this;
      const gen = [];
      let text = '', state = { w: '', last: '' }, ended = false, logp = 0, steps = 0;
      for (const t of seed) {
        gen.push(t); this.fed.push(t);
        logits = Float32Array.from(this.model.step(t));
        if (lex) state = lex.advance(state, lex.text[t]) || state;
      }
      if (seed.length) text = tok.decode(gen);
      for (let i = 0; i < o.maxTokens && this.model.pos < CTX - 2; i++) {
        let masked = logits;
        if (lex) {
          masked = Float32Array.from(logits);
          let any = false;
          for (let id = 0; id < masked.length; id++) {
            const s = lex.text[id];
            const ok = id === 0 ? (!state.w || lex.complete(state.w)) : (s !== null && lex.advance(state, s) !== null);
            if (!ok) masked[id] = -Infinity; else any = true;
          }
          if (!any) break;
        }
        const t = sample(masked, [...penalize, ...gen.slice(-24)], Object.assign({}, o, { temperature }));
        if (t === 0) { ended = true; break; }
        const next = tok.decode([...gen, t]);
        if (/\n|Customer:|Associate:/.test(next)) { ended = true; break; }
        // how confident the model was in this token (log-probability, among allowed tokens)
        let mx = -Infinity; for (let id = 0; id < masked.length; id++) if (masked[id] > mx) mx = masked[id];
        let z = 0; for (let id = 0; id < masked.length; id++) if (masked[id] > -Infinity) z += Math.exp(masked[id] - mx);
        logp += masked[t] - mx - Math.log(z); steps++;
        gen.push(t); this.fed.push(t);
        logits = Float32Array.from(this.model.step(t));
        if (lex) state = lex.advance(state, lex.text[t]) || state;
        text = next; if (onText) onText(text.trim());
        if (/[.!?]\s*$/.test(text) && text.trim().split(/\s+/).length >= o.minWords) { ended = true; break; }
      }
      if (lex && !ended && state.w && !lex.complete(state.w)) text = text.slice(0, text.length - state.w.length);
      text = text.trim().replace(/[,\s]+$/, '');
      if (!/\p{L}/u.test(text)) text = '';
      if (text) {
        if (!/[.!?]$/.test(text)) text += '.';
        text = text.replace(/\bi\b/g, 'I').replace(/^\p{Ll}/u, c => c.toUpperCase());
      }
      return { text, gen, fluency: steps ? logp / steps : -Infinity };
    }

    reply(userText, onText, opt) {
      const { tok, o } = this;
      const force = !!(opt && opt.forceTransfer);
      let user = tok.encode('Customer: ' + userText.replace(/\s+/g, ' ').trim());
      if (user.length > o.maxUserTokens) user = user.slice(0, o.maxUserTokens);
      const userLine = [...user, ...this.nl];
      // sliding window: keep the most recent lines that fit; when it must slide, keep
      // only about half the room so the next few turns can append without a replay
      const room = CTX - this.head.length - userLine.length - this.assoc.length - o.maxTokens - 8 - this.nl.length - 4;
      const size = ls => ls.reduce((n, l) => n + l.tokens.length, 0);
      if (size(this.lines) > room) {
        while (this.lines.length && size(this.lines) > room / 2) this.lines.shift();
      }
      const tokens = [...this.head, ...this.lines.flatMap(l => l.tokens), ...userLine, ...this.assoc];
      const logits = this.sync(tokens);
      const snap = this.model.snapshot(), fedLen = this.fed.length;
      const rewind = () => { this.model.restore(snap); this.fed.length = fedLen; };

      const said = this.lines.filter(l => l.said);
      const penalize = said.slice(-3).flatMap(l => l.gen);
      const norm = t => t.toLowerCase().replace(/[^\p{L}' ]/gu, '').trim();
      // anything already said in this chat, by either side, counts as a repeat
      const seen = new Set([norm(userText), ...this.said.map(norm), ...this.heard.map(norm)]);
      // near-repeats too: most of its words already used in one earlier reply
      const words = t => new Set(norm(t).split(/\s+/).filter(w => w.length > 2));
      const overlap = (a, b) => { const A = words(a), B = words(b); if (!A.size) return 0; let n = 0; for (const w of A) if (B.has(w)) n++; return n / A.size; };
      const repeats = t => !t || seen.has(norm(t)) || [...this.said, ...this.heard, userText].some(p => overlap(t, p) >= 0.6);
      // an unprompted transfer on the very first reply would end the chat before it starts
      const tooSoon = t => said.length === 0 && TRANSFER.test(t);
      // score: the model's own confidence, nudged toward addressing the customer
      // and away from speaking as the customer
      const score = c => c.fluency
        + (/\byou(r|rs)?\b/i.test(c.text) ? 0.35 : 0)
        - (/\b(my (bill|bills|account|password|plan|data|money|job)|I cannot|I can't|I need|I tried|I think I)\b/i.test(c.text) ? 1 : 0);

      let r = null;
      if (!force) {
        // a few candidate replies from the same point; keep the best one that is not a repeat
        const cands = [];
        for (let k = 0; k < o.candidates; k++) {
          rewind();
          const c = this.generate(logits, [], k ? penalize : [], o.temperature + 0.1 * k, null);
          if (!repeats(c.text) && !tooSoon(c.text)) cands.push(c);
        }
        cands.sort((a, b) => score(b) - score(a));
        r = cands[0] || null;
        if (!r && said.length === 0) r = { gen: tok.encode(' I am here for you. How can I help you today?'), text: 'I am here for you. How can I help you today?' };
      }
      // the out: when told to, or when every candidate was a repeat, the customer is
      // transferred, in one of a few fixed wordings, because this line must always land
      if (!r) {
        const line = TRANSFER_LINES[Math.floor((o.rand || Math.random)() * TRANSFER_LINES.length)];
        r = { text: line, gen: tok.encode(' ' + line) };
      }
      // replay the chosen reply so the model's state matches what was said
      rewind();
      for (const t of r.gen) { this.model.step(t); this.fed.push(t); }
      if (onText) onText(r.text);
      this.said.push(r.text); this.heard.push(userText);
      this.lines.push({ tokens: userLine });
      this.lines.push({ tokens: [...this.assoc, ...r.gen, ...this.nl], gen: r.gen, said: r.text });
      return { text: r.text, transfer: TRANSFER.test(r.text) };
    }
  }

  const api = { Pebble, Tokenizer, Lexicon, ChatSession, loadWeights, sample, CTX, EOS: 0 };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.PebbleLM = api;
})(typeof self !== 'undefined' ? self : this);
