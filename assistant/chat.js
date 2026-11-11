/* F Corp chat bubble. The associate (Pebble-10M) loads only when the bubble is first opened and
   runs in a Web Worker on the visitor's device. It can hold a short conversation; it transfers the
   customer when it decides to, when asked for something it is not allowed to do, or at the reply limit.
   On devices with WebGPU, after the second reply, the visitor may connect a senior associate
   (Qwen2.5-0.5B-Instruct via WebLLM, about 276 MB, downloaded once). Pebble keeps answering until
   it is ready, and takes over again if it fails. */
(() => {
  const BASE = new URL('.', document.currentScript.src).href;
  const MAX_REPLIES = 6;
  const SENIOR_MB = 276;
  // requests and moods an associate never handles itself: transfer straight away
  // (keep in step with ESCALATE in senior.js)
  const ESCALATE = /\b(cancel|close (my|the) account|delete|unsubscribe|manager|supervisor|speak to a human|talk to a human|complain|lawyer|legal|worst|terrible|awful|unacceptable|useless|ridiculous|furious)\b/i;
  const MARK = '<svg viewBox="28 26 48 48" aria-hidden="true"><rect x="34" y="26" width="10" height="48"/><rect x="48" y="26" width="22" height="10"/><rect x="48" y="45" width="15" height="10"/></svg>';
  const CLOSING = 'Transferred. A specialist will be in touch when convenient.';
  const FALLBACK = ['I understand.', 'That has been noted.', 'Thank you for sharing that with us.'];

  const root = document.createElement('div');
  root.className = 'fchat';
  root.innerHTML = `
    <button class="fchat-bubble" type="button" aria-expanded="false" aria-controls="fchat-panel" aria-label="Chat with F Corp">${MARK}</button>
    <section class="fchat-panel" id="fchat-panel" role="dialog" aria-label="F Corp associate" hidden>
      <header class="fchat-head">
        <span class="fchat-mark">${MARK}</span>
        <span class="fchat-title"><b>F Corp</b><small id="fchat-status">Contentment Associate</small></span>
        <button class="fchat-x" type="button" aria-label="Close chat">&times;</button>
      </header>
      <div class="fchat-progress" hidden><i></i></div>
      <ol class="fchat-log" aria-live="polite"></ol>
      <form class="fchat-form" autocomplete="off">
        <input class="fchat-in" maxlength="200" placeholder="Preparing…" aria-label="Message" disabled>
        <button class="fchat-send" type="submit" aria-label="Send" disabled>&rarr;</button>
      </form>
      <p class="fchat-note">Runs on your device. Nothing you type leaves it.</p>
    </section>`;
  document.body.appendChild(root);

  const $ = s => root.querySelector(s);
  const bubble = $('.fchat-bubble'), panel = $('.fchat-panel'), log = $('.fchat-log');
  const form = $('.fchat-form'), input = $('.fchat-in'), send = $('.fchat-send');
  const status = $('#fchat-status'), bar = $('.fchat-progress'), barFill = $('.fchat-progress i');

  let worker = null, ready = false, busy = false, replies = 0, seq = 0, pending = null;
  // senior associate: null = not offered yet, 'offered', 'loading', 'ready', 'failed'
  let senior = null, seniorState = null, offered = false;

  function add(who, text, cls) {
    const li = document.createElement('li');
    li.className = `fchat-msg ${who}${cls ? ' ' + cls : ''}`;
    li.textContent = text;
    log.appendChild(li);
    log.scrollTop = log.scrollHeight;
    return li;
  }
  function note(text) { return add('sys', text, 'note'); }
  function greet() {
    log.innerHTML = '';
    replies = 0;
    add('them', 'Hello. How may I help?');
  }
  function setInput(on, placeholder) {
    input.disabled = send.disabled = !on;
    input.placeholder = placeholder;
  }
  function setStatus() {
    status.textContent = seniorState === 'ready' ? 'Senior associate · on device' : 'Associate · on device';
  }
  function closeConversation() {
    add('them', CLOSING, 'closing');
    setInput(false, 'Chat closed');
    const li = document.createElement('li');
    li.className = 'fchat-msg sys';
    li.innerHTML = '<button type="button" class="fchat-new">New chat</button>';
    log.appendChild(li);
    li.querySelector('button').addEventListener('click', () => {
      worker.postMessage({ type: 'new' });
      if (senior) senior.postMessage({ type: 'new' });
      greet(); setInput(true, 'Message'); input.focus();
    });
    log.scrollTop = log.scrollHeight;
  }

  // one handler for replies from either associate
  function onReply(m) {
    if (!pending || m.id !== pending.id) return;
    pending.li.classList.remove('typing');
    pending.li.textContent = m.text || FALLBACK[Math.floor(Math.random() * FALLBACK.length)];
    pending = null; busy = false; replies++;
    if (m.transfer || replies >= MAX_REPLIES) closeConversation();
    else { setInput(true, 'Message'); input.focus(); if (replies === 2) offerSenior(); }
    log.scrollTop = log.scrollHeight;
  }

  function startWorker() {
    worker = new Worker(BASE + 'worker.js');
    bar.hidden = false;
    status.textContent = 'Preparing…';
    worker.onmessage = (e) => {
      const m = e.data;
      if (m.type === 'progress') {
        barFill.style.width = (100 * m.got / m.total).toFixed(1) + '%';
        status.textContent = `Preparing · ${(m.got / 1048576).toFixed(1)} of ${(m.total / 1048576).toFixed(1)} MB`;
      } else if (m.type === 'ready') {
        ready = true; bar.hidden = true;
        setStatus();
        setInput(true, 'Message');
        if (!panel.hidden) input.focus();
        probeSenior();
      } else if (m.type === 'reply') {
        onReply(m);
      } else if (m.type === 'error') {
        console.warn('F Corp associate:', m.message);
        bar.hidden = true; busy = false;
        status.textContent = 'Unavailable';
        add('them', 'Unavailable. This has been noted.', 'closing');
        setInput(false, 'Unavailable');
      }
    };
    worker.postMessage({ type: 'load' });
  }

  // ---------- senior associate (WebLLM, WebGPU) ----------
  // check for WebGPU quietly at load; the offer itself waits until the visitor is chatting
  let seniorAvail = null;
  async function probeSenior() {
    if (!navigator.gpu) return;
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (adapter) seniorAvail = { f16: adapter.features.has('shader-f16') };
    } catch (e) { seniorAvail = null; }
  }
  function offerSenior() {
    if (offered || !seniorAvail || seniorState) return;
    offered = true;
    seniorState = 'offered';
    const li = document.createElement('li');
    li.className = 'fchat-msg sys offer';
    const p = document.createElement('p');
    p.textContent = `Senior associate available · ${SENIOR_MB} MB, one-time download.`;
    p.title = 'Qwen2.5-0.5B via WebLLM, fetched from Hugging Face and kept in your browser cache. Your messages stay on your device.';
    const b = document.createElement('button');
    b.type = 'button'; b.className = 'fchat-new'; b.textContent = 'Connect';
    b.addEventListener('click', () => { li.remove(); connectSenior(seniorAvail.f16); });
    li.append(p, b);
    log.appendChild(li);
    log.scrollTop = log.scrollHeight;
  }

  function seniorFailed(message) {
    console.warn('F Corp senior associate:', message);
    seniorState = 'failed';
    bar.hidden = true;
    if (senior) { senior.terminate(); senior = null; }
    setStatus();
    note('Senior associate unavailable.');
  }

  function connectSenior(f16) {
    seniorState = 'loading';
    try {
      senior = new Worker(BASE + 'senior-worker.js', { type: 'module' });
    } catch (e) { seniorFailed(e && e.message || e); return; }
    bar.hidden = false; barFill.style.width = '0%';
    senior.onerror = (e) => seniorFailed(e.message || 'worker error');
    senior.onmessage = (e) => {
      const m = e.data;
      if (m.type === 'progress') {
        barFill.style.width = (100 * m.progress).toFixed(1) + '%';
        status.textContent = `Senior associate · ${Math.round(100 * m.progress)}%`;
      } else if (m.type === 'ready') {
        seniorState = 'ready'; bar.hidden = true;
        setStatus();
        note('Senior associate joined.');
      } else if (m.type === 'reply') {
        onReply(m);
      } else if (m.type === 'error') {
        const lost = pending && m.id === pending.id ? pending : null;
        seniorFailed(m.message);
        // the senior failed mid-reply: Pebble answers that message instead
        if (lost) worker.postMessage({ type: 'reply', id: lost.id, text: lost.text, forceTransfer: lost.forceTransfer });
      }
    };
    senior.postMessage({ type: 'load', f16 });
  }

  function open() {
    panel.hidden = false;
    bubble.setAttribute('aria-expanded', 'true');
    root.classList.add('open');
    if (!worker) { greet(); startWorker(); }
    if (ready && !input.disabled) input.focus();
  }
  function close() {
    panel.hidden = true;
    bubble.setAttribute('aria-expanded', 'false');
    root.classList.remove('open');
    bubble.focus();
  }

  bubble.addEventListener('click', () => (panel.hidden ? open() : close()));
  $('.fchat-x').addEventListener('click', close);
  panel.addEventListener('keydown', e => { if (e.key === 'Escape') close(); });
  form.addEventListener('submit', (e) => {
    e.preventDefault();
    const text = input.value.trim();
    if (!text || !ready || busy || replies >= MAX_REPLIES) return;
    input.value = '';
    add('me', text);
    busy = true;
    setInput(false, 'Responding…');
    const forceTransfer = replies >= MAX_REPLIES - 1 || ESCALATE.test(text);
    pending = { id: ++seq, li: add('them', '', 'typing'), text, forceTransfer };
    const target = seniorState === 'ready' ? senior : worker;
    target.postMessage({ type: 'reply', id: pending.id, text, forceTransfer });
  });
})();
