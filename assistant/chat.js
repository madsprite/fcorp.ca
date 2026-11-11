/* F Corp chat bubble. The associate (Pebble-10M-Chat) loads only when the bubble is first opened,
   runs in a Web Worker on the visitor's device, and handles at most two replies per conversation. */
(() => {
  const BASE = new URL('.', document.currentScript.src).href;
  const MAX_REPLIES = 2;
  const MARK = '<svg viewBox="28 26 48 48" aria-hidden="true"><rect x="34" y="26" width="10" height="48"/><rect x="48" y="26" width="22" height="10"/><rect x="48" y="45" width="15" height="10"/></svg>';
  const CLOSING = 'Thank you. Your conversation has been reassigned to a specialist, who will contact you when it is convenient.';
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
        <input class="fchat-in" maxlength="200" placeholder="Preparing your associate…" aria-label="Message" disabled>
        <button class="fchat-send" type="submit" aria-label="Send" disabled>&rarr;</button>
      </form>
      <p class="fchat-note">Your associate runs entirely on your device. Nothing you type leaves it. We don't need it to.</p>
    </section>`;
  document.body.appendChild(root);

  const $ = s => root.querySelector(s);
  const bubble = $('.fchat-bubble'), panel = $('.fchat-panel'), log = $('.fchat-log');
  const form = $('.fchat-form'), input = $('.fchat-in'), send = $('.fchat-send');
  const status = $('#fchat-status'), bar = $('.fchat-progress'), barFill = $('.fchat-progress i');

  let worker = null, ready = false, busy = false, replies = 0, seq = 0, pending = null;

  function add(who, text, cls) {
    const li = document.createElement('li');
    li.className = `fchat-msg ${who}${cls ? ' ' + cls : ''}`;
    li.textContent = text;
    log.appendChild(li);
    log.scrollTop = log.scrollHeight;
    return li;
  }
  function greet() {
    log.innerHTML = '';
    replies = 0;
    add('them', 'Hello. I am here for you.');
    add('them', 'How may I help you today?');
  }
  function setInput(on, placeholder) {
    input.disabled = send.disabled = !on;
    input.placeholder = placeholder;
  }
  function closeConversation() {
    add('them', CLOSING, 'closing');
    setInput(false, 'This conversation has been reassigned');
    const li = document.createElement('li');
    li.className = 'fchat-msg sys';
    li.innerHTML = '<button type="button" class="fchat-new">Begin a new conversation</button>';
    log.appendChild(li);
    li.querySelector('button').addEventListener('click', () => { greet(); setInput(true, 'Type a message'); input.focus(); });
    log.scrollTop = log.scrollHeight;
  }

  function startWorker() {
    worker = new Worker(BASE + 'worker.js');
    bar.hidden = false;
    status.textContent = 'Preparing your associate';
    worker.onmessage = (e) => {
      const m = e.data;
      if (m.type === 'progress') {
        barFill.style.width = (100 * m.got / m.total).toFixed(1) + '%';
        status.textContent = `Preparing your associate · ${(m.got / 1048576).toFixed(1)} of ${(m.total / 1048576).toFixed(1)} MB`;
      } else if (m.type === 'ready') {
        ready = true; bar.hidden = true;
        status.textContent = 'Contentment Associate · on your device';
        setInput(true, 'Type a message');
        if (!panel.hidden) input.focus();
      } else if ((m.type === 'partial' || m.type === 'reply') && pending && m.id === pending.id) {
        pending.li.classList.remove('typing');
        if (m.text) pending.li.textContent = m.text;
        if (m.type === 'reply') {
          if (!m.text) pending.li.textContent = FALLBACK[Math.floor(Math.random() * FALLBACK.length)];
          pending = null; busy = false; replies++;
          if (replies >= MAX_REPLIES) closeConversation();
          else { setInput(true, 'Type a message'); input.focus(); }
        }
        log.scrollTop = log.scrollHeight;
      } else if (m.type === 'error') {
        bar.hidden = true; busy = false;
        status.textContent = 'Associate unavailable';
        add('them', 'Your associate is currently unavailable. This has been noted.', 'closing');
        setInput(false, 'Associate unavailable');
      }
    };
    worker.postMessage({ type: 'load' });
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
    setInput(false, 'Your associate is responding');
    pending = { id: ++seq, li: add('them', '', 'typing') };
    worker.postMessage({ type: 'reply', id: pending.id, text });
  });
})();
