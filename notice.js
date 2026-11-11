/* F Corp cookie notice. True as written: this site sets no cookies and stores nothing,
   so dismissing the notice is not remembered and it returns on the next visit. */
(() => {
  const bar = document.createElement('aside');
  bar.className = 'fnotice';
  bar.setAttribute('role', 'region');
  bar.setAttribute('aria-label', 'Cookie notice');
  bar.innerHTML = `
    <p><b>F Corp does not use cookies.</b> We don't need to.
    <small>Nothing is stored, so this returns each visit.</small></p>
    <button type="button" class="btn">I understand</button>`;
  bar.hidden = true;
  document.body.appendChild(bar);
  bar.querySelector('button').addEventListener('click', () => {
    bar.classList.add('leaving');
    setTimeout(() => { bar.hidden = true; }, 400);
  });
  setTimeout(() => { bar.hidden = false; }, 1800);
})();
