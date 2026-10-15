'use strict';
(() => {
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const NS = 'http://www.w3.org/2000/svg';
const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
const fmt = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
const svgEl = (tag, attrs = {}, parent) => {
  const el = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  if (parent) parent.appendChild(el);
  return el;
};
const imageCache = new Map();
const loadImage = (src) => {
  if (!imageCache.has(src)) {
    imageCache.set(src, new Promise((res, rej) => {
      const im = new Image();
      im.onload = () => res(im);
      im.onerror = () => { imageCache.delete(src); rej(new Error(src)); };
      im.src = src;
    }));
  }
  return imageCache.get(src);
};
const whenVisible = (el, onIn, onOut, threshold = 0.35) => {
  new IntersectionObserver((entries) => entries.forEach((e) => (e.isIntersecting ? onIn() : onOut && onOut())), { threshold }).observe(el);
};

// Below-the-fold videos get their src only once they come within ~one screen of the viewport, so on a slow link
// the teaser is not competing with clips the reader has not reached. Created before the play/pause observers so
// the src is in place when they fire.
const nearObs = new IntersectionObserver((entries) => entries.forEach((e) => {
  if (!e.isIntersecting) return;
  const v = e.target;
  v.near = true; nearObs.unobserve(v);
  if (v.pendingSrc) v.src = v.pendingSrc;
}), { rootMargin: '100% 0px' });
const lazySrc = (v, src) => {
  if (!v.observed) { v.observed = true; nearObs.observe(v); }
  v.pendingSrc = src;
  if (v.near) v.src = src;
};

/* ---------------- video scrub bars ---------------- */
// Play/pause + draggable progress bar for a video; v.userPaused keeps autoplay from overriding the reader.
// span() gives the end of the stretch the bar covers (default: the whole video); toReal converts file time to
// the robot time shown on the clock.
function videoBar(v, bar, span = () => v.duration || 0, toReal = 1) {
  const btn = bar.querySelector('.vbar-btn'), range = bar.querySelector('.vbar-range'), clock = bar.querySelector('.vbar-clock');
  let dragging = false, resume = false, raf = 0;
  const sync = () => {
    const d = span(), t = Math.min(v.currentTime || 0, d), p = d ? t / d : 0;
    if (!dragging) range.value = Math.round(p * 1000);
    range.style.setProperty('--p', `${p * 100}%`);
    clock.textContent = `${fmt(t * toReal)} / ${fmt(d * toReal)}`;
  };
  const tick = () => { sync(); raf = v.paused ? 0 : requestAnimationFrame(tick); };
  const syncBtn = () => {
    bar.classList.toggle('playing', !v.paused);
    btn.setAttribute('aria-label', v.paused ? 'Play' : 'Pause');
  };
  v.addEventListener('play', () => { syncBtn(); if (!raf) tick(); });
  v.addEventListener('pause', () => { syncBtn(); sync(); });
  ['loadedmetadata', 'seeked', 'timeupdate', 'emptied'].forEach((e) => v.addEventListener(e, sync));
  btn.addEventListener('click', () => {
    if (v.paused) { v.userPaused = false; v.play().catch(() => {}); } else { v.userPaused = true; v.pause(); }
  });
  const seek = () => { if (v.duration) v.currentTime = Math.min((range.value / 1000) * span(), v.duration - 0.05); sync(); };
  range.addEventListener('pointerdown', () => { dragging = true; resume = !v.paused; v.pause(); });
  range.addEventListener('input', seek);
  const end = () => { if (!dragging) return; dragging = false; seek(); if (resume) v.play().catch(() => {}); };
  range.addEventListener('pointerup', end);
  range.addEventListener('pointercancel', end);
  range.addEventListener('change', end);
  range.addEventListener('keydown', (e) => {
    const k = { ArrowLeft: -2, ArrowRight: 2, Home: -1e9, End: 1e9 }[e.key];
    if (k === undefined || !v.duration) return;
    e.preventDefault();
    v.currentTime = Math.min(Math.max(Math.min(v.currentTime, span()) + k, 0), span(), v.duration - 0.05);
  });
  syncBtn(); sync();
  return sync;
}

/* ---------------- teaser ---------------- */
const teaser = $('#teaser');
videoBar(teaser, $('#teaser-bar'));
teaser.addEventListener('click', () => $('#teaser-bar .vbar-btn').click());
whenVisible(teaser, () => { if (!reduced && !teaser.userPaused) teaser.play().catch(() => {}); }, () => teaser.pause(), 0.25);

/* ---------------- table of contents ---------------- */
const tocLinks = $$('.toc a');
const tocTargets = tocLinks.map((a) => $(a.getAttribute('href'))).filter(Boolean);
const tocObs = new IntersectionObserver(() => {
  let current = tocTargets[0];
  for (const t of tocTargets) if (t.getBoundingClientRect().top < innerHeight * 0.35) current = t;
  tocLinks.forEach((a) => a.classList.toggle('active', a.getAttribute('href') === '#' + current.id));
}, { rootMargin: '0px 0px -50% 0px', threshold: [0, 1] });
tocTargets.forEach((t) => tocObs.observe(t));
addEventListener('scroll', () => {
  let current = tocTargets[0];
  for (const t of tocTargets) if (t.getBoundingClientRect().top < innerHeight * 0.35) current = t;
  tocLinks.forEach((a) => a.classList.toggle('active', a.getAttribute('href') === '#' + current.id));
}, { passive: true });

/* ---------------- rollout viewer ---------------- */
const EP = window.EPISODES;
const SRC_W = 1280, SRC_H = 720, FPS = 10, STEPS = 8;
// Recordings store one frame per control step at 10 fps, but the robot ran at 15 Hz.
const REAL_HZ = 15, REAL_SPEED = 1.5;
const stage = $('#stage');
const video = $('#ep-video');
const overlay = $('#ep-overlay');
const insetsEl = $('#insets');
const timeInput = $('#ep-time');
const playBtn = $('#ep-play');
const spark = $('#spark');
const state = { ep: EP[0], mode: 'rgb', speed: REAL_SPEED * REAL_HZ / FPS, showBox: true, atlases: [], shown: [], targets: [], tween: [], lastQ: -1, views: [], userPaused: false };

function hullPairs(b, r) {
  const bc = [[b.x, b.y], [b.x + b.s, b.y], [b.x, b.y + b.s], [b.x + b.s, b.y + b.s]];
  const ic = [[r.x, r.y], [r.x + r.w, r.y], [r.x, r.y + r.h], [r.x + r.w, r.y + r.h]];
  const pts = bc.concat(ic);
  const out = [];
  for (let i = 0; i < 4; i++) {
    const [ax, ay] = bc[i], [bx, by] = ic[i];
    const side = pts.map(([px, py]) => (bx - ax) * (py - ay) - (by - ay) * (px - ax));
    if (side.every((v) => v >= -1e-6) || side.every((v) => v <= 1e-6)) out.push([bc[i], ic[i]]);
  }
  return out.slice(0, 2);
}

function buildStage() {
  const ep = state.ep, n = ep.views;
  stage.style.setProperty('--ar', `${n * 16} / 9`);
  overlay.setAttribute('viewBox', `0 0 ${SRC_W * n} ${SRC_H}`);
  overlay.innerHTML = '';
  insetsEl.innerHTML = '';
  state.views = [];
  const defs = svgEl('defs', {}, overlay);
  for (let v = 0; v < n; v++) {
    const cp = svgEl('clipPath', { id: `cv${v}` }, defs);
    svgEl('rect', { x: v * SRC_W, y: 0, width: SRC_W, height: SRC_H }, cp);
    const g = svgEl('g', { 'clip-path': `url(#cv${v})` }, overlay);
    const dim = svgEl('path', { 'fill-rule': 'evenodd', fill: 'rgba(0,0,0,0.3)' }, g);
    const outline = svgEl('rect', { fill: 'none', stroke: '#00b99a', 'stroke-opacity': '0.6', 'stroke-width': '1.5', 'vector-effect': 'non-scaling-stroke', rx: 4 }, g);
    const brackets = svgEl('path', { fill: 'none', stroke: '#00b99a', 'stroke-width': '3.5', 'stroke-linecap': 'round', 'vector-effect': 'non-scaling-stroke' }, g);
    const lines = [0, 1].map(() => svgEl('line', { stroke: '#00b99a', 'stroke-opacity': '0.75', 'stroke-width': '1.4', 'vector-effect': 'non-scaling-stroke' }, overlay));
    const box = document.createElement('div');
    box.className = 'inset';
    const width = n === 1 ? 27 : 15.5;
    box.style.width = `${width}%`;
    box.style.left = `${((v + 1) * (100 / n)) - width - (n === 1 ? 2.2 : 1.2)}%`;
    const cv = document.createElement('canvas');
    cv.width = cv.height = 224;
    const tag = document.createElement('span');
    tag.className = 'tag';
    box.append(cv, tag);
    insetsEl.appendChild(box);
    state.views.push({ dim, outline, brackets, lines, box, cv, ctx: cv.getContext('2d'), tag, rect: null });
  }
  measureInsets();
}

function measureInsets() {
  const sr = stage.getBoundingClientRect();
  if (!sr.width) return;
  const kx = (SRC_W * state.ep.views) / sr.width, ky = SRC_H / sr.height;
  state.views.forEach((vw) => {
    const r = vw.box.getBoundingClientRect();
    vw.rect = { x: (r.left - sr.left) * kx, y: (r.top - sr.top) * ky, w: r.width * kx, h: r.height * ky };
  });
}
addEventListener('resize', () => { measureInsets(); state.lastQ = -1; });

async function loadAtlases() {
  const ep = state.ep, mode = state.mode, token = ep.id + mode;
  state.atlasToken = token;
  const imgs = await Promise.all(Array.from({ length: ep.views }, (_, v) => loadImage(`assets/${ep.id}-${mode}-${v}.webp`).catch(() => null)));
  if (state.atlasToken !== token) return;
  state.atlases = imgs;
  state.lastQ = -1;
  draw(true);
}

function queryAt(t) {
  const frame = Math.floor(t * FPS + 1e-4);
  return Math.max(0, Math.min(state.ep.boxes[0].length - 1, Math.floor(frame / STEPS)));
}

function targetBox(v, q) {
  const r = state.ep.boxes[v][q];
  if (q === 0) return { x: v * SRC_W, y: -280, s: SRC_W, full: true };
  return { x: v * SRC_W + r[0], y: r[1], s: r[2], full: false };
}

function draw(force) {
  const ep = state.ep;
  const q = queryAt(video.currentTime || 0);
  const now = performance.now();
  if (q !== state.lastQ || force) {
    ep.boxes.forEach((_, v) => {
      const tgt = targetBox(v, q);
      const from = state.shown[v] || tgt;
      state.tween[v] = { from: { ...from }, to: tgt, t0: now, dur: force || Math.abs(q - state.lastQ) > 1 ? 0 : 180 };
      const vw = state.views[v];
      const atlas = state.atlases[v];
      if (atlas && q < ep.crops) {
        vw.ctx.drawImage(atlas, (q % 10) * 224, Math.floor(q / 10) * 224, 224, 224, 0, 0, 224, 224);
      }
      const zoom = tgt.full ? 1 : SRC_W / tgt.s;
      vw.tag.textContent = tgt.full ? (ep.views > 1 ? 'Full view' : 'Initial query: full view') : `${ep.views > 1 ? `Cam ${v + 1}` : (state.mode === 'rgb' ? 'Foveal RGB' : 'Foveal depth')} ×${zoom.toFixed(1)}`;
    });
    const t0 = targetBox(0, q);
    $('#ro-query').textContent = q + 1;
    $('#ro-size').textContent = t0.full ? 'full' : `${Math.round(t0.s)}`;
    $('#ro-zoom').textContent = `${(t0.full ? 1 : SRC_W / t0.s).toFixed(1)}×`;
    state.lastQ = q;
  }
  ep.boxes.forEach((_, v) => {
    const tw = state.tween[v];
    if (!tw) return;
    const k = tw.dur ? Math.min(1, (now - tw.t0) / tw.dur) : 1;
    const e = k * k * (3 - 2 * k);
    const b = { x: tw.from.x + (tw.to.x - tw.from.x) * e, y: tw.from.y + (tw.to.y - tw.from.y) * e, s: tw.from.s + (tw.to.s - tw.from.s) * e };
    state.shown[v] = { ...b, full: tw.to.full };
    paintBox(v, b, tw.to.full);
  });
  const dur = video.duration || ep.frames / FPS;
  const p = (video.currentTime || 0) / dur;
  if (!scrubbing) timeInput.value = Math.round(p * 1000);
  timeInput.style.setProperty('--p', `${p * 100}%`);
  const real = FPS / REAL_HZ; // file seconds -> robot seconds
  $('#ep-clock').textContent = `${fmt((video.currentTime || 0) * real)} / ${fmt(dur * real)}`;
  const head = spark.querySelector('.head');
  if (head) { head.setAttribute('x1', p * 1000); head.setAttribute('x2', p * 1000); }
}

function paintBox(v, b, full) {
  const vw = state.views[v];
  const show = state.showBox && !full;
  const vx = v * SRC_W;
  vw.dim.setAttribute('d', show ? `M${vx} 0h${SRC_W}v${SRC_H}h-${SRC_W}Z M${b.x} ${b.y}h${b.s}v${b.s}h-${b.s}Z` : '');
  vw.outline.setAttribute('x', b.x); vw.outline.setAttribute('y', b.y);
  vw.outline.setAttribute('width', b.s); vw.outline.setAttribute('height', b.s);
  vw.outline.style.display = show ? '' : 'none';
  const L = Math.max(14, b.s * 0.26), x = b.x, y = b.y, s = b.s;
  vw.brackets.setAttribute('d', show ? `M${x} ${y + L}V${y}H${x + L} M${x + s - L} ${y}H${x + s}V${y + L} M${x} ${y + s - L}V${y + s}H${x + L} M${x + s - L} ${y + s}H${x + s}V${y + s - L}` : '');
  const pairs = show && vw.rect ? hullPairs(b, vw.rect) : [];
  vw.lines.forEach((ln, i) => {
    const pr = pairs[i];
    ln.style.display = pr ? '' : 'none';
    if (pr) { ln.setAttribute('x1', pr[0][0]); ln.setAttribute('y1', pr[0][1]); ln.setAttribute('x2', pr[1][0]); ln.setAttribute('y2', pr[1][1]); }
  });
}

function buildSpark() {
  const ep = state.ep;
  spark.setAttribute('viewBox', '0 0 1000 38');
  spark.innerHTML = '';
  const maxZ = Math.log2(24);
  const yOf = (z) => 34 - (Math.log2(Math.max(1, z)) / maxZ) * 30;
  ep.boxes.forEach((rows, v) => {
    let d = '';
    rows.forEach((r, q) => {
      const z = q === 0 ? 1 : SRC_W / r[2];
      const x0 = (q * STEPS) / ep.frames * 1000, x1 = Math.min(1000, ((q + 1) * STEPS) / ep.frames * 1000);
      d += `${q ? 'L' : 'M'}${x0.toFixed(1)} ${yOf(z).toFixed(1)}H${x1.toFixed(1)}`;
    });
    const area = d + 'V34H0Z';
    if (v === 0) svgEl('path', { d: area, fill: 'rgba(0,185,154,0.12)' }, spark);
    svgEl('path', { d, fill: 'none', stroke: v === 0 ? '#00b99a' : '#7fd9c7', 'stroke-width': '1.6', 'stroke-dasharray': v ? '4 3' : '', 'vector-effect': 'non-scaling-stroke' }, spark);
  });
  svgEl('line', { class: 'head', x1: 0, x2: 0, y1: 0, y2: 38, stroke: '#8c1515', 'stroke-width': '1.5', 'vector-effect': 'non-scaling-stroke' }, spark);
  const lab = svgEl('text', { x: 4, y: 9, fill: '#6d6c69', 'font-size': '9', 'font-family': 'DM Sans, Arial', style: 'text-transform:uppercase;letter-spacing:.06em' }, spark);
  lab.textContent = 'detail ×';
  $('#ro-queries').textContent = ep.boxes[0].length;
}

let scrubbing = false;
function loadEpisode(id) {
  state.ep = EP.find((e) => e.id === id);
  const ep = state.ep;
  state.shown = []; state.tween = []; state.lastQ = -1; state.atlases = [];
  $$('.thumb').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.ep === id)));
  video.pause();
  video.poster = `assets/${ep.id}-poster.webp`;
  video.defaultPlaybackRate = state.speed; // a new src resets playbackRate to this
  lazySrc(video, `assets/${ep.id}.mp4`);
  video.loop = true;
  video.playbackRate = state.speed;
  $('#stage-tag').textContent = `Trial ${ep.trial}  ·  “${ep.instruction}”  ·  ${ep.model}`;
  buildStage();
  buildSpark();
  loadAtlases();
  draw(true);
  video.addEventListener('loadedmetadata', () => { video.playbackRate = state.speed; measureInsets(); draw(true); if (viewerVisible && !state.userPaused && !reduced) video.play().catch(() => {}); }, { once: true });
}

EP.forEach((ep) => {
  const b = document.createElement('button');
  b.className = 'thumb'; b.dataset.ep = ep.id; b.type = 'button';
  b.innerHTML = `<span class="im"><img src="assets/${ep.id}-poster.webp" alt="" loading="lazy"${ep.views > 1 ? ' style="object-position:left"' : ''}><span class="badge">${ep.model.replace('FoveaVLA-', '')}</span></span><span class="t">${ep.label}</span><span class="m">${ep.instruction}</span>`;
  b.addEventListener('click', () => { state.userPaused = false; loadEpisode(ep.id); });
  $('#thumbs').appendChild(b);
});

playBtn.addEventListener('click', () => {
  if (video.paused) { state.userPaused = false; video.play().catch(() => {}); }
  else { state.userPaused = true; video.pause(); }
});
video.addEventListener('play', () => { playBtn.classList.add('playing'); playBtn.setAttribute('aria-label', 'Pause'); });
video.addEventListener('pause', () => { playBtn.classList.remove('playing'); playBtn.setAttribute('aria-label', 'Play'); });
video.addEventListener('seeked', () => draw());
video.addEventListener('ratechange', () => { if (video.playbackRate !== state.speed) video.playbackRate = state.speed; });
timeInput.addEventListener('input', () => {
  scrubbing = true;
  const dur = video.duration || state.ep.frames / FPS;
  video.currentTime = (timeInput.value / 1000) * dur;
  draw();
});
timeInput.addEventListener('change', () => { scrubbing = false; });
timeInput.addEventListener('pointerup', () => { scrubbing = false; });
$$('[data-mode]').forEach((b) => b.addEventListener('click', () => {
  state.mode = b.dataset.mode;
  $$('[data-mode]').forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
  loadAtlases();
}));
$('#box-toggle').addEventListener('click', (e) => {
  state.showBox = !state.showBox;
  e.currentTarget.setAttribute('aria-pressed', String(state.showBox));
  state.lastQ = -1; draw(true);
});
stage.addEventListener('click', () => playBtn.click());

let viewerVisible = false;
whenVisible($('#viewer'), () => { viewerVisible = true; if (!state.userPaused && !reduced) video.play().catch(() => {}); }, () => { viewerVisible = false; video.pause(); }, 0.3);
loadEpisode(EP[0].id);
(function loop() { if (state.views.length) draw(); requestAnimationFrame(loop); })();

/* ---------------- versus ---------------- */
const SUITE_NAME = { pick: 'Pick-and-Place', clutter: 'Distractors', fold: 'Fold/Unfold', move: 'Move' };
// Two independent clips per trial (π0.5 | FoveaVLA-Single), one recorded control step per frame at 15 fps,
// so file time is robot time; each has its own play/pause and scrub bar.
const VS = window.VERSUS;
const vsVids = [$('#vs-base'), $('#vs-ours')];
let vsVisible = false;
const vsPlay = (v) => { if (vsVisible && !reduced && !v.userPaused) v.play().catch(() => {}); };
function loadVs(trial) {
  const it = VS.find((v) => v.trial === trial);
  $$('.vs-tab').forEach((t) => t.setAttribute('aria-selected', String(Number(t.dataset.trial) === trial)));
  [[it.base, it.basePoster], [it.ours, it.oursPoster]].forEach(([src, poster], i) => {
    const v = vsVids[i];
    v.poster = poster;
    v.defaultPlaybackRate = REAL_SPEED;
    lazySrc(v, src);
    v.playbackRate = REAL_SPEED;
    v.loop = true;
    v.userPaused = false;
    vsPlay(v);
  });
  $('#vs-instruction').innerHTML = `<em>Trial ${it.trial}</em>“${it.instruction}”` + (it.note ? `<span class="vs-note">${it.note}</span>` : '');
}
VS.forEach((it) => {
  const t = document.createElement('button');
  t.className = 'vs-tab'; t.dataset.trial = it.trial; t.setAttribute('role', 'tab');
  t.innerHTML = `${SUITE_NAME[it.suite]}<span>Trial ${it.trial}</span>`;
  t.addEventListener('click', () => loadVs(it.trial));
  $('#vs-tabs').appendChild(t);
});
[['#vs-bar-base', 0], ['#vs-bar-ours', 1]].forEach(([bar, i]) => {
  const v = vsVids[i];
  videoBar(v, $(bar));
  v.addEventListener('ratechange', () => { if (v.playbackRate !== REAL_SPEED) v.playbackRate = REAL_SPEED; });
  v.addEventListener('click', () => $(`${bar} .vbar-btn`).click());
});
whenVisible($('#vs-pair'), () => { vsVisible = true; vsVids.forEach(vsPlay); }, () => { vsVisible = false; vsVids.forEach((v) => v.pause()); }, 0.4);
loadVs(VS[0].trial);

/* ---------------- why crop ---------------- */
const DROID = window.DROID;
const SAMPLE_IDS = [1, 2, 3, 4, 6, 11];
const cmpF = $('#cmp-fovea').getContext('2d');
const cmpG = $('#cmp-global').getContext('2d');
const gridCv = $('#cmp-grid');
const gridCtx = gridCv.getContext('2d');
let curSample = null, split = 50;
function drawGrid() {
  gridCtx.clearRect(0, 0, 448, 448);
  if (!$('#grid-toggle').checked || !curSample) return;
  const L = curSample.box[2] - curSample.box[0];
  const gPx = L * 224 / 1280;            // region size in the 224 global input
  const stepG = 448 / gPx * 14;          // 14-px patches, displayed
  const off = (v) => ((v * 224 / 1280) % 14) / gPx * 448;
  const xs = split / 100 * 448;
  gridCtx.lineWidth = 1;
  gridCtx.save(); gridCtx.beginPath(); gridCtx.rect(0, 0, xs, 448); gridCtx.clip();
  gridCtx.strokeStyle = 'rgba(255,255,255,0.75)';
  const ox = -off(curSample.box[0]), oy = -off(curSample.box[1] + 280);
  for (let x = ox; x < 448; x += stepG) { gridCtx.beginPath(); gridCtx.moveTo(x, 0); gridCtx.lineTo(x, 448); gridCtx.stroke(); }
  for (let y = oy; y < 448; y += stepG) { gridCtx.beginPath(); gridCtx.moveTo(0, y); gridCtx.lineTo(448, y); gridCtx.stroke(); }
  gridCtx.restore();
  gridCtx.save(); gridCtx.beginPath(); gridCtx.rect(xs, 0, 448 - xs, 448); gridCtx.clip();
  gridCtx.strokeStyle = 'rgba(255,255,255,0.4)';
  for (let k = 0; k <= 16; k++) { const p = k * 28; gridCtx.beginPath(); gridCtx.moveTo(p, 0); gridCtx.lineTo(p, 448); gridCtx.stroke(); gridCtx.beginPath(); gridCtx.moveTo(0, p); gridCtx.lineTo(448, p); gridCtx.stroke(); }
  gridCtx.restore();
}
async function setSample(id) {
  const s = DROID.find((d) => d.id === id);
  curSample = s;
  $$('#sample-tabs button').forEach((b) => b.setAttribute('aria-pressed', String(Number(b.dataset.id) === id)));
  const im = await loadImage(s.src).catch(() => null);
  if (!im || curSample !== s) return;
  const [l, t, r, b] = s.box, L = r - l;
  $('#why-img').src = s.src;
  $('#why-img').alt = `DROID external image: ${s.title}`;
  $('#why-task').textContent = s.title;
  Object.assign($('#why-box').style, { left: `${l / 12.8}%`, top: `${t / 7.2}%`, width: `${L / 12.8}%`, height: `${(b - t) / 7.2}%` });
  // foveal: crop at source resolution -> 224 -> display
  const f = document.createElement('canvas'); f.width = f.height = 224;
  f.getContext('2d').drawImage(im, l, t, L, L, 0, 0, 224, 224);
  cmpF.imageSmoothingEnabled = false; cmpF.drawImage(f, 0, 0, 448, 448);
  // global: whole frame -> 224x126 letterboxed into 224x224 -> same region -> display
  const g = document.createElement('canvas'); g.width = g.height = 224;
  const gc = g.getContext('2d'); gc.fillStyle = '#000'; gc.fillRect(0, 0, 224, 224); gc.drawImage(im, 0, 49, 224, 126);
  const k = 224 / 1280;
  cmpG.imageSmoothingEnabled = false;
  cmpG.fillStyle = '#000'; cmpG.fillRect(0, 0, 448, 448);
  cmpG.drawImage(g, l * k, 49 + t * k, L * k, L * k, 0, 0, 448, 448);
  const tokens = Math.pow(L * k / 14, 2);
  $('#tok-global').textContent = tokens < 10 ? tokens.toFixed(1) : Math.round(tokens);
  drawGrid();
}
SAMPLE_IDS.forEach((id) => {
  const s = DROID.find((d) => d.id === id);
  const b = document.createElement('button');
  b.type = 'button'; b.dataset.id = id; b.textContent = s.title;
  b.addEventListener('click', () => setSample(id));
  $('#sample-tabs').appendChild(b);
});
const slider = $('#cmp-slider');
const setSplit = (v) => {
  split = Math.max(0, Math.min(100, v));
  slider.value = split;
  $('#cmp-low').style.clipPath = `inset(0 ${100 - split}% 0 0)`;
  $('#cmp-line').style.left = `${split}%`;
  drawGrid();
};
slider.addEventListener('input', () => setSplit(Number(slider.value)));
const cmp = $('#compare');
const dragTo = (e) => { const r = cmp.getBoundingClientRect(); setSplit((e.clientX - r.left) / r.width * 100); };
cmp.addEventListener('pointerdown', (e) => { cmp.setPointerCapture(e.pointerId); dragTo(e); });
cmp.addEventListener('pointermove', (e) => { if (cmp.hasPointerCapture(e.pointerId)) dragTo(e); });
$('#grid-toggle').addEventListener('change', drawGrid);
setSample(1);

/* ---------------- method stages ---------------- */
const MX_STAGES = [
  { on: null, t: 'FoveaVLA learns where to look from DROID demonstrations with RoboInter interaction annotations. From one shared observation, it predicts how to act and where to look next. Pick a stage, or let the tour step through one policy query.' },
  { on: ['inputs', 'siglip', 'prompt'], t: '<b>Observe.</b> Full and foveal RGB images and aligned depth maps go through a shared SigLIP encoder; the task prompt and discretized robot state complete the shared input prefix.' },
  { on: ['expert', 'noisy', 'arrExp', 'cur'], t: '<b>Act.</b> The flow-matching action expert turns noisy actions and flow time τ into the action chunk A<sub>t</sub> (16 actions); the robot executes the first w = 8.' },
  { on: ['vlm', 'arrBtw', 'Btw', 'fov'], t: '<b>Look ahead.</b> The language-model head decodes the foveation action B<sub>t+w</sub>: the interaction box at the next query, as four location tokens per view. It is decoded from the cached prefix while the robot moves, so it adds no wait.' },
  { on: ['Bt', 'crop', 'inputs'], t: '<b>Crop next.</b> The box predicted at the previous query, B<sub>t</sub>, crops the newly captured images at source resolution into the foveal crops. Then the loop repeats.' },
];
const MX_DWELL = 5200;
const mxSvg = $('#mx-svg');
const mxTabs = $$('#mx [role=tab]');
const mxPlay = $('#mx-play');
const mxBar = $('#mx-bar');
let mxStage = 0, mxTimer = 0, mxVisible = false, mxUserPaused = reduced;
const mxSet = (i) => {
  mxStage = i;
  const on = MX_STAGES[i].on;
  mxSvg.classList.toggle('staged', !!on);
  $$('[data-g]', mxSvg).forEach((g) => g.classList.toggle('on', !on || on.includes(g.dataset.g)));
  mxTabs.forEach((b, k) => { b.setAttribute('aria-selected', String(k === i)); b.tabIndex = k === i ? 0 : -1; });
  $('#mx-text').innerHTML = MX_STAGES[i].t;
  const cv = mxSvg.parentElement;
  if (cv.scrollWidth > cv.clientWidth + 1) {
    // narrow screens: scroll the highlighted part of the diagram into view
    const vb = mxSvg.viewBox.baseVal, k = mxSvg.clientWidth / vb.width;
    let x0 = Infinity, x1 = -Infinity;
    $$('[data-g].on', mxSvg).forEach((g) => { const b = g.getBBox(); x0 = Math.min(x0, b.x); x1 = Math.max(x1, b.x + b.width); });
    const mid = on ? ((x0 + x1) / 2 - vb.x) * k : 0;
    cv.scrollTo({ left: mid - cv.clientWidth / 2, behavior: reduced ? 'auto' : 'smooth' });
  }
};
const mxStop = () => { clearTimeout(mxTimer); mxBar.style.transition = 'none'; mxBar.style.width = '0'; };
const mxRun = () => {
  mxStop();
  if (mxUserPaused || !mxVisible) return;
  void mxBar.offsetWidth;
  mxBar.style.transition = `width ${MX_DWELL}ms linear`; mxBar.style.width = '100%';
  mxTimer = setTimeout(() => { mxSet((mxStage + 1) % MX_STAGES.length); mxRun(); }, MX_DWELL);
};
const mxSyncPlay = () => { mxPlay.classList.toggle('paused', mxUserPaused); mxPlay.setAttribute('aria-label', mxUserPaused ? 'Play stage tour' : 'Pause stage tour'); };
mxTabs.forEach((b, k) => b.addEventListener('click', () => { mxUserPaused = true; mxSyncPlay(); mxSet(k); mxStop(); }));
$('#mx .mx-steps').addEventListener('keydown', (e) => {
  const d = e.key === 'ArrowRight' ? 1 : e.key === 'ArrowLeft' ? -1 : 0;
  if (!d || !e.target.matches('[role=tab]')) return;
  e.preventDefault();
  const k = (mxStage + d + MX_STAGES.length) % MX_STAGES.length;
  mxTabs[k].click(); mxTabs[k].focus();
});
mxPlay.addEventListener('click', () => { mxUserPaused = !mxUserPaused; mxSyncPlay(); if (!mxUserPaused) mxSet((mxStage + 1) % MX_STAGES.length); mxRun(); });
whenVisible($('#mx'), () => { mxVisible = true; mxRun(); }, () => { mxVisible = false; mxStop(); }, 0.4);
mxSet(0); mxSyncPlay();

/* ---------------- results charts ---------------- */
const SUITES = ['Pick-and-Place<br>(plain)', 'Pick-and-Place<br>(w/ distractors)', 'Fold/Unfold', 'Move<br>(far objects)', 'Overall'];
const SERIES = [
  { cls: 'pi', name: 'π0.5', v: [100, 25.0, 41.7, 41.7, 52.1], e: [0, 13.1, 14.9, 14.9, 7.3] },
  { cls: 'single', name: 'FoveaVLA-Single', v: [100, 58.3, 50.0, 75.0, 70.8], e: [0, 14.9, 15.1, 13.1, 6.6] },
  { cls: 'dual', name: 'FoveaVLA-Dual', v: [100, 50.0, 75.0, 83.3, 77.1], e: [0, 15.1, 13.1, 11.2, 6.1] },
];
const bars = $('#bars');
const axis = document.createElement('div'); axis.className = 'axis';
[0, 25, 50, 75, 100].forEach((p) => { const i = document.createElement('i'); i.style.bottom = `${p}%`; i.innerHTML = `<span>${p}</span>`; axis.appendChild(i); });
bars.appendChild(axis);
SUITES.forEach((label, gi) => {
  const g = document.createElement('div'); g.className = 'group' + (gi === 4 ? ' overall' : '');
  const cols = document.createElement('div'); cols.className = 'cols';
  SERIES.forEach((s) => {
    const c = document.createElement('div'); c.className = `col ${s.cls}`;
    c.dataset.h = s.v[gi];
    const rel = s.e[gi] / s.v[gi] * 100;
    c.innerHTML = (s.e[gi] ? `<span class="err" style="bottom:${100 - rel}%;height:${2 * rel}%"></span>` : '') +
      `<span class="v" style="bottom:calc(100% + ${s.e[gi] ? rel : 0}% + 4px)">${s.v[gi] % 1 ? s.v[gi].toFixed(1) : s.v[gi]}</span>`;
    c.title = `${s.name}: ${s.v[gi]}% ± ${s.e[gi]}`;
    cols.appendChild(c);
  });
  const lab = document.createElement('div'); lab.className = 'lab'; lab.innerHTML = label;
  g.append(cols, lab); bars.appendChild(g);
});
whenVisible(bars, () => {
  bars.classList.add('in');
  $$('.col', bars).forEach((c) => { c.style.height = `${c.dataset.h}%`; });
}, null, 0.3);

/* ---------------- ablations ---------------- */
const WRIST = 52.1;
const AB1 = [
  // name, what it is, [full view, foveal crop, depth], success, style
  ['π0.5-Blackout', 'Trained with the wrist camera, which is blacked out at test time', [1, 0, 0], 0.0, 'base'],
  ['π0.5-Single-External', 'Trained on the external view alone', [1, 0, 0], 6.3, 'base'],
  ['π0.5-Single-External-RGB-D', 'Adds depth', [1, 0, 1], 6.3, 'base'],
  ['FoveaVLA-Single-RGB', 'Adds the foveal crop, no depth', [1, 1, 0], 20.8, 'rgb'],
  ['FoveaVLA-Single', 'Foveal crop and depth together', [1, 1, 1], 70.8, 'ours'],
];
const pi = (m) => m.replace('π0.5', 'π<sub>0.5</sub>');
const ab1 = $('#ab1');
ab1.innerHTML = '<div class="ab1-h">Policy</div><div class="ab1-h c">Full view</div><div class="ab1-h c">Foveal crop</div><div class="ab1-h c">Depth</div>' +
  `<div class="ab1-h ab1-axis" aria-hidden="true">${[0, 25, 50, 75, 100].map((t) => `<span style="left:${t}%">${t}%</span>`).join('')}</div>` +
  AB1.map(([name, desc, ins, v, cls]) =>
    `<div class="ab1-nm ${cls}"><b>${pi(name)}</b><span>${desc}</span></div>` +
    ins.map((on) => `<div class="ab1-in${on ? ' on' : ''}" aria-hidden="true"><i></i></div>`).join('') +
    `<div class="ab1-bar ${cls}" title="${name.replace('π0.5', 'π₀.₅')}: ${v.toFixed(1)}%">` +
    [25, 50, 75].map((t) => `<u style="left:${t}%"></u>`).join('') +
    `<s style="left:${WRIST}%"></s><i data-w="${v}"></i><b class="v" style="left:${v}%">${v.toFixed(1)}%</b></div>`).join('') +
  `<div class="ab1-ref" aria-hidden="true"><em style="left:${WRIST}%">π<sub>0.5</sub> with wrist camera · ${WRIST}%</em></div>`;
whenVisible(ab1, () => { ab1.classList.add('in'); $$('.ab1-bar i', ab1).forEach((i) => { i.style.width = `${i.dataset.w}%`; }); }, null, 0.3);

/* ---------------- citation ---------------- */
$('#copy-bib').addEventListener('click', async (e) => {
  const btn = e.currentTarget;
  try { await navigator.clipboard.writeText($('#bibtex').textContent); btn.textContent = 'Copied'; }
  catch { const r = document.createRange(); r.selectNodeContents($('#bibtex')); const s = getSelection(); s.removeAllRanges(); s.addRange(r); btn.textContent = 'Selected'; }
  setTimeout(() => { btn.textContent = 'Copy BibTeX'; }, 1800);
});

/* ---------------- reveal ---------------- */
if (!reduced && 'IntersectionObserver' in window) {
  const items = $$('.block > .heading, .block > .lead, .trio, .viewer, .vs, .why, .mx, .facts, .chart-card, .ab1, .paper-fig');
  const ro = new IntersectionObserver((entries) => entries.forEach((e) => { if (e.isIntersecting) { e.target.classList.add('in'); ro.unobserve(e.target); } }), { threshold: 0.08 });
  items.forEach((el) => { el.classList.add('reveal'); ro.observe(el); });
}
document.addEventListener('visibilitychange', () => { if (document.hidden) { teaser.pause(); video.pause(); vsVids.forEach((v) => v.pause()); } });
})();
