/* Registration Playground: in-browser deformable registration with ONNX Runtime Web.
 * Models: 2D HyperVFA-SPR (spatially varying regularization) and HyperVFA (spatially invariant),
 * trained with MIR (https://github.com/junyuchen245/MIR). See _pages/playground.html.
 */
(function () {
  'use strict';
  var root = document.getElementById('playground');
  if (!root) return;
  var ASSETS = root.getAttribute('data-assets');
  var ORT_CDN = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/';
  var MODELS = {
    spr:   { file: 'models/hypervfa_spr_2d_fp16w.onnx',   meta: 'models/hypervfa_spr_2d.json' },
    plain: { file: 'models/hypervfa_plain_2d_fp16w.onnx', meta: 'models/hypervfa_plain_2d.json' }
  };
  var GRID_SPACING = 10;

  var $ = function (id) { return document.getElementById(id); };
  // Debug log: with ?debug in the URL, messages are also written into the page (useful for headless checks).
  var DEBUG = /[?&]debug/.test(location.search), debugEl = null;
  function log() {
    var msg = Array.prototype.map.call(arguments, function (a) { return (a && a.message) ? a.message : (typeof a === 'object' ? JSON.stringify(a) : String(a)); }).join(' ');
    msg = (performance.now() / 1000).toFixed(1) + 's ' + msg;
    console.log('[playground] ' + msg);
    if (DEBUG) {
      if (!debugEl) { debugEl = document.createElement('pre'); debugEl.id = 'pg-debug'; root.appendChild(debugEl); }
      debugEl.textContent += msg + '\n';
    }
  }
  window.addEventListener('error', function (e) { log('window.error:', e.message, e.filename, e.lineno); });
  window.addEventListener('unhandledrejection', function (e) { log('unhandledrejection:', e.reason && e.reason.message ? e.reason.message : e.reason); });
  var ui = {
    status: $('pg-status'), statusText: $('pg-status-text'), progress: $('pg-progress'),
    alpha: $('pg-alpha'), alphaRow: $('pg-alpha-row'), alphaVal: $('pg-alpha-val'),
    reg: $('pg-reg'), regVal: $('pg-reg-val'), grid: $('pg-grid'), before: $('pg-before'),
    thumbsMov: $('pg-thumbs-mov'), thumbsFix: $('pg-thumbs-fix'), swap: $('pg-swap'),
    cvMov: $('cv-mov'), cvFix: $('cv-fix'), cvRes: $('cv-res'), cvLam: $('cv-lam'), cvJac: $('cv-jac'),
    figLam: $('fig-lam'), capRes: $('cap-res'), modelHint: $('pg-model-hint'),
    stDice: $('st-dice'), stFold: $('st-fold'), stLam: $('st-lam'), stLamWrap: $('st-lam-wrap'), stTime: $('st-time')
  };

  var state = {
    H: 192, W: 160, samples: [], imgs: [], segs: [],
    mov: 0, fix: 1, kind: 'spr', view: 'warped', showGrid: true, before: false,
    sessions: {}, meta: {}, loading: {}, busy: false, pending: false, result: null, backend: 'wasm'
  };

  /* ------------------------------------------------------------------ utils */
  function setStatus(text, frac, done) {
    ui.statusText.textContent = text;
    if (typeof frac === 'number') ui.progress.style.width = Math.round(frac * 100) + '%';
    ui.status.classList.toggle('is-done', !!done);
    ui.status.classList.toggle('is-error', false);
  }
  function setError(text) {
    ui.statusText.textContent = text; ui.status.classList.add('is-error'); ui.status.classList.remove('is-done');
  }
  function fetchWithProgress(url, onProgress) {
    return fetch(url).then(function (res) {
      if (!res.ok) throw new Error('HTTP ' + res.status + ' for ' + url);
      var total = +res.headers.get('Content-Length') || 0;
      if (!res.body || !total) return res.arrayBuffer().then(function (b) { return new Uint8Array(b); });
      var reader = res.body.getReader(), chunks = [], got = 0;
      return (function pump() {
        return reader.read().then(function (r) {
          if (r.done) {
            var out = new Uint8Array(got), off = 0;
            chunks.forEach(function (c) { out.set(c, off); off += c.length; });
            return out;
          }
          chunks.push(r.value); got += r.value.length; onProgress(got / total);
          return pump();
        });
      })();
    });
  }
  function loadImageData(url) {
    return new Promise(function (resolve, reject) {
      var im = new Image();
      im.crossOrigin = 'anonymous';   // allows pixel reads when assets are served from another origin (CORS)
      im.onload = function () {
        var c = document.createElement('canvas'); c.width = im.width; c.height = im.height;
        var ctx = c.getContext('2d'); ctx.drawImage(im, 0, 0);
        resolve(ctx.getImageData(0, 0, im.width, im.height));
      };
      im.onerror = function () { reject(new Error('Could not load ' + url)); };
      im.src = url;
    });
  }
  function toGray(imageData) {
    var n = imageData.width * imageData.height, out = new Float32Array(n), d = imageData.data;
    for (var i = 0; i < n; i++) out[i] = d[i * 4] / 255;
    return out;
  }
  function toLabels(imageData) {
    var n = imageData.width * imageData.height, out = new Uint8Array(n), d = imageData.data;
    for (var i = 0; i < n; i++) out[i] = d[i * 4];
    return out;
  }

  /* ------------------------------------------------------------- colormaps */
  var VIRIDIS = [[68,1,84],[71,24,106],[72,40,120],[69,55,129],[64,70,136],[58,84,140],[52,97,141],[46,109,142],[41,121,142],[36,133,142],[32,144,141],[30,156,137],[34,168,132],[47,180,124],[68,191,112],[94,201,98],[122,209,81],[155,217,60],[189,223,38],[223,227,24],[253,231,37]];
  function lut(table, t) {
    t = Math.max(0, Math.min(1, t)) * (table.length - 1);
    var i = Math.floor(t), f = t - i, a = table[i], b = table[Math.min(i + 1, table.length - 1)];
    return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
  }
  function diverging(j) {              // |J| in log2 scale: 0.5 -> blue, 1 -> white, 2 -> red; <=0 -> dark magenta
    if (j <= 0) return [120, 0, 120];
    var t = Math.max(-1, Math.min(1, Math.log(j) / Math.LN2));
    if (t < 0) { t = -t; return [255 - t * (255 - 33), 255 - t * (255 - 102), 255 - t * (255 - 172)]; }
    return [255 - t * (255 - 178), 255 - t * (255 - 24), 255 - t * (255 - 43)];
  }
  function hot(t) { t = Math.max(0, Math.min(1, t)); return [Math.min(255, t * 3 * 255), Math.min(255, Math.max(0, t * 3 - 1) * 255), Math.min(255, Math.max(0, t * 3 - 2) * 255)]; }

  /* ---------------------------------------------------------------- drawing */
  function paint(canvas, rgb) {           // rgb: Float32Array/Array of length 3*n in 0..255
    var ctx = canvas.getContext('2d'), img = ctx.createImageData(canvas.width, canvas.height), d = img.data, n = canvas.width * canvas.height;
    for (var i = 0; i < n; i++) { d[i * 4] = rgb[i * 3]; d[i * 4 + 1] = rgb[i * 3 + 1]; d[i * 4 + 2] = rgb[i * 3 + 2]; d[i * 4 + 3] = 255; }
    ctx.putImageData(img, 0, 0);
  }
  function grayRGB(g) { var n = g.length, o = new Float32Array(n * 3); for (var i = 0; i < n; i++) { var v = Math.max(0, Math.min(1, g[i])) * 255; o[i * 3] = o[i * 3 + 1] = o[i * 3 + 2] = v; } return o; }
  function drawGray(canvas, g) { paint(canvas, grayRGB(g)); }

  /* -------------------------------------------------------------- warping */
  // flow: Float32Array [2,H,W] voxel displacements, channel 0 along rows (y), channel 1 along columns (x);
  // warped(y,x) = src(y + fy, x + fx), bilinear, zero padding (matches MIR SpatialTransformer).
  function warpBilinear(src, flow, H, W) {
    var out = new Float32Array(H * W), n = H * W;
    for (var y = 0; y < H; y++) for (var x = 0; x < W; x++) {
      var i = y * W + x, sy = y + flow[i], sx = x + flow[n + i];
      var y0 = Math.floor(sy), x0 = Math.floor(sx), fy = sy - y0, fx = sx - x0, v = 0;
      for (var dy = 0; dy < 2; dy++) for (var dx = 0; dx < 2; dx++) {
        var yy = y0 + dy, xx = x0 + dx;
        if (yy < 0 || yy >= H || xx < 0 || xx >= W) continue;
        v += src[yy * W + xx] * (dy ? fy : 1 - fy) * (dx ? fx : 1 - fx);
      }
      out[i] = v;
    }
    return out;
  }
  function warpNearest(src, flow, H, W) {
    var out = new Uint8Array(H * W), n = H * W;
    for (var y = 0; y < H; y++) for (var x = 0; x < W; x++) {
      var i = y * W + x, yy = Math.round(y + flow[i]), xx = Math.round(x + flow[n + i]);
      out[i] = (yy < 0 || yy >= H || xx < 0 || xx >= W) ? 0 : src[yy * W + xx];
    }
    return out;
  }
  function jacobian(flow, H, W) {        // forward differences, |J| on (H-1)x(W-1), padded to HxW; returns {det, folds}
    var det = new Float32Array(H * W), n = H * W, folds = 0, count = 0;
    for (var y = 0; y < H - 1; y++) for (var x = 0; x < W - 1; x++) {
      var i = y * W + x;
      var j11 = flow[i + W] - flow[i] + 1, j12 = flow[i + 1] - flow[i];
      var j21 = flow[n + i + W] - flow[n + i], j22 = flow[n + i + 1] - flow[n + i] + 1;
      var d = j11 * j22 - j12 * j21; det[i] = d; count++; if (d <= 0) folds++;
    }
    for (var yy = 0; yy < H; yy++) det[yy * W + W - 1] = det[yy * W + W - 2];
    for (var xx = 0; xx < W; xx++) det[(H - 1) * W + xx] = det[(H - 2) * W + xx];
    return { det: det, foldPct: 100 * folds / count };
  }
  function dice(a, b) {
    var sumA = {}, sumB = {}, inter = {}, l;
    for (var i = 0; i < a.length; i++) {
      if (b[i]) { sumB[b[i]] = (sumB[b[i]] || 0) + 1; }
      if (a[i]) { sumA[a[i]] = (sumA[a[i]] || 0) + 1; if (a[i] === b[i]) inter[a[i]] = (inter[a[i]] || 0) + 1; }
    }
    var vals = [];
    for (l in sumB) { var den = (sumA[l] || 0) + sumB[l]; if (den > 0) vals.push(2 * (inter[l] || 0) / den); }
    return vals.length ? vals.reduce(function (s, v) { return s + v; }, 0) / vals.length : NaN;
  }
  function makeGrid(H, W, spacing) {
    var g = new Float32Array(H * W);
    for (var y = 0; y < H; y++) for (var x = 0; x < W; x++) if (y % spacing === 0 || x % spacing === 0) g[y * W + x] = 1;
    return g;
  }

  /* ------------------------------------------------------------- rendering */
  function composite(a, b) {              // result view of image a against reference b
    var H = state.H, W = state.W, n = H * W, out = new Float32Array(n * 3), i, v;
    if (state.view === 'warped') return grayRGB(a);
    if (state.view === 'checker') {
      var T = 16;
      for (i = 0; i < n; i++) { var y = (i / W) | 0, x = i % W; v = (((y / T) | 0) + ((x / T) | 0)) % 2 ? a[i] : b[i]; v = Math.max(0, Math.min(1, v)) * 255; out[i * 3] = out[i * 3 + 1] = out[i * 3 + 2] = v; }
      return out;
    }
    if (state.view === 'diff') {
      for (i = 0; i < n; i++) { var c = hot(Math.abs(a[i] - b[i]) * 2); out[i * 3] = c[0]; out[i * 3 + 1] = c[1]; out[i * 3 + 2] = c[2]; }
      return out;
    }
    for (i = 0; i < n; i++) { out[i * 3] = Math.min(1, a[i]) * 255; out[i * 3 + 1] = Math.min(1, b[i]) * 255; out[i * 3 + 2] = Math.min(1, a[i]) * 255; }
    return out;
  }
  function render() {
    var H = state.H, W = state.W, n = H * W, r = state.result;
    drawGray(ui.cvMov, state.imgs[state.mov]); drawGray(ui.cvFix, state.imgs[state.fix]);
    if (!r) return;
    var showBefore = state.before;
    var rgb = composite(showBefore ? state.imgs[state.mov] : r.warped, state.imgs[state.fix]);
    if (state.showGrid && !showBefore) {
      for (var i = 0; i < n; i++) { var g = r.grid[i]; if (g > 0.15) { var a = Math.min(1, g); rgb[i * 3] = rgb[i * 3] * (1 - a) + 0 * a; rgb[i * 3 + 1] = rgb[i * 3 + 1] * (1 - a) + 200 * a; rgb[i * 3 + 2] = rgb[i * 3 + 2] * (1 - a) + 255 * a; } }
    }
    paint(ui.cvRes, rgb);
    var viewName = { warped: 'image', checker: 'checkerboard with fixed', diff: 'absolute difference to fixed', overlay: 'overlay (magenta) on fixed (green)' }[state.view];
    ui.capRes.textContent = (showBefore ? 'Moving, before registration' : 'Moving, warped to fixed') + ' — ' + viewName;
    if (r.lam) {
      var lamRGB = new Float32Array(n * 3);
      for (var k = 0; k < n; k++) { var c = lut(VIRIDIS, r.lam[k]); lamRGB[k * 3] = c[0]; lamRGB[k * 3 + 1] = c[1]; lamRGB[k * 3 + 2] = c[2]; }
      paint(ui.cvLam, lamRGB);
    }
    var jacRGB = new Float32Array(n * 3);
    for (var m = 0; m < n; m++) { var cj = diverging(r.jac.det[m]); jacRGB[m * 3] = cj[0]; jacRGB[m * 3 + 1] = cj[1]; jacRGB[m * 3 + 2] = cj[2]; }
    paint(ui.cvJac, jacRGB);
    ui.stDice.innerHTML = r.diceBefore.toFixed(3) + ' <span class="pg-arrow">&rarr;</span> <strong>' + r.diceAfter.toFixed(3) + '</strong>';
    ui.stFold.textContent = r.jac.foldPct.toFixed(2) + ' %';
    ui.stLam.textContent = r.lam ? r.lamMean.toFixed(3) : '–';
    ui.stTime.textContent = Math.round(r.ms) + ' ms (' + state.backend + ')';
  }

  /* ----------------------------------------------------------- inference */
  function logmap(c, lo, hi) { return lo * Math.pow(hi / lo, c); }
  function fmt(x) { return x >= 1 ? x.toFixed(2) : x >= 0.01 ? x.toFixed(3) : x.toExponential(1); }
  function hyperValues() {
    var meta = state.meta[state.kind] || {};
    var hm = meta.hyper_map || {};
    var rr = hm.reg_range || [0.0005, 0.2], kr = hm.kappa_range || [0.03, 3];
    var k = +ui.alpha.value, g = +ui.reg.value;
    var reg = logmap(g, rr[0], rr[1]);
    var kappa = logmap(k, kr[0], kr[1]);
    ui.alphaVal.textContent = '= ' + fmt(kappa) + '  (\u03b1 = 1 + \u03ba\u00b7w = ' + (1 + kappa * reg).toFixed(4) + ')';
    ui.regVal.textContent = '= ' + fmt(reg);
    return state.kind === 'spr' ? [k, g] : [g];
  }
  function getSession(kind) {
    if (state.sessions[kind]) return Promise.resolve(state.sessions[kind]);
    if (state.loading[kind]) return state.loading[kind];
    var m = MODELS[kind];
    setStatus('Downloading ' + (kind === 'spr' ? 'spatially varying' : 'spatially invariant') + ' model…', 0);
    state.loading[kind] = Promise.all([
      fetchWithProgress(ASSETS + '/' + m.file, function (f) { setStatus('Downloading model… ' + Math.round(f * 100) + '%', f * 0.9); }),
      fetch(ASSETS + '/' + m.meta).then(function (r) { return r.ok ? r.json() : {}; })
    ]).then(function (res) {
      state.meta[kind] = res[1] || {};
      setStatus('Initializing model…', 0.95); log('model bytes', res[0].length, 'creating session');
      return ort.InferenceSession.create(res[0], { executionProviders: [state.backend], graphOptimizationLevel: 'all' });
    }).then(function (sess) {
      state.sessions[kind] = sess; delete state.loading[kind]; log('session ready', kind, 'threads', ort.env.wasm.numThreads);
      var meta = state.meta[kind];
      setStatus('Ready' + (meta.params ? ' · ' + (meta.params / 1e6).toFixed(1) + 'M parameters' : '') + ' · runs locally via WebAssembly', 1, true);
      return sess;
    }).catch(function (e) { delete state.loading[kind]; setError('Could not load the model: ' + e.message); throw e; });
    return state.loading[kind];
  }
  function run() {
    if (state.busy) { state.pending = true; log('run queued'); return; }
    state.busy = true; root.classList.add('is-busy');
    var H = state.H, W = state.W, kind = state.kind, hv = hyperValues();
    getSession(kind).then(function (sess) {
      var feeds = {
        moving: new ort.Tensor('float32', state.imgs[state.mov], [1, 1, H, W]),
        fixed: new ort.Tensor('float32', state.imgs[state.fix], [1, 1, H, W]),
        hyper: new ort.Tensor('float32', new Float32Array(hv), [1, hv.length])
      };
      var t0 = performance.now(); log('run start', kind, JSON.stringify(hv));
      return sess.run(feeds).then(function (out) {
        var ms = performance.now() - t0; log('run done in', Math.round(ms), 'ms');
        var flow = out.flow.data, lam = out.weights ? out.weights.data : null;
        var warped = warpBilinear(state.imgs[state.mov], flow, H, W);
        var wseg = warpNearest(state.segs[state.mov], flow, H, W);
        var lamMean = 0; if (lam) { for (var i = 0; i < lam.length; i++) lamMean += lam[i]; lamMean /= lam.length; }
        state.result = {
          flow: flow, lam: lam, lamMean: lamMean, warped: warped, ms: ms,
          grid: warpBilinear(state.gridImg, flow, H, W), jac: jacobian(flow, H, W),
          diceBefore: dice(state.segs[state.mov], state.segs[state.fix]), diceAfter: dice(wseg, state.segs[state.fix])
        };
        render();
      });
    }).catch(function (e) { log('inference failed', e); setError('Inference failed: ' + e.message); })
      .then(function () {
        state.busy = false; root.classList.remove('is-busy');
        if (state.pending) { state.pending = false; run(); }
      });
  }

  /* ------------------------------------------------------------------- UI */
  function buildThumbs(container, which) {
    container.innerHTML = '';
    state.samples.forEach(function (s, idx) {
      var b = document.createElement('button'); b.type = 'button'; b.className = 'pg-thumb'; b.title = 'Subject ' + s.subject;
      b.setAttribute('aria-label', (which === 'mov' ? 'Moving' : 'Fixed') + ' image: subject ' + s.subject);
      var c = document.createElement('canvas'); c.width = state.W; c.height = state.H; drawGray(c, state.imgs[idx]);
      b.appendChild(c);
      b.addEventListener('click', function () { state[which] = idx; refreshThumbs(); state.result = null; render(); run(); });
      container.appendChild(b);
    });
  }
  function refreshThumbs() {
    [['thumbsMov', 'mov'], ['thumbsFix', 'fix']].forEach(function (p) {
      Array.prototype.forEach.call(ui[p[0]].children, function (el, i) { el.classList.toggle('is-active', i === state[p[1]]); });
    });
  }
  function setKind(kind) {
    state.kind = kind;
    Array.prototype.forEach.call(document.querySelectorAll('.pg-seg [data-model]'), function (b) { b.classList.toggle('is-active', b.getAttribute('data-model') === kind); });
    var spr = kind === 'spr';
    ui.alphaRow.style.display = spr ? '' : 'none'; ui.figLam.style.display = spr ? '' : 'none'; ui.stLamWrap.style.display = spr ? '' : 'none';
    ui.modelHint.textContent = spr ? 'Predicts a per-pixel regularization weight λ(x) together with the deformation.'
                                   : 'One global regularization weight for the whole image (the conventional setting).';
    state.result = null; render(); run();
  }
  function bind() {
    Array.prototype.forEach.call(document.querySelectorAll('.pg-seg [data-model]'), function (b) { b.addEventListener('click', function () { setKind(b.getAttribute('data-model')); }); });
    Array.prototype.forEach.call(document.querySelectorAll('#pg-view [data-view]'), function (b) {
      b.addEventListener('click', function () {
        state.view = b.getAttribute('data-view');
        Array.prototype.forEach.call(b.parentNode.children, function (x) { x.classList.toggle('is-active', x === b); });
        render();
      });
    });
    ui.alpha.addEventListener('input', function () { hyperValues(); run(); });
    ui.reg.addEventListener('input', function () { hyperValues(); run(); });
    ui.grid.addEventListener('change', function () { state.showGrid = ui.grid.checked; render(); });
    ui.before.addEventListener('change', function () { state.before = ui.before.checked; render(); });
    ui.swap.addEventListener('click', function () { var t = state.mov; state.mov = state.fix; state.fix = t; refreshThumbs(); state.result = null; render(); run(); });
  }

  /* ----------------------------------------------------------------- init */
  function init() {
    if (typeof ort === 'undefined') { setError('ONNX Runtime Web could not be loaded (network or script blocker).'); return; }
    ort.env.wasm.wasmPaths = ORT_CDN;
    ort.env.wasm.numThreads = (self.crossOriginIsolated && navigator.hardwareConcurrency > 1) ? Math.min(4, navigator.hardwareConcurrency) : 1;
    setStatus('Loading images…', 0.05);
    fetch(ASSETS + '/samples/manifest.json').then(function (r) { return r.json(); }).then(function (man) {
      state.H = man.size[0]; state.W = man.size[1]; state.samples = man.samples;
      state.gridImg = makeGrid(state.H, state.W, GRID_SPACING);
      return Promise.all(man.samples.map(function (s) {
        return Promise.all([loadImageData(ASSETS + '/samples/' + s.img), loadImageData(ASSETS + '/samples/' + s.seg)]);
      }));
    }).then(function (pairs) {
      state.imgs = pairs.map(function (p) { return toGray(p[0]); });
      state.segs = pairs.map(function (p) { return toLabels(p[1]); });
      buildThumbs(ui.thumbsMov, 'mov'); buildThumbs(ui.thumbsFix, 'fix'); refreshThumbs();
      bind(); hyperValues(); render();
      log('images loaded', state.imgs.length, state.H + 'x' + state.W);
      run();
    }).catch(function (e) { log('init failed', e); setError('Could not load demo images: ' + e.message); });
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
