(function () {
  "use strict";
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const CLASSES = window.MODEL_META.classes;
  const S = { net: null, ready: false, live: false, stream: null, busy: false, hist: [], locked: false, noLeaf: false, auto: true, src: null, log: [], countdown: null, lastMs: 0 };
  const LEAF_MIN = 0.5; // leaf-detector probability needed to unlock the nozzle
  const ICON = {
    spray: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 3h6v4H9zM7 7h10v4H7zM12 11v3M8 17l-2 4M12 16v5M16 17l2 4"/></svg>',
    leaf: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M5 19c0-9 6-14 15-14 0 9-5 15-14 15"/><path d="M5 19 13 11"/></svg>',
    soil: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 18h18M6 14c2-3 4-3 6 0s4 3 6 0M12 4v6"/></svg>',
    hand: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 12h3l3 6h6l3-8h-6l-2-6"/></svg>',
    book: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 5h7a3 3 0 0 1 3 3v12a3 3 0 0 0-3-3H4zM20 5h-6"/></svg>',
  };
  const DEC = {
    SPRAY: { badge: "SPRAY", text: (k) => `Spray this patch only: ${k.product.kind.toLowerCase()}. Healthy rows stay dry.` },
    VECTOR: { badge: "NO CURE · CONTROL VECTOR", text: () => "Remove infected plants. Spray only to control the insect that spreads it." },
    REMOVE: { badge: "DO NOT SPRAY · REMOVE", text: () => "No chemical works. Pull infected plants and sanitise tools and hands." },
    MONITOR: { badge: "DO NOT SPRAY · MONITOR", text: () => "A leaf spray will not pay here. Tag the patch and keep watching." },
    NONE: { badge: "HEALTHY · NO SPRAY", text: () => "Leaf looks healthy. Zero chemical for this patch." },
  };

  // ---------- storage ----------
  function loadLog() { try { S.log = JSON.parse(localStorage.getItem("nv-log") || "[]"); } catch (e) { S.log = []; } }
  function saveLog() { try { localStorage.setItem("nv-log", JSON.stringify(S.log.slice(-60))); } catch (e) {} }
  function toast(t) { const d = document.createElement("div"); d.className = "toast"; d.textContent = t; document.body.appendChild(d); setTimeout(() => d.remove(), 2200); }

  // ---------- routing ----------
  const VIEWS = ["scan", "log", "library", "how"];
  function route() {
    const v = VIEWS.includes(location.hash.slice(1)) ? location.hash.slice(1) : "scan";
    VIEWS.forEach((x) => ($("#view-" + x).hidden = x !== v));
    $$(".tabs a").forEach((a) => a.toggleAttribute("aria-current", a.dataset.v === v) || (a.dataset.v === v && a.setAttribute("aria-current", "page")));
    $$(".tabs a").forEach((a) => (a.dataset.v === v ? a.setAttribute("aria-current", "page") : a.removeAttribute("aria-current")));
    if (v === "log") renderLog(); if (v === "library") renderLibrary(); if (v === "how") renderHow();
  }
  window.addEventListener("hashchange", () => { if (window.FPV) { FPV.stop(); const f = $("#fpv"); f.hidden = true; f.innerHTML = ""; } route(); });

  // ---------- model ----------
  async function initModel() {
    const chip = $("#chipModel");
    try {
      let backend = "cpu";
      try { await tf.setBackend("webgl"); await tf.ready(); backend = tf.getBackend(); } catch (e) { await tf.setBackend("cpu"); await tf.ready(); }
      $("#chipBackend").className = "chip ok"; $("#chipBackend span").textContent = backend === "webgl" ? "GPU (WebGL)" : "CPU";
      S.net = await LeafNetRuntime.load("model/", (p) => { chip.querySelector("span").textContent = "Loading model " + Math.round(p * 100) + "%"; });
      const x = tf.zeros([1, 224, 224, 3]); const t0 = performance.now(); const y = S.net.run(x); await y.data(); x.dispose(); y.dispose();
      S.ready = true; chip.className = "chip ok"; chip.querySelector("span").textContent = "LeafNet ready · 38 classes";
      setMs(performance.now() - t0);
    } catch (e) {
      chip.className = "chip err"; chip.querySelector("span").textContent = "Model failed to load"; console.error(e);
      $("#camNote").textContent = "The model could not load: " + e.message; $("#camNote").className = "note err";
    }
  }
  function setMs(ms) { S.lastMs = ms; $("#chipMs").className = "chip ok"; $("#chipMs span").textContent = Math.round(ms) + " ms / frame"; }

  // ---------- pixel analysis ----------
  const inCan = document.createElement("canvas"); inCan.width = inCan.height = 224;
  const inCtx = inCan.getContext("2d", { willReadFrequently: true });
  const maskCan = document.createElement("canvas"); maskCan.width = maskCan.height = 224;
  function pixelStats() {
    const d = inCtx.getImageData(0, 0, 224, 224).data;
    const md = maskCan.getContext("2d").createImageData(224, 224);
    let plant = 0, les = 0, green = 0;
    for (let i = 0; i < d.length; i += 4) {
      const r = d[i] / 255, g = d[i + 1] / 255, b = d[i + 2] / 255;
      const mx = Math.max(r, g, b), mn = Math.min(r, g, b), v = mx, s = mx ? (mx - mn) / mx : 0;
      let h = 0; const c = mx - mn;
      if (c) { h = mx === r ? ((g - b) / c) % 6 : mx === g ? (b - r) / c + 2 : (r - g) / c + 4; h *= 60; if (h < 0) h += 360; }
      const isPlant = (s > 0.18 && v > 0.16 && h >= 18 && h <= 170) || (s > 0.3 && v > 0.12 && v < 0.85 && (h < 18 || h > 340));
      if (s > 0.15 && v > 0.12 && h >= 55 && h <= 170) green++;
      if (!isPlant) continue;
      plant++;
      if (h < 62 || h > 300 || v < 0.24) { les++; md.data[i] = 227; md.data[i + 1] = 105; md.data[i + 2] = 60; md.data[i + 3] = 150; }
    }
    maskCan.getContext("2d").putImageData(md, 0, 0);
    const n = d.length / 4;
    // brown soil or bark passes the colour test, so a real leaf must also show some green tissue
    return { leaf: green / n >= 0.05 ? plant / n : green / n, lesion: plant ? les / plant : 0 };
  }

  // ---------- frame drawing ----------
  const frame = $("#frame"), fctx = frame.getContext("2d"), overlay = $("#overlay"), octx = overlay.getContext("2d");
  function coverT(w, h) { const s = Math.max(640 / w, 480 / h); return { s, dx: (640 - w * s) / 2, dy: (480 - h * s) / 2 }; }
  const BOX = (() => { const side = 0.62 * 640; return { x: (640 - side) / 2, y: (480 - side) / 2, w: side, h: side }; })();
  function drawSource(src, w, h) {
    const T = coverT(w, h);
    fctx.drawImage(src, T.dx, T.dy, w * T.s, h * T.s);
    return T;
  }
  function drawOverlay(rect, showMask) {
    octx.clearRect(0, 0, 640, 480);
    if (showMask && $("#showMask").checked) { octx.globalAlpha = 1; octx.imageSmoothingEnabled = false; octx.drawImage(maskCan, rect.x, rect.y, rect.w, rect.h); }
  }

  // ---------- inference ----------
  async function classify(tta) {
    const t0 = performance.now();
    const probs = tf.tidy(() => {
      const x = tf.browser.fromPixels(inCan).toFloat().div(255).expandDims(0);
      let batch = x;
      if (tta) {
        const fl = tf.reverse(x, 2);
        const cr = tf.image.cropAndResize(x, [[0.06, 0.06, 0.94, 0.94]], [0], [224, 224]);
        batch = tf.concat([x, fl, cr], 0);
      }
      return S.net.run(batch);
    });
    const arr = await probs.array(); probs.dispose();
    setMs((performance.now() - t0) / (tta ? 3 : 1));
    // outputs: 38 disease probabilities, then the leaf / not-leaf probability
    const avg = new Array(38).fill(0); let leafP = 0;
    arr.forEach((p) => { for (let i = 0; i < 38; i++) avg[i] += p[i] / arr.length; leafP += (p[38] ?? 1) / arr.length; });
    const tops = arr.map((p) => { const q = p.slice(0, 38); return q.indexOf(Math.max(...q)); });
    return { avg, tops, leafP };
  }
  function top3(p) { return p.map((v, i) => [i, v]).sort((a, b) => b[1] - a[1]).slice(0, 3); }
  const pretty = (c) => { const k = window.KB[c]; return k ? `${k.crop} · ${k.name}` : c; };

  const LOCK_SVG = `<svg viewBox="0 0 320 210" aria-hidden="true">
    <defs><linearGradient id="vlg" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#c6e86b" stop-opacity=".55"/><stop offset="1" stop-color="#c6e86b" stop-opacity="0"/></linearGradient>
    <linearGradient id="vlr" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#e3693c" stop-opacity=".6"/><stop offset="1" stop-color="#e3693c" stop-opacity="0"/></linearGradient></defs>
    <g opacity=".7"><path d="M0 186h320" stroke="#5a4630" stroke-width="2"/><path d="M14 196h40M80 200h52M160 195h36M224 201h60" stroke="#3f3222" stroke-width="3" stroke-linecap="round"/>
    <circle cx="70" cy="190" r="3" fill="#6b5338"/><circle cx="210" cy="191" r="2.5" fill="#6b5338"/><circle cx="262" cy="189" r="2" fill="#6b5338"/></g>
    <g class="vl-miss" fill="none" stroke="#e3693c" stroke-width="3" stroke-linecap="round"><path d="M112 172l12 12M124 172l-12 12"/><path d="M196 172l12 12M208 172l-12 12"/><path d="M154 178l12 12M166 178l-12 12"/></g>
    <g class="vl-drone"><g class="vl-hover">
      <polygon class="vl-cone" points="160,96 100,188 220,188" fill="url(#vlg)"/>
      <polygon class="vl-cone-red" points="160,96 100,188 220,188" fill="url(#vlr)"/>
      <g stroke="#9fb3a5" stroke-width="4" stroke-linecap="round"><path d="M120 66h80"/><path d="M128 66l-4 -10M192 66l4 -10"/></g>
      <ellipse class="vl-prop" cx="122" cy="54" rx="24" ry="3.5" fill="#cfe1d4" opacity=".75"/><ellipse class="vl-prop" cx="198" cy="54" rx="24" ry="3.5" fill="#cfe1d4" opacity=".75"/>
      <rect x="140" y="58" width="40" height="22" rx="7" fill="#1d3226" stroke="#c6e86b" stroke-width="2"/>
      <circle cx="160" cy="69" r="5" fill="#0a130f" stroke="#7fd6e8" stroke-width="2"/>
      <path d="M154 80h12v10h-12z" fill="#2a4434"/><path d="M157 90h6v6h-6z" fill="#7fd6e8"/>
      <g class="vl-lock"><circle class="vl-ring" cx="160" cy="120" r="16" fill="none" stroke="#e3693c" stroke-width="3"/>
        <path class="vl-shackle" d="M151 116v-7a9 9 0 0 1 18 0v7" fill="none" stroke="#e3693c" stroke-width="4"/>
        <rect x="146" y="115" width="28" height="22" rx="4" fill="#e3693c"/><circle cx="160" cy="125" r="3" fill="#0a130f"/><path d="M160 127v5" stroke="#0a130f" stroke-width="2.5"/></g>
    </g></g></svg>`;
  function showLock(on, live) {
    const el = $("#vpLock");
    if (!on) { el.hidden = true; el.innerHTML = ""; S.noLeaf = false; return; }
    if (S.noLeaf && !el.hidden) return;
    S.noLeaf = true; el.hidden = false;
    el.innerHTML = `<div>${LOCK_SVG}<div class="vl-text">NO LEAF IN VIEW · NOZZLE LOCKED</div><div class="vl-sub">${live ? "Bring one leaf into the box to unlock the spray" : "Try a closer photo of a single leaf"}</div></div>`;
  }
  function renderStatus(leaf, cls, conf) {
    const nz = $("#nozzle"), bar = $("#stLeafBar i");
    if (leaf == null) { $("#stLeafV").textContent = "–"; bar.style.width = "0"; $("#stDxV").textContent = "–"; $("#stActV").textContent = "–"; nz.className = "nozzle"; $("#nzT").textContent = "Nozzle locked"; $("#nzS").textContent = "Waiting for a leaf in the box"; return; }
    const ok = leaf >= LEAF_MIN;
    $("#stLeafV").textContent = ok ? "Yes · " + Math.round(leaf * 100) + "%" : "No · " + Math.round(leaf * 100) + "%";
    bar.style.width = Math.min(100, leaf * 100) + "%"; bar.style.background = ok ? "var(--lime)" : "var(--rust)";
    if (!ok) {
      $("#stDxV").textContent = "–"; $("#stActV").textContent = "Locked";
      if (!nz.classList.contains("locked")) { nz.className = "nozzle"; void nz.offsetWidth; nz.className = "nozzle locked"; }
      $("#nzT").textContent = "Nozzle locked"; $("#nzS").textContent = "No leaf detected, nothing will be sprayed";
      return;
    }
    const k = window.KB[cls];
    $("#stDxV").textContent = k ? `${k.crop} · ${k.name}` + (conf != null ? ` (${Math.round(conf * 100)}%)` : "") : "Analysing…";
    $("#stActV").textContent = k ? DEC[k.action].badge : "–";
    nz.className = "nozzle armed"; $("#nzT").textContent = "Nozzle armed";
    $("#nzS").textContent = k ? (k.action === "SPRAY" || k.action === "VECTOR" ? "Ready to spray " + (k.product.actives[0] || "") : "Leaf found, no spray needed") : "Leaf found";
  }
  function renderPreds(p) {
    $("#preds").innerHTML = top3(p).map(([i, v]) => `<div class="pred"><span>${esc(pretty(CLASSES[i]))}</span><em>${(v * 100).toFixed(1)}%</em><div class="bar"><i style="width:${(v * 100).toFixed(1)}%"></i></div></div>`).join("");
  }
  function setLabel(cls, conf, leafOK) {
    const l = $("#vpLabel"); l.hidden = false;
    if (!leafOK) { l.className = "vp-label"; l.textContent = "No leaf in box"; return; }
    const k = window.KB[cls]; l.className = "vp-label " + (k.healthy ? "ok" : "dis"); l.textContent = `${k.crop} · ${k.name} · ${Math.round(conf * 100)}%`;
    $("#viewport").classList.toggle("dis", !k.healthy);
  }

  // ---------- camera ----------
  async function startCamera() {
    if (S.live) return stopCamera();
    $("#camNote").textContent = ""; $("#camNote").className = "note";
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) return camFail("This browser does not offer camera access here.");
    try {
      S.stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 }, height: { ideal: 960 } }, audio: false });
    } catch (e) { return camFail(e && e.name === "NotAllowedError" ? "Camera permission was refused, or this viewer blocks cameras." : "Camera could not start (" + (e.name || e.message) + ")."); }
    const v = $("#video"); v.srcObject = S.stream; await v.play().catch(() => {});
    S.live = true; S.locked = false; S.hist = []; clearResult();
    $("#viewport").classList.add("live"); $("#vpEmpty").hidden = true; $("#btnRescan").hidden = true;
    $("#vpTag").innerHTML = '<span class="dot"></span>LIVE SCAN'; $("#btnCam span").textContent = "Stop camera";
    requestAnimationFrame(drawLive); loopInfer();
  }
  function camFail(msg) {
    $("#camNote").className = "note err";
    $("#camNote").textContent = msg + " Upload a leaf photo or tap a test leaf instead. For live scanning, open the standalone app on a phone or laptop over https or localhost.";
  }
  function stopCamera() {
    S.live = false; if (S.stream) S.stream.getTracks().forEach((t) => t.stop()); S.stream = null;
    $("#viewport").classList.remove("live"); $("#btnCam span").textContent = "Start camera"; $("#vpTag").textContent = "STANDBY";
  }
  let liveT = null;
  function drawLive() {
    if (!S.live) return;
    const v = $("#video");
    if (v.videoWidth && !S.locked) { liveT = drawSource(v, v.videoWidth, v.videoHeight); drawOverlay(BOX, true); }
    requestAnimationFrame(drawLive);
  }
  async function loopInfer() {
    while (S.live && !S.locked) {
      const v = $("#video");
      if (!S.ready || !v.videoWidth || !liveT) { await new Promise((r) => setTimeout(r, 150)); continue; }
      const sx = (BOX.x - liveT.dx) / liveT.s, sy = (BOX.y - liveT.dy) / liveT.s, sw = BOX.w / liveT.s;
      inCtx.drawImage(v, sx, sy, sw, sw, 0, 0, 224, 224);
      const st = pixelStats();
      $("#viewport").classList.add("analyzing");
      const { avg, leafP } = await classify(false);
      st.leaf = leafP;
      if (st.leaf < LEAF_MIN) {
        S.hist = []; renderStatus(st.leaf); showLock(true, true); $("#vpLabel").hidden = true; $("#viewport").classList.remove("dis");
        $("#vpPerf").textContent = ""; await new Promise((r) => setTimeout(r, 250)); continue;
      }
      showLock(false);
      const [bi, bv] = top3(avg)[0];
      S.hist.push(bi); if (S.hist.length > 4) S.hist.shift();
      renderPreds(avg); setLabel(CLASSES[bi], bv, true); renderStatus(st.leaf, CLASSES[bi], bv);
      $("#vpPerf").textContent = Math.round(S.lastMs) + " ms";
      if (S.hist.length >= 2 && S.hist[S.hist.length - 2] === bi) {
        S.locked = true; $("#viewport").classList.add("lock"); $("#video").pause();
        finalize({ cls: CLASSES[bi], conf: bv, probs: avg, leaf: st.leaf, lesion: st.lesion, source: "Live camera", thumb: snapshot() });
        break;
      }
      await tf.nextFrame();
    }
  }
  function snapshot() { const c = document.createElement("canvas"); c.width = c.height = 120; c.getContext("2d").drawImage(inCan, 0, 0, 120, 120); return c.toDataURL("image/jpeg", 0.75); }

  // ---------- photos ----------
  async function analyseImage(img, label) {
    if (S.live) stopCamera();
    if (!S.ready) { $("#camNote").textContent = "Model is still loading. Try again in a moment."; return; }
    S.locked = false; clearResult(); $("#vpEmpty").hidden = true; $("#viewport").classList.remove("lock", "dis");
    const w = img.naturalWidth || img.width, h = img.naturalHeight || img.height;
    fctx.fillStyle = "#06100b"; fctx.fillRect(0, 0, 640, 480);
    const T = drawSource(img, w, h);
    const m = Math.min(w, h), sx = (w - m) / 2, sy = (h - m) / 2;
    inCtx.drawImage(img, sx, sy, m, m, 0, 0, 224, 224);
    const rect = { x: T.dx + sx * T.s, y: T.dy + sy * T.s, w: m * T.s, h: m * T.s };
    $("#vpTag").textContent = label || "PHOTO";
    $("#viewport").classList.add("analyzing"); octx.clearRect(0, 0, 640, 480);
    showLock(false); renderStatus(null); await new Promise((r) => setTimeout(r, 450));
    const st = pixelStats();
    const { avg, leafP } = await classify(true);
    st.leaf = leafP;
    const ok = st.leaf >= LEAF_MIN;
    if (!ok) {
      renderStatus(st.leaf); showLock(true, false);
      $("#viewport").classList.remove("analyzing"); $("#vpLabel").hidden = true; $("#btnRescan").hidden = false;
      $("#preds").innerHTML = '<p class="small">No crop leaf found, so there is no diagnosis to show.</p>';
      finalize({ cls: null, conf: 0, probs: null, leaf: st.leaf, lesion: 0, source: label === "TEST LEAF" ? "Test leaf" : "Photo", thumb: snapshot() });
      return;
    }
    renderStatus(st.leaf);
    await new Promise((r) => setTimeout(r, 250));
    const [bi, bv] = top3(avg)[0];
    renderPreds(avg); setLabel(CLASSES[bi], bv, true); renderStatus(st.leaf, CLASSES[bi], bv);
    drawOverlay(rect, true);
    $("#viewport").classList.remove("analyzing"); $("#viewport").classList.add("lock");
    $("#vpPerf").textContent = Math.round(S.lastMs) + " ms";
    $("#btnRescan").hidden = false;
    finalize({ cls: CLASSES[bi], conf: bv, probs: avg, leaf: st.leaf, lesion: st.lesion, source: label === "TEST LEAF" ? "Test leaf" : "Photo", thumb: snapshot() });
  }
  function loadFile(f) {
    if (!f || !f.type.startsWith("image/")) { toast("Choose an image file"); return; }
    const url = URL.createObjectURL(f); const img = new Image();
    img.onload = () => { analyseImage(img, "PHOTO"); URL.revokeObjectURL(url); };
    img.onerror = () => toast("That image could not be read");
    img.src = url;
  }

  // ---------- result ----------
  function clearResult() { $("#result").innerHTML = ""; if (S.countdown) { clearInterval(S.countdown); S.countdown = null; } $("#vpLabel").hidden = true; }
  async function finalize(r) {
    const leafOK = r.leaf >= LEAF_MIN;
    if (!leafOK) { renderResult(r, null, "LOCKED", null, false); return; }
    const k = window.KB[r.cls];
    const action = k.action;
    const entry = {
      id: Date.now(), t: new Date().toISOString(), cls: r.cls, crop: k.crop, name: k.name, healthy: !!k.healthy,
      conf: +r.conf.toFixed(4), leaf: +r.leaf.toFixed(3), lesion: +r.lesion.toFixed(3), action, source: r.source,
      product: action === "SPRAY" || action === "VECTOR" ? (k.product.actives[0] || "") : "", thumb: r.thumb,
    };
    await chain(entry); S.log.push(entry); saveLog();
    renderResult(r, k, action, entry, leafOK);
  }
  async function sha(s) {
    try { const b = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(s)); return Array.from(new Uint8Array(b)).map((x) => x.toString(16).padStart(2, "0")).join(""); }
    catch (e) { let h = 2166136261; for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); } return (h >>> 0).toString(16).padStart(8, "0").repeat(8); }
  }
  const body = (e) => JSON.stringify([e.t, e.cls, e.conf, e.action, e.lesion]);
  async function chain(e) { const prev = S.log.length ? S.log[S.log.length - 1].hash : "GENESIS"; e.prev = prev; e.hash = await sha(prev + body(e)); }

  function renderResult(r, k, action, entry, leafOK) {
    const host = $("#result");
    if (!leafOK) {
      host.innerHTML = `<article class="result"><div class="res-lock">${LOCK_SVG}<div><span class="dec-badge dec-UNSURE" style="color:var(--rust);border-color:rgba(227,105,60,.5)">NO LEAF · SPRAY LOCKED</span>
        <p style="margin-top:10px">The leaf detector is only ${Math.round(r.leaf * 100)}% sure a crop leaf is in the box, so the drone keeps its nozzle locked and sprays nothing. No leaf, no chemical.</p>
        <p class="small">Fill the box with one leaf, in daylight, and scan again.</p>
        <div class="res-foot" style="padding:12px 0 0;border:0"><button class="btn primary" id="btnAgain">Scan again</button></div></div></div></article>`;
      $("#btnAgain").onclick = rescan;
      return;
    }
    const dis = !k.healthy, cls = action === "VECTOR" ? "vec" : dis ? "dis" : "";
    const conf = Math.round(r.conf * 100), lesion = Math.round(r.lesion * 100);
    const t3 = top3(r.probs);
    const srcs = (k.src || []).map((s) => window.SOURCES[s]).filter(Boolean);
    const prod = k.product;
    const unsure = false, low = r.conf < 0.5;
    host.innerHTML = `
    <article class="result" aria-live="polite">
      <div class="res-head ${cls}">
        <img class="res-thumb" src="${r.thumb}" alt="Analysed leaf">
        <div style="min-width:0">
          <div class="res-kicker">${esc(r.source)} · ${new Date(entry.t).toLocaleTimeString()} · ${esc(k.type)}</div>
          <div class="res-title">${esc(k.crop)} · ${esc(k.name)}</div>
          <div class="res-sub">${k.healthy ? "No disease signs found." : `Cause: <i>${esc(k.pathogen)}</i>`}</div>
        </div>
        <div class="ring" style="--p:${conf};--c:${unsure ? "var(--mute)" : dis ? "var(--amber)" : "var(--healthy)"}"><b>${conf}%</b><small>CONFIDENCE</small></div>
      </div>
      <div class="decision">
        <span class="dec-badge dec-${action}">${DEC[action].badge}</span>
        <p>${esc(DEC[action].text(k))}${low ? ` <span class="small" style="color:var(--amber)">Low confidence (${conf}%): confirm by eye or rescan closer before spraying.</span>` : ""}</p>
        <div class="res-metrics">
          <span><b>${lesion}%</b>leaf area off-colour</span>
          <span><b>${Math.round(r.leaf * 100)}%</b>sure it is a leaf</span>
          <span><b>${Math.round(S.lastMs)} ms</b>per frame</span>
        </div>
      </div>
      <div class="res-body">
        <section>
          <h4>${ICON.spray}What to spray</h4>
          ${unsure ? `<p>Nothing yet. Best guesses:</p><ul>${t3.map(([i, v]) => `<li>${esc(pretty(CLASSES[i]))} <span class="small">${(v * 100).toFixed(0)}%</span></li>`).join("")}</ul>`
          : prod ? `<p><b>${esc(prod.kind)}</b></p><div>${prod.actives.map((a) => `<span class="pill">${esc(a)}</span>`).join("")}</div>
              ${prod.organic && prod.organic.length ? `<p class="small" style="margin-top:6px">Organic option</p><div>${prod.organic.map((a) => `<span class="pill org">${esc(a)}</span>`).join("")}</div>` : ""}
              <p class="small" style="margin-top:8px">Use only products registered for this crop in India (CIB&amp;RC) and follow the label dose and waiting period.</p>`
          : `<p>No chemical needed.</p><p class="small">The drone flies over this patch without spraying.</p>`}
        </section>
        <section>
          <h4>${ICON.soil}Fertiliser and nutrition</h4>
          <p><b>Fertiliser spray: ${k.healthy ? "not needed" : "will not cure this"}</b></p>
          <p>${esc(k.nutrition)}</p>
          <p class="small">Base any NPK on a soil test (Soil Health Card).</p>
        </section>
        <section>
          <h4>${ICON.hand}In the field</h4>
          <p class="small" style="margin-bottom:6px">${esc(k.symptoms)}</p>
          <ul>${k.cultural.map((c) => `<li>${esc(c)}</li>`).join("")}</ul>
          <p class="small src" style="margin-top:8px">Source: ${srcs.map((s) => `<a href="${s[1]}" target="_blank" rel="noopener">${esc(s[0])}</a>`).join(" · ")}</p>
        </section>
      </div>
      <div class="res-foot">
        ${unsure ? `<button class="btn primary" id="btnAgain">Scan again</button>` : `<button class="btn primary" id="btnFly">${ICON.spray} Fly the mission</button><button class="btn ghost" id="btnAgain">Scan another leaf</button>
        <span class="countdown" id="cd" ${S.auto ? "" : "hidden"}><svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="10" fill="none" stroke="#24392d" stroke-width="3"/><circle class="fg" cx="12" cy="12" r="10" fill="none" stroke-width="3" transform="rotate(-90 12 12)"/></svg><span id="cdt">Drone launches in 4 s</span> <button class="btn ghost" id="cdStop" style="padding:4px 8px;font-size:12px">Cancel</button></span>`}
        <span class="small" style="margin-left:auto">Logged · hash <span class="mono">${entry.hash ? entry.hash.slice(0, 10) : "–"}</span></span>
      </div>
    </article>`;
    host.scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "nearest" });
    $("#btnAgain").onclick = rescan;
    if (!unsure) {
      const fly = () => {
        if (S.countdown) { clearInterval(S.countdown); S.countdown = null; } const cd = $("#cd"); if (cd) cd.hidden = true;
        FPV.play({ action, label: `${k.crop} · ${k.name}`, conf: r.conf, product: prod && action !== "NONE" ? prod.actives[0] : "", decisionText: DEC[action].text(k), onDone: () => {} });
      };
      $("#btnFly").onclick = fly;
      if (S.auto) {
        let n = 4; const fg = $("#cd circle.fg");
        S.countdown = setInterval(() => { n--; if (fg) fg.style.strokeDashoffset = String(63 * (1 - n / 4)); const t = $("#cdt"); if (t) t.textContent = "Drone launches in " + n + " s"; if (n <= 0) fly(); }, 1000);
        $("#cdStop").onclick = () => { clearInterval(S.countdown); S.countdown = null; $("#cd").hidden = true; };
      }
    }
  }
  function rescan() {
    clearResult(); S.locked = false; S.hist = []; $("#viewport").classList.remove("lock", "dis"); octx.clearRect(0, 0, 640, 480);
    renderStatus(null); showLock(false); $("#preds").innerHTML = '<p class="small">Top three classes appear here as soon as a leaf is analysed.</p>';
    if (S.stream) { $("#video").play(); S.live = true; requestAnimationFrame(drawLive); loopInfer(); }
    else { fctx.clearRect(0, 0, 640, 480); $("#vpEmpty").hidden = false; $("#btnRescan").hidden = true; $("#vpTag").textContent = "STANDBY"; $("#vpPerf").textContent = ""; }
  }

  // ---------- samples ----------
  function renderSamples() {
    const name = (x) => (x.truth ? pretty(x.truth) : "Not a leaf");
    const tag = { field: "Field", lab: "Lab", neg: "Object" };
    $("#samples").innerHTML = window.SAMPLES.map((x, i) => `<button class="sample" data-i="${i}" title="${esc(name(x))}"><img src="${x.file}" alt="" loading="lazy"><em class="skind ${x.kind}">${tag[x.kind] || ""}</em><span>${esc(name(x))}</span></button>`).join("");
    $$("#samples .sample").forEach((b) => (b.onclick = () => { const img = new Image(); img.onload = () => analyseImage(img, "TEST LEAF"); img.src = window.SAMPLES[+b.dataset.i].file; }));
  }

  // ---------- log ----------
  async function verifyChain() {
    let prev = "GENESIS";
    for (const e of S.log) { if (e.prev !== prev) return false; if ((await sha(prev + body(e))) !== e.hash) return false; prev = e.hash; }
    return true;
  }
  async function renderLog() {
    const v = $("#view-log"); loadLog();
    const n = S.log.length, sprays = S.log.filter((e) => e.action === "SPRAY" || e.action === "VECTOR").length;
    const healthy = S.log.filter((e) => e.action === "NONE").length, held = S.log.filter((e) => e.action === "UNSURE").length;
    const ok = await verifyChain();
    v.innerHTML = `
      <div class="tiles">
        <div class="tile"><b>${n}</b><span>Leaves scanned</span></div>
        <div class="tile"><b style="color:var(--cyan)">${sprays}</b><span>Patches that need a spray</span></div>
        <div class="tile"><b style="color:var(--healthy)">${n ? Math.round(((n - sprays) / n) * 100) : 0}%</b><span>Scans that used zero chemical</span></div>
        <div class="tile"><b style="color:${ok ? "var(--lime)" : "var(--rust)"}">${ok ? "Intact" : "Broken"}</b><span>Spray Passport hash chain</span></div>
      </div>
      <div class="card">
        <h2>Spray Passport · this device</h2>
        <p class="small" style="margin-top:-6px">Every decision is chained to the one before it with SHA-256, so editing any row breaks every later hash. Healthy: ${healthy}.</p>
        <div class="log-actions"><button class="btn" id="cpCsv">Copy as CSV</button><button class="btn" id="dlCsv" hidden>Save CSV file</button><button class="btn ghost warn" id="clr">Clear log</button><span id="clrAsk" hidden class="small">Delete all ${n} entries? <button class="btn ghost warn" id="clrYes" style="padding:4px 10px">Delete</button> <button class="btn ghost" id="clrNo" style="padding:4px 10px">Keep</button></span></div>
        ${n ? `<div class="table-wrap"><table><thead><tr><th></th><th>Time</th><th>Crop · diagnosis</th><th>Confidence</th><th>Off-colour</th><th>Decision</th><th>Product</th><th>Hash</th></tr></thead><tbody>
          ${S.log.slice().reverse().map((e) => `<tr><td>${e.thumb ? `<img src="${e.thumb}" alt="">` : ""}</td><td class="mono">${new Date(e.t).toLocaleString()}</td><td>${esc(e.crop)} · ${esc(e.name)}<div class="small">${esc(e.source)}</div></td><td class="mono">${Math.round(e.conf * 100)}%</td><td class="mono">${Math.round(e.lesion * 100)}%</td><td><span class="tag ${e.action}">${e.action}</span></td><td>${esc(e.product || "–")}</td><td><span class="hash">${(e.hash || "").slice(0, 12)}</span></td></tr>`).join("")}
          </tbody></table></div>` : `<div class="empty">No scans yet. Analyse a leaf on the Scan tab and it appears here with its spray decision.</div>`}
      </div>`;
    const csv = () => ["time,crop,diagnosis,confidence,offcolour,decision,product,hash"].concat(S.log.map((e) => [e.t, e.crop, e.name, e.conf, e.lesion, e.action, e.product, e.hash].map((x) => `"${String(x ?? "").replace(/"/g, '""')}"`).join(","))).join("\n");
    $("#cpCsv").onclick = () => { const t = csv(); (navigator.clipboard ? navigator.clipboard.writeText(t) : Promise.reject()).then(() => toast("Copied " + n + " rows")).catch(() => toast("Copy blocked here")); };
    try {
      const dl = window.claude && window.claude.use ? await window.claude.use("downloads") : null;
      if (dl) { $("#dlCsv").hidden = false; $("#dlCsv").onclick = () => dl.save({ filename: "spray-passport.csv", data: csv() }).catch(() => {}); }
    } catch (e) {}
    $("#clr").onclick = () => { $("#clrAsk").hidden = false; };
    $("#clrNo") && ($("#clrNo").onclick = () => { $("#clrAsk").hidden = true; });
    $("#clrYes") && ($("#clrYes").onclick = () => { S.log = []; saveLog(); renderLog(); toast("Log cleared"); });
  }

  // ---------- library ----------
  let libCrop = "All", libQ = "";
  function renderLibrary() {
    const v = $("#view-library");
    const crops = ["All", ...Array.from(new Set(Object.values(window.KB).map((k) => k.crop)))];
    const items = Object.entries(window.KB).filter(([c, k]) => (libCrop === "All" || k.crop === libCrop) && (!libQ || (k.name + " " + k.crop + " " + (k.pathogen || "")).toLowerCase().includes(libQ)))
      .sort((a, b) => (a[1].healthy - b[1].healthy) || a[1].crop.localeCompare(b[1].crop));
    v.innerHTML = `
      <div class="lib-tools"><input id="libq" type="search" placeholder="Search disease, crop or pathogen" value="${esc(libQ)}">${crops.map((c) => `<button class="fchip" aria-pressed="${c === libCrop}" data-c="${esc(c)}">${esc(c)}</button>`).join("")}</div>
      <p class="small">${items.length} of 38 classes the model knows. Tap a card for treatment and sources.</p>
      <div class="lib">${items.map(([c, k]) => `
        <details><summary><span class="crop">${esc(k.crop)} · ${esc(k.type)}</span><span class="nm">${esc(k.name)}</span><span class="meta"><span class="tag ${k.action}">${DEC[k.action].badge}</span><span class="small mono">test acc ${Math.round((window.MODEL_META.per_class[c] || 0) * 100)}%</span></span></summary>
        <div class="more"><p><b>Looks like:</b> ${esc(k.symptoms)}</p>${k.product ? `<p><b>${esc(k.product.kind)}:</b> ${esc(k.product.actives.join(", "))}${k.product.organic && k.product.organic.length ? ` · organic: ${esc(k.product.organic.join(", "))}` : ""}</p>` : ""}<p><b>Nutrition:</b> ${esc(k.nutrition)}</p><p class="small src">${(k.src || []).map((s) => window.SOURCES[s]).filter(Boolean).map((s) => `<a href="${s[1]}" target="_blank" rel="noopener">${esc(s[0])}</a>`).join(" · ")}</p></div></details>`).join("")}</div>`;
    $("#libq").oninput = (e) => { libQ = e.target.value.toLowerCase(); const pos = e.target.selectionStart; renderLibrary(); const q = $("#libq"); q.focus(); q.setSelectionRange(pos, pos); };
    $$(".fchip", v).forEach((b) => (b.onclick = () => { libCrop = b.dataset.c; renderLibrary(); }));
  }

  // ---------- how ----------
  function renderHow() {
    const M = window.MODEL_META, b = M.binary;
    const missed = b["False->True"] || 0, falseAlarm = b["True->False"] || 0;
    const nd = (b["False->False"] || 0) + missed, nh = (b["True->True"] || 0) + falseAlarm;
    const rows = Object.entries(M.per_class).sort((a, b) => a[1] - b[1]);
    $("#view-how").innerHTML = `
      <div class="pipe">
        ${[["01", "Camera", "Frames from the phone or drone camera; centre box cropped to 224 px."], ["02", "Leaf lock", "A second model head trained on leaves vs everyday objects, soil and skin. Below 50% the nozzle stays locked."], ["03", "LeafNet", "MobileNetV3 runs on this device and scores 38 crop-disease classes, trained on lab and field photos."], ["04", "Diagnosis", "Top class becomes the diagnosis; live mode waits for two matching frames to avoid flicker."], ["05", "Decision", "Spray, control the vector, remove, monitor or leave alone."], ["06", "Drone", "Pod fires the nozzle over that patch and writes the Spray Passport."]]
          .map((s) => `<div class="step"><span class="k">${s[0]}</span><b>${s[1]}</b><p>${s[2]}</p></div>`).join("")}
      </div>
      <div class="how-grid">
        <div class="card">
          <h2>Model card</h2>
          <dl class="kv">
            <dt>Network</dt><dd>MobileNetV3-Large (ImageNet start), last blocks fine-tuned, two heads: 38 diseases + leaf / not-leaf</dd>
            <dt>Training images</dt><dd>${M.n_train.toLocaleString()} originals (PlantVillage lab photos, PlantDoc web-scraped field photos, ${M.n_neg.toLocaleString()} non-leaf photos) plus augmented copies</dd>
            <dt>Lab photos, held out</dt><dd style="color:var(--lime)">${(M.test_acc * 100).toFixed(1)}% of ${M.n_test.toLocaleString()}</dd>
            <dt>Field photos, held out</dt><dd>${(M.pd_acc * 100).toFixed(1)}% of ${M.pd_n} (right answer in top 3: ${(M.pd_top3 * 100).toFixed(0)}%)</dd>
            <dt>Non-leaf photos let through</dt><dd>${M.gate.nonleaf_passed} of ${M.gate.nonleaf_n}</dd>
            <dt>Leaves wrongly locked</dt><dd>${M.gate.leaf_missed} of ${M.gate.leaf_n}</dd>
            <dt>Diseased called healthy</dt><dd>${missed} of ${nd} (${((missed / nd) * 100).toFixed(1)}%)</dd>
            <dt>Healthy called diseased</dt><dd>${falseAlarm} of ${nh} (${((falseAlarm / nh) * 100).toFixed(1)}%)</dd>
            <dt>Runs on</dt><dd>TensorFlow.js in this browser; no image leaves the device</dd>
          </dl>
          <h3 style="margin-top:18px">Limits to say out loud</h3>
          <ul class="lim">
            <li>Lab photos (one leaf, plain background) score far higher than field photos with clutter, shadows and many leaves. Fill the box with one leaf for the best answer.</li>
            <li>It knows 38 classes across 14 crops. Anything else (paddy, chilli, cotton) gets forced into the nearest class, so check the confidence before acting on it.</li>
            <li>Off-colour area is a colour measure, not a lab severity score.</li>
            <li>Treatments are summaries of the linked extension guides. Confirm with your KVK and the product label before spraying.</li>
          </ul>
          <h3 style="margin-top:18px">On the real drone</h3>
          <p class="small">The flown Neerakrit prototype runs YOLOv8n on a Raspberry Pi with an AI accelerator and fires the pump over MAVLink. This web prototype shows the same gate-then-spray flow on any phone or laptop.</p>
        </div>
        <div class="card">
          <h2>Accuracy by class (held-out lab photos)</h2>
          <div class="cls-chart">${rows.map(([c, v]) => `<div class="cls-row"><span title="${esc(pretty(c))}">${esc(pretty(c))}</span><div class="bar"><i style="width:${v * 100}%;background:${v >= 0.95 ? "var(--lime)" : v >= 0.9 ? "var(--amber)" : "var(--rust)"}"></i></div><em>${Math.round(v * 100)}%</em></div>`).join("")}</div>
        </div>
      </div>`;
  }

  // ---------- wire ----------
  $("#btnCam").onclick = startCamera;
  $("#file").onchange = (e) => { loadFile(e.target.files[0]); e.target.value = ""; };
  $("#btnRescan").onclick = rescan;
  $("#auto").onchange = (e) => { S.auto = e.target.checked; };
  $("#showMask").onchange = () => { if (S.live) return; };
  const vp = $("#viewport");
  vp.addEventListener("dragover", (e) => { e.preventDefault(); vp.style.outline = "2px dashed var(--lime)"; });
  vp.addEventListener("dragleave", () => (vp.style.outline = ""));
  vp.addEventListener("drop", (e) => { e.preventDefault(); vp.style.outline = ""; loadFile(e.dataTransfer.files[0]); });
  document.addEventListener("paste", (e) => { const f = Array.from(e.clipboardData?.files || [])[0]; if (f) loadFile(f); });
  loadLog(); renderSamples(); renderStatus(null); route(); initModel();
})();
