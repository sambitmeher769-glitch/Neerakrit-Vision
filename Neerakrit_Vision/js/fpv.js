// FPV drone mission: Three.js field + HUD. window.FPV.play(opts)
(function () {
  const ACTION = {
    SPRAY: { tint: 0x9fe8ff, css: "var(--cyan)", verb: "SPRAYING", done: "PATCH TREATED" },
    VECTOR: { tint: 0xffd27a, css: "var(--amber)", verb: "VECTOR CONTROL SPRAY", done: "VECTOR SPRAY DONE · PLANTS FLAGGED" },
    REMOVE: { tint: null, css: "var(--rust)", verb: "MARKING FOR REMOVAL", done: "PLANTS FLAGGED FOR REMOVAL" },
    MONITOR: { tint: null, css: "var(--amber)", verb: "TAGGING FOR MONITORING", done: "PATCH TAGGED · NO SPRAY" },
    NONE: { tint: null, css: "var(--healthy)", verb: "HEALTHY · NO SPRAY", done: "HEALTHY · ZERO CHEMICAL" },
  };
  const ease = (x) => (x < 0 ? 0 : x > 1 ? 1 : x * x * (3 - 2 * x));
  const lerp = (a, b, t) => a + (b - a) * t;
  let state = null;

  function el(tag, cls, html) { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; }

  function buildHUD(root) {
    const hud = el("div", "fpv-hud");
    hud.innerHTML = `
      <div class="fpv-props"><span class="prop l"></span><span class="prop r"></span><span class="arm l"></span><span class="arm r"></span></div>
      <div class="fpv-top"><span class="rec">● REC</span><span id="fpvMode">AUTO MISSION</span><span id="fpvSats">GPS 3D · 14 SAT</span><span id="fpvBat">BAT 98%</span><span>LINK 915 MHz</span></div>
      <div class="fpv-alt"><b id="fpvAlt">0.0</b><small>ALT m</small></div>
      <div class="fpv-spd"><b id="fpvSpd">0.0</b><small>GS m/s</small></div>
      <div class="fpv-reticle"><i></i><i></i><i></i><i></i></div>
      <div class="fpv-target" id="fpvTarget"></div>
      <div class="fpv-status" id="fpvStatus">PRE-FLIGHT CHECK</div>
      <canvas class="fpv-map" id="fpvMap" width="150" height="150"></canvas>
      <div class="fpv-nozzles"><span></span><span></span></div>
      <div class="fpv-vignette"></div>`;
    root.appendChild(hud);
    return hud;
  }

  function makeScene(opts) {
    const scene = new THREE.Scene();
    const sky = new THREE.Color(0xb7d3e0);
    scene.background = sky; scene.fog = new THREE.Fog(sky, 25, 150);
    scene.add(new THREE.HemisphereLight(0xeaf6ff, 0x5a4326, 0.95));
    const sun = new THREE.DirectionalLight(0xfff0d0, 0.9); sun.position.set(40, 60, 25); scene.add(sun);
    const ground = new THREE.Mesh(new THREE.PlaneGeometry(600, 600), new THREE.MeshLambertMaterial({ color: 0x6e5236 }));
    ground.rotation.x = -Math.PI / 2; scene.add(ground);
    const ROWS = 17, X0 = -9.6, DX = 1.2, Z0 = 24, Z1 = -170, DZ = 0.55;
    const bedGeo = new THREE.BoxGeometry(0.75, 0.14, Z0 - Z1);
    const bedMat = new THREE.MeshLambertMaterial({ color: 0x7d5d3c });
    for (let r = 0; r < ROWS; r++) { const b = new THREE.Mesh(bedGeo, bedMat); b.position.set(X0 + r * DX, 0.07, (Z0 + Z1) / 2); scene.add(b); }
    const perRow = Math.floor((Z0 - Z1) / DZ);
    const geo = new THREE.IcosahedronGeometry(0.34, 0);
    const mat = new THREE.MeshLambertMaterial({ color: 0xffffff });
    const mesh = new THREE.InstancedMesh(geo, mat, ROWS * perRow);
    const dummy = new THREE.Object3D(), col = new THREE.Color();
    const T = { x: 0, z: -60, r: 2.4 };
    const diseased = [];
    let k = 0;
    for (let r = 0; r < ROWS; r++) for (let i = 0; i < perRow; i++) {
      const x = X0 + r * DX + (Math.random() - 0.5) * 0.12, z = Z0 - i * DZ + (Math.random() - 0.5) * 0.1;
      const s = 0.75 + Math.random() * 0.55;
      dummy.position.set(x, 0.3 + s * 0.18, z); dummy.scale.set(s, s * (0.9 + Math.random() * 0.4), s); dummy.rotation.y = Math.random() * 6.28; dummy.updateMatrix();
      mesh.setMatrixAt(k, dummy.matrix);
      const d = Math.hypot(x - T.x, z - T.z);
      if (opts.action !== "NONE" && d < T.r * (0.75 + Math.random() * 0.45)) {
        col.setHSL(0.09 + Math.random() * 0.05, 0.62, 0.36 + Math.random() * 0.12); diseased.push(k);
      } else col.setHSL(0.27 + Math.random() * 0.06, 0.55, 0.26 + Math.random() * 0.1);
      mesh.setColorAt(k, col); k++;
    }
    scene.add(mesh);
    // edge trees + hills
    const treeGeo = new THREE.ConeGeometry(1.6, 5, 6), treeMat = new THREE.MeshLambertMaterial({ color: 0x2f5a35 });
    for (let i = 0; i < 46; i++) { const t = new THREE.Mesh(treeGeo, treeMat); const side = i % 2 ? 1 : -1; t.position.set(side * (16 + Math.random() * 10), 2.5, 20 - i * 4.2); t.scale.setScalar(0.7 + Math.random() * 0.7); scene.add(t); }
    const hillMat = new THREE.MeshLambertMaterial({ color: 0x5f8a5a });
    for (let i = 0; i < 7; i++) { const h = new THREE.Mesh(new THREE.IcosahedronGeometry(30 + Math.random() * 20, 1), hillMat); h.position.set(-120 + i * 40, -14, -190 - Math.random() * 30); scene.add(h); }
    // pad
    const pad = new THREE.Mesh(new THREE.CylinderGeometry(1.4, 1.4, 0.08, 32), new THREE.MeshLambertMaterial({ color: 0x2b2f2c }));
    pad.position.set(0, 0.04, 12); scene.add(pad);
    // target ring
    const ring = new THREE.Mesh(new THREE.RingGeometry(T.r + 0.15, T.r + 0.45, 64), new THREE.MeshBasicMaterial({ color: 0xff5a3c, transparent: true, opacity: 0.85, side: THREE.DoubleSide }));
    ring.rotation.x = -Math.PI / 2; ring.position.set(T.x, 0.75, T.z); scene.add(ring);
    // flag (for remove / monitor)
    const flag = new THREE.Group();
    const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.04, 2.2, 8), new THREE.MeshLambertMaterial({ color: 0xeeeeee })); pole.position.y = 1.1; flag.add(pole);
    const cloth = new THREE.Mesh(new THREE.PlaneGeometry(0.8, 0.5), new THREE.MeshBasicMaterial({ color: opts.action === "REMOVE" ? 0xff4d2e : 0xffc23d, side: THREE.DoubleSide }));
    cloth.position.set(0.42, 1.95, 0); flag.add(cloth);
    flag.position.set(T.x + 0.4, 12, T.z); flag.visible = false; scene.add(flag);
    // particles
    const N = 1800, pg = new THREE.BufferGeometry(), pos = new Float32Array(N * 3), vel = new Float32Array(N * 3), life = new Float32Array(N);
    pg.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    const dc = document.createElement("canvas"); dc.width = dc.height = 32; const dx = dc.getContext("2d"); const gr = dx.createRadialGradient(16, 16, 0, 16, 16, 16); gr.addColorStop(0, "rgba(255,255,255,1)"); gr.addColorStop(0.45, "rgba(255,255,255,.55)"); gr.addColorStop(1, "rgba(255,255,255,0)"); dx.fillStyle = gr; dx.fillRect(0, 0, 32, 32);
    const pm = new THREE.PointsMaterial({ color: ACTION[opts.action].tint || 0xffffff, size: 0.06, map: new THREE.CanvasTexture(dc), transparent: true, opacity: 0.8, depthWrite: false });
    const pts = new THREE.Points(pg, pm); pts.frustumCulled = false; scene.add(pts);
    for (let i = 0; i < N; i++) { life[i] = 0; pos[i * 3 + 1] = -99; }
    return { scene, mesh, diseased, ring, flag, T, P: { N, pos, vel, life, pg, next: 0 } };
  }

  function play(opts) {
    stop();
    const root = document.getElementById("fpv");
    root.hidden = false; root.innerHTML = "";
    const reduce = window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches;
    const A = ACTION[opts.action] || ACTION.NONE;
    if (!window.THREE || reduce) { return summary(root, opts, A, true); }
    const canvas = el("canvas", "fpv-canvas"); root.appendChild(canvas);
    const hud = buildHUD(root);
    const close = el("button", "fpv-close", "Skip ✕"); close.onclick = () => { stop(); summary(root, opts, A, false); }; root.appendChild(close);
    let renderer;
    try { renderer = new THREE.WebGLRenderer({ canvas, antialias: true }); } catch (e) { return summary(root, opts, A, true); }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    const S = makeScene(opts);
    const cam = new THREE.PerspectiveCamera(78, 1, 0.05, 400);
    const resize = () => { const w = root.clientWidth, h = root.clientHeight; renderer.setSize(w, h, false); cam.aspect = w / h; cam.updateProjectionMatrix(); };
    resize(); window.addEventListener("resize", resize);
    const $ = (id) => hud.querySelector("#" + id);
    const statusEl = $("fpvStatus"), tgt = $("fpvTarget"), map = $("fpvMap").getContext("2d");
    const spray = opts.action === "SPRAY" || opts.action === "VECTOR";
    const hold = spray ? 4.2 : 2.4;
    const tHover = 8.6, tAct = 9.8, tLeave = tAct + hold, tEnd = tLeave + 3.2;
    const path = [];
    let start = null, last = null, prevPos = new THREE.Vector3(0, 0.4, 12), msg = "";
    const say = (m, cls) => { if (m !== msg) { msg = m; statusEl.textContent = m; statusEl.className = "fpv-status " + (cls || ""); } };
    const tmp = new THREE.Vector3();
    function pose(t) {
      let y, z, pitch;
      if (t < 2) { const e = ease(t / 2); y = lerp(0.4, 4.6, e); z = 12; pitch = lerp(-5, -22, e); }
      else if (t < tHover) { const e = (t - 2) / (tHover - 2); const ee = e < 0.75 ? e : 0.75 + (1 - Math.pow(1 - (e - 0.75) / 0.25, 2)) * 0.25; y = lerp(4.6, 3.6, ease(e)); z = lerp(12, -55.5, ee); pitch = lerp(-22, -38, ease(e)); }
      else if (t < tAct) { const e = ease((t - tHover) / (tAct - tHover)); y = lerp(3.6, 2.9, e); z = lerp(-55.5, -58.2, e); pitch = lerp(-38, -62, e); }
      else if (t < tLeave) { y = 2.9 + Math.sin(t * 3) * 0.04; z = -58.2 + Math.sin(t * 1.3) * 0.05; pitch = -62; }
      else { const e = ease((t - tLeave) / (tEnd - tLeave)); y = lerp(2.9, 6.5, e); z = lerp(-58.2, -84, e * e); pitch = lerp(-62, -18, e); }
      const roll = Math.sin(t * 1.7) * 0.012 + (t > 2 && t < tHover ? Math.sin(t * 0.9) * 0.02 : 0);
      return { x: Math.sin(t * 0.6) * 0.15, y, z, pitch: pitch * Math.PI / 180, roll };
    }
    function emit(dt) {
      const P = S.P, rate = 900;
      let n = Math.floor(rate * dt);
      cam.updateMatrixWorld();
      while (n-- > 0) {
        const i = P.next; P.next = (P.next + 1) % P.N;
        const side = i % 2 ? 1 : -1;
        tmp.set(side * 0.55, -0.35, -0.55).applyMatrix4(cam.matrixWorld);
        P.pos[i * 3] = tmp.x; P.pos[i * 3 + 1] = tmp.y; P.pos[i * 3 + 2] = tmp.z;
        P.vel[i * 3] = (Math.random() - 0.5) * 1.6 + side * 0.4; P.vel[i * 3 + 1] = -3.5 - Math.random() * 2.5; P.vel[i * 3 + 2] = (Math.random() - 0.5) * 1.6 - 0.9;
        P.life[i] = 1.4;
      }
    }
    function stepParticles(dt) {
      const P = S.P;
      for (let i = 0; i < P.N; i++) {
        if (P.life[i] <= 0) continue;
        P.life[i] -= dt; P.vel[i * 3 + 1] -= 3.0 * dt;
        P.pos[i * 3] += P.vel[i * 3] * dt; P.pos[i * 3 + 1] += P.vel[i * 3 + 1] * dt; P.pos[i * 3 + 2] += P.vel[i * 3 + 2] * dt;
        if (P.pos[i * 3 + 1] < 0.45) { P.pos[i * 3 + 1] = 0.45; P.vel[i * 3] *= 0.5; P.vel[i * 3 + 2] *= 0.5; P.vel[i * 3 + 1] = 0; }
        if (P.life[i] <= 0) P.pos[i * 3 + 1] = -99;
      }
      P.pg.attributes.position.needsUpdate = true;
    }
    const healCol = new THREE.Color();
    function healPatch(f) { // treated plants slowly turn back toward green tint
      if (opts.action !== "SPRAY" && opts.action !== "VECTOR") return;
      const c = new THREE.Color();
      for (const k of S.diseased) { S.mesh.getColorAt(k, c); healCol.setHSL(0.24, 0.5, 0.32); c.lerp(healCol, f * 0.04); S.mesh.setColorAt(k, c); }
      S.mesh.instanceColor.needsUpdate = true;
    }
    function drawMap(p) {
      const w = 150, h = 150, sx = (x) => w / 2 + x * 3.2, sz = (z) => h - 14 - (12 - z) * 1.25;
      map.clearRect(0, 0, w, h);
      map.fillStyle = "rgba(10,20,14,.72)"; map.fillRect(0, 0, w, h);
      map.strokeStyle = "rgba(198,232,107,.25)"; map.lineWidth = 1;
      for (let r = -9.6; r <= 9.6; r += 1.2) { map.beginPath(); map.moveTo(sx(r), 0); map.lineTo(sx(r), h); map.stroke(); }
      map.strokeStyle = "#ff5a3c"; map.lineWidth = 2; map.beginPath(); map.arc(sx(S.T.x), sz(S.T.z), S.T.r * 3.2, 0, 7); map.stroke();
      map.strokeStyle = "#7fd6e8"; map.lineWidth = 2; map.beginPath();
      path.forEach((q, i) => (i ? map.lineTo(sx(q[0]), sz(q[1])) : map.moveTo(sx(q[0]), sz(q[1])))); map.stroke();
      map.fillStyle = "#c6e86b"; map.beginPath(); map.arc(sx(p.x), sz(p.z), 4, 0, 7); map.fill();
      map.fillStyle = "#c6e86b"; map.font = "10px JetBrains Mono, monospace"; map.fillText("P-07", sx(S.T.x) + 10, sz(S.T.z) - 8);
    }
    let sprayed = 0, spdS = 0;
    function frame(ts) {
      if (!state) return;
      if (start === null) { start = ts; last = ts; }
      const t = (ts - start) / 1000, rawDt = Math.max(1e-3, (ts - last) / 1000), dt = Math.min(0.05, rawDt); last = ts;
      const p = pose(t);
      cam.position.set(p.x, p.y, p.z);
      cam.rotation.set(p.pitch, 0, p.roll, "YXZ");
      spdS += (Math.min(14, cam.position.distanceTo(prevPos) / rawDt) - spdS) * 0.15; const spd = spdS; prevPos.copy(cam.position);
      $("fpvAlt").textContent = p.y.toFixed(1); $("fpvSpd").textContent = (t < 0.1 ? 0 : spd).toFixed(1);
      $("fpvBat").textContent = "BAT " + Math.max(71, Math.round(98 - t * 1.4)) + "%";
      if (path.length === 0 || Math.abs(path[path.length - 1][1] - p.z) > 0.6) path.push([p.x, p.z]);
      drawMap(p);
      S.ring.material.opacity = 0.55 + Math.sin(t * 6) * 0.3;
      const distT = Math.max(0, Math.hypot(p.z - S.T.z, p.x - S.T.x) - 1.8);
      if (t < 2) say("TAKEOFF · CLIMB TO 4.6 m");
      else if (t < tHover - 2.5) say("EN ROUTE · SCANNING ROWS · PATCH P-07 IN " + distT.toFixed(0) + " m");
      else if (t < tAct) { say("TARGET ACQUIRED · " + opts.label.toUpperCase(), "warn"); tgt.className = "fpv-target on"; tgt.innerHTML = `<b>${opts.label}</b><span>${Math.round(opts.conf * 100)}% · GATES 3/3 ✓</span>`; }
      else if (t < tLeave) {
        const left = (tLeave - t).toFixed(1);
        if (spray) { emit(dt); sprayed += rawDt; healPatch(dt * 20); say(A.verb + " · " + (opts.product || "") + " · " + left + " s", "act"); hud.classList.add("spraying"); }
        else { say(A.verb, opts.action === "NONE" ? "ok" : "warn"); if (opts.action === "REMOVE" || opts.action === "MONITOR") { S.flag.visible = true; S.flag.position.y = Math.max(0, S.flag.position.y - dt * 14); } }
        if (opts.action === "NONE") S.ring.material.color.set(0x6fd69a);
      } else {
        hud.classList.remove("spraying");
        if (spray || opts.action === "NONE") S.ring.material.color.set(0x6fd69a);
        tgt.className = "fpv-target"; say(A.done + " · LOG WRITTEN", "ok");
      }
      stepParticles(dt);
      renderer.render(S.scene, cam);
      if (t > tEnd) { stop(); summary(root, opts, A, false, sprayed); return; }
      state.raf = requestAnimationFrame(frame);
    }
    state = { renderer, resize, raf: requestAnimationFrame(frame) };
  }

  function summary(root, opts, A, instant, sprayed) {
    root.hidden = false;
    const card = el("div", "fpv-summary");
    const secs = sprayed ? sprayed.toFixed(1) : (opts.action === "SPRAY" || opts.action === "VECTOR" ? "4.2" : "0");
    card.innerHTML = `
      <div class="sum-badge" style="--c:${A.css}">${A.done}</div>
      <h3>${opts.label}</h3>
      <dl>
        <div><dt>Decision</dt><dd>${opts.decisionText}</dd></div>
        <div><dt>Spray time</dt><dd>${secs} s over 1 patch</dd></div>
        <div><dt>Product</dt><dd>${opts.product || "None"}</dd></div>
        <div><dt>Model confidence</dt><dd>${Math.round(opts.conf * 100)}%</dd></div>
      </dl>
      <p class="sum-note">${instant ? "Animation skipped (reduced motion or no WebGL). " : ""}Saved to the Field Log as a Spray Passport entry.</p>
      <div class="sum-actions"><button class="btn ghost" id="fpvReplay">Replay flight</button><button class="btn primary" id="fpvDone">Back to scanner</button></div>`;
    root.appendChild(card);
    card.querySelector("#fpvReplay").onclick = () => play(opts);
    card.querySelector("#fpvDone").onclick = () => { stop(); root.hidden = true; root.innerHTML = ""; opts.onDone && opts.onDone(); };
  }
  function stop() {
    if (!state) return;
    cancelAnimationFrame(state.raf); window.removeEventListener("resize", state.resize);
    try { state.renderer.dispose(); } catch (e) {}
    state = null;
  }
  window.FPV = { play, stop };
})();
