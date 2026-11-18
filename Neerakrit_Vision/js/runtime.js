// LeafNet runtime: runs the exported MobileNetV3 graph with TensorFlow.js ops (NHWC).
(function () {
  function f16ToF32(u16) {
    const out = new Float32Array(u16.length);
    for (let i = 0; i < u16.length; i++) {
      const h = u16[i], s = (h & 0x8000) >> 15, e = (h & 0x7c00) >> 10, f = h & 0x03ff;
      let v;
      if (e === 0) v = (f / 1024) * Math.pow(2, -14);
      else if (e === 31) v = f ? NaN : Infinity;
      else v = (1 + f / 1024) * Math.pow(2, e - 15);
      out[i] = s ? -v : v;
    }
    return out;
  }
  async function load(base, onProgress) {
    const g = await (await fetch(base + "graph.json")).json();
    onProgress && onProgress(0.15);
    const res = await fetch(base + "weights.wasm");
    const total = +res.headers.get("content-length") || 0;
    let buf;
    if (res.body && total) {
      const reader = res.body.getReader(); const chunks = []; let got = 0;
      for (;;) { const { done, value } = await reader.read(); if (done) break; chunks.push(value); got += value.length; onProgress && onProgress(0.15 + 0.75 * got / total); }
      buf = new Uint8Array(got); let o = 0; for (const c of chunks) { buf.set(c, o); o += c.length; }
      buf = buf.buffer;
    } else buf = await res.arrayBuffer();
    const W = {};
    for (const [name, t] of Object.entries(g.tensors)) {
      if (t.int) { W[name] = { int: t.int }; continue; }
      const n = t.s.reduce((a, b) => a * b, 1);
      const arr = f16ToF32(new Uint16Array(buf, t.o, n));
      W[name] = tf.tensor(arr, t.s.length ? t.s : [1], "float32");
    }
    onProgress && onProgress(1);
    return new LeafNet(g, W);
  }
  class LeafNet {
    constructor(g, W) { this.g = g; this.W = W; }
    run(x) { // x: [1,224,224,3] float in 0..1
      const g = this.g, W = this.W;
      return tf.tidy(() => {
        const env = { [g.input]: x };
        const get = (k) => (k in env ? env[k] : W[k]);
        for (const n of g.nodes) {
          const a = n.a, i0 = get(n.in[0]);
          let y;
          switch (n.op) {
            case "Conv": {
              const w = W[n.in[1]], b = n.in[2] ? W[n.in[2]] : null;
              const s = a.strides || [1, 1], p = a.pads || [0, 0, 0, 0], grp = a.group || 1;
              let inp = (p[0] || p[1] || p[2] || p[3]) ? tf.pad(i0, [[0, 0], [p[0], p[2]], [p[1], p[3]], [0, 0]]) : i0;
              y = grp === 1 ? tf.conv2d(inp, w, s, "valid") : tf.depthwiseConv2d(inp, w, s, "valid");
              if (b) y = tf.add(y, b);
              break;
            }
            case "Relu": y = tf.relu(i0); break;
            case "HardSigmoid": y = tf.clipByValue(tf.add(tf.mul(i0, a.alpha ?? 0.2), a.beta ?? 0.5), 0, 1); break;
            case "HardSwish": y = tf.mul(i0, tf.clipByValue(tf.add(tf.mul(i0, 1 / 6), 0.5), 0, 1)); break;
            case "Sigmoid": y = tf.sigmoid(i0); break;
            case "Clip": { const lo = n.in[1] ? get(n.in[1]) : null, hi = n.in[2] ? get(n.in[2]) : null;
              y = tf.clipByValue(i0, lo ? lo.dataSync()[0] : -Infinity, hi ? hi.dataSync()[0] : Infinity); break; }
            case "Add": y = tf.add(i0, get(n.in[1])); break;
            case "Sub": y = tf.sub(i0, get(n.in[1])); break;
            case "Mul": y = tf.mul(i0, get(n.in[1])); break;
            case "Div": y = tf.div(i0, get(n.in[1])); break;
            case "GlobalAveragePool": y = tf.mean(i0, [1, 2], true); break;
            case "ReduceMean": y = tf.mean(i0, [1, 2], (a.keepdims ?? 1) === 1); break;
            case "Flatten": y = tf.reshape(i0, [i0.shape[0], -1]); break;
            case "Gemm": { y = tf.matMul(i0, W[n.in[1]]); if (n.in[2]) y = tf.add(y, W[n.in[2]]); break; }
            case "MatMul": y = tf.matMul(i0, get(n.in[1])); break;
            case "Softmax": y = tf.softmax(i0, -1); break;
            case "Identity": y = i0; break;
            case "Concat": y = tf.concat(n.in.map(get), i0.shape.length === 4 ? [0, 3, 1, 2][(a.axis + 4) % 4] : (a.axis < 0 ? i0.shape.length + a.axis : a.axis)); break;
            default: throw new Error("Unsupported op " + n.op);
          }
          env[n.out[0]] = y;
        }
        return env[g.output];
      });
    }
  }
  window.LeafNetRuntime = { load };
})();
