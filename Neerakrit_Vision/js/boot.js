// Boot: load the JSON data files, expose them to the app, then load the app scripts in order.
(async function () {
  const DATA = {
    KB: "data/diseases.json",          // disease knowledge base: symptoms, spray decision, product
    SOURCES: "data/sources.json",      // extension-service references cited in the knowledge base
    SAMPLES: "data/samples.json",      // test photos shown under "Test photos"
    MODEL_META: "data/model-meta.json" // class list and test accuracy of the trained model
  };
  const SCRIPTS = ["js/runtime.js", "js/fpv.js", "js/app.js"];

  try {
    await Promise.all(Object.entries(DATA).map(async ([name, url]) => {
      const r = await fetch(url);
      if (!r.ok) throw new Error(url + " (" + r.status + ")");
      window[name] = await r.json();
    }));
  } catch (e) {
    document.body.insertAdjacentHTML("afterbegin",
      '<div style="position:fixed;inset:0;z-index:9999;display:grid;place-items:center;background:#0b120d;color:#e9f0e6;font:16px/1.5 system-ui;padding:24px;text-align:center">' +
      "<div><b>Could not load the app data.</b><br>Run it through a local server instead of opening index.html directly:<br>" +
      "<code>python -m http.server 8000</code> then open http://localhost:8000<br><small>" + String(e.message || e) + "</small></div></div>");
    return;
  }

  for (const src of SCRIPTS) {
    await new Promise((ok, fail) => {
      const s = document.createElement("script");
      s.src = src; s.onload = ok; s.onerror = () => fail(new Error(src));
      document.body.appendChild(s);
    });
  }
})();
