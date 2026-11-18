# Neerakrit Vision: see-then-spray prototype

## Run it
- Windows: double-click `start.bat`
- Mac / Linux: run `./start.sh`
- Or open a terminal in this folder, run `python -m http.server 8000`, then go to http://localhost:8000

Use Chrome or Edge and press "Start camera". Double-clicking index.html will not work,
because browsers block the app from reading its own data and model files over file://.
Everything except the web fonts is in this folder, so it runs offline once started.

For a phone, upload the whole folder to any HTTPS host (GitHub Pages, Netlify Drop).
Browsers only allow the camera over https or localhost.

## Folder map
```
index.html            page layout; links css/style.css and js/boot.js
css/style.css         all styles and animations
js/boot.js            loads data/*.json, then runtime.js, fpv.js, app.js in order
js/runtime.js         runs the leaf model in the browser with TensorFlow.js
js/fpv.js             3D drone spray mission (Three.js)
js/app.js             camera, scanning, results, field log, disease library
js/vendor/            TensorFlow.js 4.22.0 and Three.js r128 (local copies)
model/graph.json      model layers (MobileNetV3, 38 diseases + leaf / not-leaf check)
model/weights.wasm    model weights, fp16 (named .wasm only so web hosts serve it)
data/diseases.json    per-class symptoms, spray decision, product, cultural control
data/sources.json     extension-service references behind diseases.json
data/samples.json     list of the built-in test photos
data/model-meta.json  class list and test accuracy shown in "How it works"
assets/samples/       the 29 test photos (field, lab and non-leaf)
dataset/              what the model was trained and tested on (see dataset/README.md)
training/             Python scripts that built the model
```

## Editing
- Change a fertiliser or spray recommendation: edit `data/diseases.json`, refresh the page.
- Add a test photo: drop a jpg in `assets/samples/` and add a line to `data/samples.json`
  (`truth` is the class name from `dataset/classes.csv`, or null for a non-leaf).
- Retrain: see `training/README.md`. The new model goes into `model/`.
