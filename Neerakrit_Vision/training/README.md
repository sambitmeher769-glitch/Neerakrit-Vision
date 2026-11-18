# Training scripts

Run in this order (Python 3.10+, `pip install torch timm pillow numpy onnx`):

1. `cache.py`: builds the image list (`ds.json`), generates the synthetic non-leaf images,
   and caches MobileNetV3-Large features for the original and augmented copies of each image.
   It expects the source repos cloned under `data/dl/` (PlantDoc, ImageNet samples)
   and PlantVillage under `pv/raw/color/`, plus ImageNet-pretrained weights in `mnv3l.pth`
   (timm `mobilenetv3_large_100`). Edit the paths at the top if yours differ.
2. `ft.py`: fine-tunes the last blocks plus two heads (38 diseases, leaf / not-leaf)
   for 7 epochs, prints test accuracy, writes `res2.json` and exports `leafnet2.onnx`.
3. `convert.py leafnet2.onnx`: converts the ONNX file into `../model/graph.json`
   and `../model/weights.wasm`, which the web app loads.

On a 2-core CPU the whole run takes about 1.5 hours.
