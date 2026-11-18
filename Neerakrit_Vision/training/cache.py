# Stage 1: build dataset lists and cache backbone activations after blocks[4] (112x14x14) as fp16 memmap.
import os, random, json, time, re
import numpy as np, torch, timm
from PIL import Image, ImageEnhance, ImageFilter, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.set_num_threads(2)
random.seed(0); np.random.seed(0); torch.manual_seed(0)
PV = "pv/raw/color"; D = "/home/claude/data/dl"
classes = sorted(os.listdir(PV)); assert len(classes) == 38
PD_MAP = {
 "Apple Scab Leaf": "Apple___Apple_scab", "Apple leaf": "Apple___healthy", "Apple rust leaf": "Apple___Cedar_apple_rust",
 "Bell_pepper leaf": "Pepper,_bell___healthy", "Bell_pepper leaf spot": "Pepper,_bell___Bacterial_spot", "Blueberry leaf": "Blueberry___healthy",
 "Cherry leaf": "Cherry_(including_sour)___healthy", "Corn Gray leaf spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
 "Corn leaf blight": "Corn_(maize)___Northern_Leaf_Blight", "Corn rust leaf": "Corn_(maize)___Common_rust_", "Peach leaf": "Peach___healthy",
 "Potato leaf early blight": "Potato___Early_blight", "Potato leaf late blight": "Potato___Late_blight", "Raspberry leaf": "Raspberry___healthy",
 "Soyabean leaf": "Soybean___healthy", "Squash Powdery mildew leaf": "Squash___Powdery_mildew", "Strawberry leaf": "Strawberry___healthy",
 "Tomato Early blight leaf": "Tomato___Early_blight", "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot", "Tomato leaf": "Tomato___healthy",
 "Tomato leaf bacterial spot": "Tomato___Bacterial_spot", "Tomato leaf late blight": "Tomato___Late_blight", "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
 "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato mold leaf": "Tomato___Leaf_Mold",
 "Tomato two spotted spider mites leaf": "Tomato___Spider_mites Two-spotted_spider_mite", "grape leaf": "Grape___healthy", "grape leaf black rot": "Grape___Black_rot"}
ci = {c: i for i, c in enumerate(classes)}
def ok(p):
    try:
        with Image.open(p) as im: im.verify()
        return True
    except Exception: return False

items = []  # (path, cls or -1, leaf 1/0, split, source)
# PlantVillage: 40 per class held out
for c in classes:
    fs = sorted(os.listdir(os.path.join(PV, c))); random.shuffle(fs)
    for k, f in enumerate(fs):
        items.append((os.path.join(PV, c, f), ci[c], 1, "test" if k < 40 else "train", "pv"))
# PlantDoc (web-scraped field photos), official split
for sp in ["train", "test"]:
    base = os.path.join(D, "PlantDoc-Dataset", sp)
    for folder in sorted(os.listdir(base)):
        if folder not in PD_MAP: print("skip", folder); continue
        for f in sorted(os.listdir(os.path.join(base, folder))):
            items.append((os.path.join(base, folder, f), ci[PD_MAP[folder]], 1, sp, "pd"))
# Negatives: ImageNet samples minus plant-like classes
PLANT = re.compile(r"cabbage|broccoli|cardoon|artichoke|rapeseed|daisy|slipper|corn|_ear|pot\b|flowerpot|greenhouse|hay|plant|leaf|tree|flower|grass|fern|herb|vegetable|lettuce|shrub|vine|moss|bush|weed|lily|orchid|cactus|palm|ivy|basil|spinach|kale|maize|wheat|rice|garden|forest|hedge|bean|pea\b|zucchini|cucumber|squash|lemon|orange|apple|strawberry|banana|fig|pineapple|jackfruit|custard|pomegranate|acorn|buckeye|hip", re.I)
negs = []
for d in ["imagenet-sample-images", "imagen"]:
    for root, _, fs in os.walk(os.path.join(D, d)):
        for f in fs:
            if f.lower().endswith((".jpg", ".jpeg", ".png")) and not PLANT.search(f): negs.append(os.path.join(root, f))
negs.sort(); random.shuffle(negs)
for k, p in enumerate(negs): items.append((p, -1, 0, "test" if k < 300 else "train", "neg"))
# Synthetic negatives: soil, bare backgrounds, flat colours, skin-like, blurred noise
os.makedirs(os.path.join(D, "synth"), exist_ok=True)
for k in range(500):
    p = os.path.join(D, "synth", f"n{k:03d}.jpg")
    if not os.path.exists(p):
        kind = k % 5; rng = np.random.default_rng(k)
        if kind == 0:  # soil
            base = rng.uniform(60, 150); a = np.stack([base, base * rng.uniform(.7, .85), base * rng.uniform(.45, .65)], -1) * np.ones((224, 224, 3))
            a += rng.normal(0, rng.uniform(8, 30), (224, 224, 1)); im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(rng.uniform(0, 1.5)))
        elif kind == 1:  # flat / gradient surfaces incl. green walls
            c1, c2 = rng.uniform(0, 255, 3), rng.uniform(0, 255, 3); t = np.linspace(0, 1, 224)[:, None, None]
            im = Image.fromarray(np.clip(c1 * (1 - t) + c2 * t + rng.normal(0, 4, (224, 224, 3)), 0, 255).astype(np.uint8))
        elif kind == 2:  # skin-like
            c = np.array([rng.uniform(150, 235), rng.uniform(105, 175), rng.uniform(80, 140)]); a = c * np.ones((224, 224, 3)) + rng.normal(0, 6, (224, 224, 3))
            im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(2))
        elif kind == 3:  # colour noise blobs
            a = rng.uniform(0, 255, (14, 14, 3)); im = Image.fromarray(a.astype(np.uint8)).resize((224, 224), Image.BICUBIC)
        else:  # PlantVillage-style empty background
            g = rng.uniform(60, 200); a = np.array([g, g * rng.uniform(.95, 1.05), g * rng.uniform(1, 1.1)]) * np.ones((224, 224, 3)) + rng.normal(0, 5, (224, 224, 3))
            im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1))
        im.save(p, quality=90)
    items.append((p, -1, 0, "test" if k % 10 == 0 else "train", "synth"))

items = [it for it in items if ok(it[0])]
from collections import Counter
print(Counter((s, src) for _, _, _, s, src in items))
json.dump({"classes": classes, "items": items}, open("ds.json", "w"))

m = timm.create_model("mobilenetv3_large_100", pretrained=False)
m.load_state_dict(torch.load("mnv3l.pth", map_location="cpu")); m.eval()
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
def load(p, aug):
    im = Image.open(p); im.draft("RGB", (448, 448)); im = im.convert("RGB")
    if aug:
        w, h = im.size; s = random.uniform(0.6, 1.0)
        cw, chh = int(w * s), int(h * s); x = random.randint(0, w - cw); y = random.randint(0, h - chh)
        im = im.crop((x, y, x + cw, y + chh))
        if random.random() < 0.5: im = im.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5: im = im.transpose(Image.FLIP_TOP_BOTTOM)
        im = im.rotate(random.uniform(-30, 30) + random.choice([0, 90, 180, 270]), expand=False)
        im = ImageEnhance.Brightness(im).enhance(random.uniform(0.6, 1.35))
        im = ImageEnhance.Color(im).enhance(random.uniform(0.6, 1.35))
        im = ImageEnhance.Contrast(im).enhance(random.uniform(0.75, 1.25))
        if random.random() < 0.3: im = im.filter(ImageFilter.GaussianBlur(random.uniform(0.5, 2)))
    else:
        w, h = im.size; s = min(w, h); im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    im = im.resize((224, 224), Image.BILINEAR)
    return ((np.asarray(im, np.float32) / 255. - MEAN) / STD).transpose(2, 0, 1)

# copies: test 1 (clean); train: pv 2 (clean+aug), pd 4 (clean+3 aug), neg 3, synth 2
rows = []
for idx, (p, c, l, sp, src) in enumerate(items):
    if sp == "test": rows.append((idx, False))
    else:
        n = {"pv": 2, "pd": 4, "neg": 3, "synth": 2}[src]
        rows += [(idx, k > 0) for k in range(n)]
print("rows", len(rows))
N = len(rows)
X = np.lib.format.open_memmap("cache_x.npy", mode="w+", dtype=np.float16, shape=(N, 112, 14, 14))
meta = np.array([[i, int(a)] for i, a in rows], np.int32); np.save("cache_rows.npy", meta)
t = time.time(); bs = 64
with torch.no_grad():
    for i in range(0, N, bs):
        x = torch.from_numpy(np.stack([load(items[j][0], a) for j, a in rows[i:i + bs]]))
        f = m.conv_stem(x); f = m.bn1(f)
        for b in m.blocks[:5]: f = b(f)
        X[i:i + len(x)] = f.numpy().astype(np.float16)
        if i % 3200 == 0: print(i, N, round(time.time() - t), "s", flush=True)
X.flush(); print("done", round(time.time() - t))
