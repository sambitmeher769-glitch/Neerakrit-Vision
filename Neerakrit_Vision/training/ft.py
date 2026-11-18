# Stage 2: fine-tune blocks[5:], conv_head and two heads (38 diseases + leaf / not-leaf) on cached activations, then export ONNX.
import json, time, collections, random
import numpy as np, torch, timm, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(2); torch.manual_seed(0); random.seed(0); np.random.seed(0)
ds = json.load(open("ds.json")); classes, items = ds["classes"], ds["items"]
X = np.load("cache_x.npy", mmap_mode="r"); rows = np.load("cache_rows.npy")
cls = np.array([items[i][1] for i, _ in rows]); leaf = np.array([items[i][2] for i, _ in rows], np.float32)
split = np.array([items[i][3] for i, _ in rows]); src = np.array([items[i][4] for i, _ in rows])
tr = np.where(split == "train")[0]; te = np.where(split == "test")[0]
print("train rows", len(tr), "test rows", len(te))

m = timm.create_model("mobilenetv3_large_100", pretrained=False)
m.load_state_dict(torch.load("mnv3l.pth", map_location="cpu"))
for p in m.parameters(): p.requires_grad = False
tail = nn.ModuleList([m.blocks[5], m.blocks[6]])
for mod in [tail, m.conv_head]:
    for p in mod.parameters(): p.requires_grad = True
dis = nn.Sequential(nn.Dropout(0.25), nn.Linear(1280, 38)); gate = nn.Linear(1280, 1)
def body(x):
    for b in tail: x = b(x)
    return m.forward_head(x, pre_logits=True)
params = [{"params": list(tail.parameters()) + list(m.conv_head.parameters()), "lr": 4e-4},
          {"params": list(dis.parameters()) + list(gate.parameters()), "lr": 2e-3}]
opt = torch.optim.AdamW(params, weight_decay=1e-3)
EP = 7; bs = 96; steps = EP * (len(tr) // bs); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[4e-4, 2e-3], total_steps=steps, pct_start=0.15)
w_src = {"pv": 1.0, "pd": 2.0, "neg": 1.0, "synth": 1.0}
sw = torch.tensor([w_src[s] for s in src], dtype=torch.float32)

def evaluate():
    m.eval(); tail.eval(); dis.eval(); P, G = [], []
    with torch.no_grad():
        for i in range(0, len(te), 256):
            idx = te[i:i + 256]; f = body(torch.from_numpy(X[np.sort(idx)].astype(np.float32)))
            P.append(F.softmax(dis(f), 1).numpy()); G.append(torch.sigmoid(gate(f)).squeeze(1).numpy())
    return np.concatenate(P), np.concatenate(G), np.sort(te)

t = time.time()
for ep in range(EP):
    tail.train(); m.conv_head.train(); dis.train(); perm = np.random.permutation(tr); tot = 0
    for i in range(0, len(perm) - bs + 1, bs):
        idx = np.sort(perm[i:i + bs]); x = torch.from_numpy(X[idx].astype(np.float32))
        f = body(x); lg = gate(f).squeeze(1); y = torch.from_numpy(cls[idx]).long(); L = torch.from_numpy(leaf[idx]); w = sw[idx]
        loss = (F.binary_cross_entropy_with_logits(lg, L, reduction="none") * w).mean()
        pos = y >= 0
        if pos.any():
            loss = loss + (F.cross_entropy(dis(f[pos]), y[pos], label_smoothing=0.1, reduction="none") * w[pos]).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item()
    P, G, ix = evaluate(); yy = cls[ix]; s = src[ix]
    for name, mask in [("pv", s == "pv"), ("pd", s == "pd")]:
        print(f"ep{ep} {name} acc {(P[mask].argmax(1) == yy[mask]).mean():.4f}", end="  ")
    print(f"gate acc {((G > 0.5) == (leaf[ix] > 0.5)).mean():.4f}  loss {tot / (len(perm) // bs):.3f}  {round(time.time() - t)}s", flush=True)

P, G, ix = evaluate(); yy = cls[ix]; s = src[ix]; L = leaf[ix]
res = {}
for name in ["pv", "pd"]:
    mk = s == name; pr = P[mk].argmax(1); res[name + "_acc"] = float((pr == yy[mk]).mean()); res[name + "_n"] = int(mk.sum())
    res[name + "_top3"] = float(np.mean([yy[mk][k] in np.argsort(-P[mk][k])[:3] for k in range(mk.sum())]))
mk = s == "pv"; pr = P[mk].argmax(1)
res["per_class"] = {classes[c]: float((pr[yy[mk] == c] == c).mean()) for c in range(38)}
hl = np.array(["healthy" in c for c in classes]); res["binary"] = {f"{a}->{b}": v for (a, b), v in collections.Counter((bool(hl[a]), bool(hl[b])) for a, b in zip(yy[mk], pr)).items()}
neg = L == 0
res["gate"] = {"leaf_n": int((~neg).sum()), "leaf_missed": int((G[~neg] <= 0.5).sum()), "nonleaf_n": int(neg.sum()), "nonleaf_passed": int((G[neg] > 0.5).sum()),
               "pd_missed": int((G[s == "pd"] <= 0.5).sum()), "pd_n": int((s == "pd").sum())}
print(json.dumps({k: v for k, v in res.items() if k != "per_class"}, indent=1))
json.dump(res, open("res2.json", "w"))

# export: input [0,1] NCHW -> 39 values: 38 disease probs + leaf probability
m.classifier = nn.Identity()
class Wrap(nn.Module):
    def __init__(s):
        super().__init__(); s.net = m; s.dis = dis[1]; s.gate = gate
        s.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)); s.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    def forward(s, x):
        f = s.net((x - s.mean) / s.std)
        return torch.cat([torch.softmax(s.dis(f), 1), torch.sigmoid(s.gate(f))], 1)
w = Wrap().eval()
torch.onnx.export(w, torch.rand(1, 3, 224, 224), "leafnet2.onnx", input_names=["x"], output_names=["y"], opset_version=17, dynamic_axes={"x": {0: "n"}, "y": {0: "n"}}, dynamo=False)
torch.save({"dis": dis.state_dict(), "gate": gate.state_dict(), "tail": tail.state_dict(), "conv_head": m.conv_head.state_dict()}, "ft.pt")
print("exported")
