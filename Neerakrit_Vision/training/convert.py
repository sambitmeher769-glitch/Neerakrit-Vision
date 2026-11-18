import onnx, numpy as np, json, sys
from onnx import numpy_helper, helper
m = onnx.load(sys.argv[1] if len(sys.argv) > 1 else "leafnet.onnx")
g = m.graph
inits = {i.name: numpy_helper.to_array(i) for i in g.initializer}
# Constant nodes -> initializers
nodes = []
for n in g.node:
    if n.op_type == "Constant":
        for a in n.attribute:
            if a.name == "value":
                inits[n.output[0]] = numpy_helper.to_array(a.t)
        continue
    nodes.append(n)
print(sorted(set(n.op_type for n in nodes)))
conv_w = {}
for n in nodes:
    if n.op_type == "Conv":
        conv_w[n.input[1]] = n
blob = bytearray(); tensors = {}
def put(name, arr, kind):
    global blob
    a = np.ascontiguousarray(arr.astype(np.float16))
    off = len(blob); blob += a.tobytes()
    while len(blob) % 4: blob += b"\0"
    tensors[name] = {"o": off, "s": list(a.shape), "k": kind}
used = set(i for n in nodes for i in n.input)
for name, arr in inits.items():
    if name not in used: continue
    if arr.dtype in (np.int64, np.int32):
        tensors[name] = {"int": arr.tolist(), "s": list(arr.shape)}; continue
    if name in conv_w:
        n = conv_w[name]; grp = next((a.i for a in n.attribute if a.name == "group"), 1)
        O, I, H, W = arr.shape
        if grp == 1:
            put(name, arr.transpose(2, 3, 1, 0), "conv")          # HWIO
        elif grp == O and I == 1:
            put(name, arr.transpose(2, 3, 0, 1), "dw")            # H W C 1
        else:
            raise SystemExit("grouped conv unsupported")
    elif arr.ndim == 4 and arr.shape[0] == 1 and arr.shape[2] == 1 and arr.shape[3] == 1:
        put(name, arr.transpose(0, 2, 3, 1), "c")                 # 1,1,1,C
    else:
        gem = [n for n in nodes if n.op_type == "Gemm" and n.input[1] == name]
        if gem:
            tb = next((a.i for a in gem[0].attribute if a.name == "transB"), 0)
            put(name, arr.T if tb else arr, "mat")                 # in,out
        else:
            put(name, arr, "t")
out_nodes = []
for n in nodes:
    attrs = {}
    for a in n.attribute:
        v = helper.get_attribute_value(a)
        if isinstance(v, bytes): v = v.decode()
        if isinstance(v, (list, tuple)): v = list(v)
        attrs[a.name] = v
    out_nodes.append({"op": n.op_type, "in": list(n.input), "out": list(n.output), "a": attrs})
json.dump({"input": g.input[0].name, "output": g.output[0].name, "nodes": out_nodes, "tensors": tensors},
          open("../model/graph.json", "w"), separators=(",", ":"))
open("../model/weights.wasm", "wb").write(bytes(blob))
print(len(out_nodes), "nodes", len(blob) / 1e6, "MB")
