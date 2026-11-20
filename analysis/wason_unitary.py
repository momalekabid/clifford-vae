# wason card task with random unitary concept vectors in place of image codes, at every clifford code length 2d.
# two variants: structural symbols as in every other curve (unit-length gaussian), or unitary as well.
#   python analysis/wason_unitary.py runs/_aggregate/wason_reference_unitary.json   (from the repo root)
import json, os, sys
sys.path.insert(0, os.getcwd())
import numpy as np, torch
import wason as W
from clifford_vae import vsa as V
LENGTHS = [4, 10, 20, 40, 80, 256, 512, 1024, 2048, 4096, 8192]
def unitary(n, D, seed):
    g = torch.Generator().manual_seed(seed)
    return V.unitary_init(n, D, g).double().numpy()
def run_unitary_symbols(codes, seed):
    # the same as wason.run, with unitary structural symbols
    rng = np.random.default_rng(seed); D = codes.shape[1]
    codes = codes / np.linalg.norm(codes, axis=1, keepdims=True); res = []
    for t in range(W.N_TRIALS):
        s = unitary(5, D, 1000 + t); s = s / np.linalg.norm(s, axis=1, keepdims=True)
        sym = dict(zip(("ANT", "RELATION", "IMPLIES", "CONS", "NOT"), s))
        pick = rng.choice(10, 6, replace=False)
        vecs = [codes[rng.choice(np.nonzero(np.arange(1000) % 10 == c)[0])] for c in pick]
        res.append(W.trial(vecs, sym)["transform"])
    v = np.array(res); return {"transform": float(v[:, 0].mean()), "transform_trials": len(v)}
out = {"gaussian_symbols": {}, "unitary_symbols": {}}
path = sys.argv[1]
for D in LENGTHS:
    codes = unitary(1000, D, 0)
    a = W.run(codes, np.arange(1000) % 10, "random", 0, mlp=False)
    b = run_unitary_symbols(codes, 0)
    out["gaussian_symbols"][str(D)] = {k: v for k, v in a.items()}
    out["unitary_symbols"][str(D)] = b
    print(D, "unitary concepts, gaussian symbols:", round(a["transform"], 2), "| all unitary:", round(b["transform"], 2), flush=True)
    json.dump(out, open(path, "w"), indent=1)
