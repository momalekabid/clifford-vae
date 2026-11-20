# 3d shapes factor analyses on every saved run (no training). per run:
#   pairs    binding and vector analogy for every pair of the six factors (all six factors of the decoding right)
#   offsets  move one factor with one fixed transform per value, by binding and by vector addition
#   cosine   mean cosine between the codes a, b, c, target, binding result, vector result (held-out hue x shape)
# and, for one seed, the images of a cube scene moved along each factor by both operations.
#   python analysis/shapes3d_factors.py <runs root> <out prefix> [--seeds 1,2,3,4,5] [--quick]   (from the repo root)
import glob, json, os, sys, time
sys.path.insert(0, os.getcwd())
import numpy as np, torch
from clifford_vae import data as D, disentangle as DIS, models as M, vsa as V
from clifford_vae.data import HELD_OUT, SHAPES3D_SIZES
from clifford_vae.dists import clifford_embed
from clifford_vae.train import load_weights

ROOT, OUT = sys.argv[1], sys.argv[2]
SEEDS = [int(s) for s in sys.argv[sys.argv.index("--seeds") + 1].split(",")] if "--seeds" in sys.argv else [1, 2, 3, 4, 5]
QUICK = "--quick" in sys.argv
NAMES = ["floor hue", "wall hue", "object hue", "scale", "shape", "orientation"]
HUE, SHAPE = 2, 4
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t0 = time.time()
data = D.load("shapes3d"); allx, alll = data.meta["all"]["x"], data.meta["all"]["labels"]
ulab = data.meta["unseen"].extra["labels"]; heldset = set(HELD_OUT)
held = np.zeros(len(alll), bool)
for h, s in HELD_OUT:
    held |= (alll[:, 2] == h) & (alll[:, 4] == s)
net = DIS.factor_net(data.meta["all"], dev).eval()
im = lambda idx: torch.from_numpy(allx[idx]).to(dev).float() / 127.5 - 1  # noqa: E731
idx_of = lambda lab: np.ravel_multi_index(lab.T, SHAPES3D_SIZES)  # noqa: E731
cosf = lambda a, b: torch.nn.functional.cosine_similarity(a, b, dim=-1)  # noqa: E731

@torch.no_grad()
def predict(x):
    return torch.stack([o.argmax(1) for o in net(x.clamp(-1, 1))], 1).cpu().numpy()

# ---------------------------------------------------------------- fixed evaluation sets (the same for every run)
rng = np.random.RandomState(0)
def analogy_set(f, g, n):
    tri, tgt = [], []
    pool = np.nonzero(held if {f, g} == {HUE, SHAPE} else ~held)[0]
    for i in rng.permutation(pool):
        t = alll[i]
        for _ in range(6):
            f2, g2 = rng.randint(SHAPES3D_SIZES[f]), rng.randint(SHAPES3D_SIZES[g])
            if f2 == t[f] or g2 == t[g]:
                continue
            terms = np.stack([t, t, t]).copy()
            terms[0][g] = g2; terms[1][f], terms[1][g] = f2, g2; terms[2][f] = f2
            j = idx_of(terms)
            if held[j].any():
                continue
            tri.append(j); tgt.append(i); break
        if len(tgt) == n:
            break
    return np.array(tri), np.array(tgt)
SETS = {(f, g): analogy_set(f, g, 100 if QUICK else 600) for f in range(6) for g in range(f + 1, 6)}

def contexts(f, n, seed):
    r = np.random.RandomState(seed); rows = []
    while len(rows) < n:
        t = np.array([r.randint(s) for s in SHAPES3D_SIZES])
        lab = np.repeat(t[None], SHAPES3D_SIZES[f], 0); lab[:, f] = np.arange(SHAPES3D_SIZES[f])
        j = idx_of(lab)
        if not held[j].any():
            rows.append(j)
    return np.array(rows)
OFF_F = [5, 3, 0, 1]   # orientation, scale, floor hue, wall hue (object hue and shape have no fully seen settings)
CTX = {f: (contexts(f, 60 if QUICK else 300, 1), contexts(f, 30 if QUICK else 150, 2)) for f in OFF_F}

def cos_triples(n, seed):
    r = np.random.RandomState(seed); tri, tgt = [], []
    for i in r.choice(len(ulab), n, replace=False):
        t = ulab[i].copy(); h, s = t[HUE], t[SHAPE]
        opts = [(h2, s2) for h2 in range(10) for s2 in range(4) if h2 != h and s2 != s and not {(h, s2), (h2, s2), (h2, s)} & heldset]
        h2, s2 = opts[r.randint(len(opts))]
        terms = []
        for hh, ss in ((h, s2), (h2, s2), (h2, s)):
            v = t.copy(); v[HUE], v[SHAPE] = hh, ss; terms.append(np.ravel_multi_index(v, SHAPES3D_SIZES))
        tri.append(terms); tgt.append(np.ravel_multi_index(t, SHAPES3D_SIZES))
    return np.array(tri), np.array(tgt)
COS_TRI, COS_TGT = cos_triples(200 if QUICK else 2000, 1)
KEYS = ["a", "b", "c", "target", "binding", "vector"]
CUBE = 0
SCENES = [(5, [3, 6, 1, 5, CUBE, 7]), (0, [0, 7, 2, 5, CUBE, 4]), (1, [8, 0, 2, 5, CUBE, 10]), (3, [0, 2, 6, 0, CUBE, 7])]
def scene(f, t):
    lab = np.repeat(np.array(t)[None], SHAPES3D_SIZES[f], 0); lab[:, f] = np.arange(SHAPES3D_SIZES[f])
    return idx_of(lab)

# ---------------------------------------------------------------- per-run helpers
@torch.no_grad()
def codes_by_value(model, ctx):
    """[contexts, values, code] mean codes, and for clifford the mean phases as well."""
    nv = ctx.shape[1]
    Z = torch.stack([M.mean_code(model, im(ctx[:, v])) for v in range(nv)], 1)
    TH = torch.stack([model.encode_mean(im(ctx[:, v])) for v in range(nv)], 1) if model.prior in M.CLIFFORD else None
    return Z, TH

@torch.no_grad()
def transforms(model, Z, TH):
    """one transform per value, relative to value 0. binding: the mean of z_v (*) z_0^* (clifford: its phase, so the
    transform is unitary); vector: the mean of z_v - z_0."""
    vec = (Z - Z[:, :1]).mean(0)
    if TH is not None:
        ph = torch.angle(torch.exp(1j * (TH - TH[:, :1])).mean(0))
        return ph, vec
    T = torch.stack([V.unbind(Z[:, v], Z[:, 0]).mean(0) for v in range(Z.shape[1])])
    return T, vec

@torch.no_grad()
def move(model, z, th, bindT, vec, v, w, sig):
    """the code of an image at value v, moved to value w by binding and by vector addition; both decoded."""
    if th is not None:
        xb = model.decoder(clifford_embed(th + bindT[w] - bindT[v]))
    else:
        zb = V.unbind(V.bind(z, bindT[w].expand_as(z)), bindT[v].expand_as(z))
        xb = M.decode(model, zb, sig)
    xv = M.decode(model, z + vec[w] - vec[v], sig)
    return xb, xv

res = {"names": NAMES, "runs": {}}
images = {}
PRIORS = [("N", "Gaussian"), ("N_L2", "Gaussian (L2)"), ("S", "Power Spherical"), ("C", "Clifford")]
for prior, pname in PRIORS:
    for seed in SEEDS:
        hits = glob.glob(os.path.join(ROOT, "shapes3d", f"{prior}-d128", f"seed{seed}", "*_best.pt")) or \
               [p for p in glob.glob(os.path.join(ROOT, "shapes3d", f"{prior}-d128", "*_best.pt")) if f"seed{seed}_" in p]
        if not hits:
            print("missing", prior, seed, flush=True); continue
        model = M.build(prior, 128, data, dev, kappa_shared=M.kappa_shared_in(hits[0])); load_weights(model, hits[0]); model.eval()
        r = {"pairs": {}, "offsets": {}}
        with torch.no_grad():
            # pairs
            for (f, g), (tri, tgt) in SETS.items():
                xs = [im(tri[:, q]) for q in range(3)]
                za, zb, zc = (M.mean_code(model, x) for x in xs)
                sig = M.code_sigma(model, torch.cat(xs)).view(3, len(xs[0]), -1).mean(0) if prior == "N_L2" else None
                lab = alll[tgt]; o = {}
                for kind, z in (("bind", V.unbind(V.bind(za, zc), zb)), ("vector", za - zb + zc)):
                    o[kind] = float((predict(M.decode(model, z, sig)) == lab).all(1).mean())
                r["pairs"][f"{NAMES[f]} x {NAMES[g]}"] = o
            # offsets
            for f in OFF_F:
                tr, te = CTX[f]; nv = SHAPES3D_SIZES[f]
                bindT, vec = transforms(model, *codes_by_value(model, tr))
                Zte, THte = codes_by_value(model, te)
                rr = np.random.RandomState(3); okb, okv, n = 0, 0, 0
                for _ in range(12):
                    v, w = rr.choice(nv, 2, replace=False)
                    sig = M.code_sigma(model, im(te[:, v]))
                    xb, xv = move(model, Zte[:, v], None if THte is None else THte[:, v], bindT, vec, v, w, sig)
                    lab = alll[te[:, w]]
                    okb += (predict(xb) == lab).all(1).sum(); okv += (predict(xv) == lab).all(1).sum(); n += len(lab)
                r["offsets"][NAMES[f]] = {"bind": float(okb / n), "vector": float(okv / n)}
            # cosine
            za, zb, zc = (M.mean_code(model, im(COS_TRI[:, q])) for q in range(3))
            z = {"a": za, "b": zb, "c": zc, "target": M.mean_code(model, im(COS_TGT)), "binding": V.unbind(V.bind(za, zc), zb), "vector": za - zb + zc}
            r["cosine"] = {f"{p}|{q}": float(cosf(z[p], z[q]).mean()) for i, p in enumerate(KEYS) for q in KEYS[i + 1:]}
            # images, first seed only
            if seed == SEEDS[0]:
                for f, t in SCENES:
                    te = scene(f, t); nv = SHAPES3D_SIZES[f]; src = nv // 2
                    bindT, vec = transforms(model, *codes_by_value(model, CTX[f][0]))
                    z0 = M.mean_code(model, im(te[src:src + 1])); th0 = model.encode_mean(im(te[src:src + 1])) if prior in M.CLIFFORD else None
                    sig = M.code_sigma(model, im(te[src:src + 1]))
                    xb = torch.cat([move(model, z0, th0, bindT, vec, src, w, sig)[0] for w in range(nv)])
                    xv = torch.cat([move(model, z0, th0, bindT, vec, src, w, sig)[1] for w in range(nv)])
                    to8 = lambda x: ((x.clamp(-1, 1) + 1) * 127.5).round().byte().cpu().numpy()  # noqa: E731
                    images[f"{prior}|{NAMES[f]}|bind"], images[f"{prior}|{NAMES[f]}|vector"] = to8(xb), to8(xv)
                    images[f"true|{NAMES[f]}"] = allx[te]; images[f"src|{NAMES[f]}"] = np.array(src)
                    images[f"ok|{prior}|{NAMES[f]}|bind"] = (predict(xb) == alll[te]).all(1); images[f"ok|{prior}|{NAMES[f]}|vector"] = (predict(xv) == alll[te]).all(1)
        res["runs"][f"{prior}|{seed}"] = r
        pm = np.mean([v["bind"] for v in r["pairs"].values()]), np.mean([v["vector"] for v in r["pairs"].values()])
        print(f"{pname} seed {seed}: pairs bind {pm[0]:.3f} vector {pm[1]:.3f} | offsets {({k: (round(v['bind'], 2), round(v['vector'], 2)) for k, v in r['offsets'].items()})} | cos b-t {r['cosine']['target|binding']:.2f} v-t {r['cosine']['target|vector']:.2f} | {round(time.time() - t0)} s", flush=True)
        json.dump(res, open(OUT + ".json", "w"), indent=1)
        np.savez_compressed(OUT + "_images.npz", **images)
print("SUITE_DONE", round(time.time() - t0), "s")
