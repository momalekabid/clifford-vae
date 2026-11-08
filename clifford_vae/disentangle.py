# reviewer 1's compositionality test on 3d shapes: vaes see 30 of the 40 object-hue x shape pairs
# (data.HELD_OUT). on seen vs unseen test pairs we measure
#   recon_mae      reconstruction error (mean code), and the unseen / seen gap
#   attr_acc       hue and shape of the reconstruction both right, by a factor classifier trained
#                  on the full grid (never seen by a vae)
#   probe_acc      linear probe from mean codes to hue / shape, fit on seen pairs, scored on unseen
#   analogy        build the code of an unseen pair from three seen ones that share every other
#                  factor: hrr  unbind(a (*) c, b)  and  vector  a - b + c, for
#                  a = (hue h, shape s'), b = (h', s'), c = (h', s)  ->  target (h, s).
#                  decoded and scored with the factor classifier.
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

from . import models as M
from . import vsa as V
from .data import CACHE, HELD_OUT, SHAPES3D_SIZES, GpuImages, Split

HUE, SHAPE = 2, 4
CLS_PATH = os.path.join(CACHE, "shapes3d_factor_net.pt")


class FactorNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(3, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(),
                               nn.MaxPool2d(2), nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.heads = nn.ModuleList([nn.Linear(128, n) for n in SHAPES3D_SIZES])

    def forward(self, x):
        h = self.f(x)
        return [head(h) for head in self.heads]


@torch.enable_grad()  # called from the no_grad test
def factor_net(all_arrays, dev, n=50000, epochs=5):
    """trained once on random images of the full grid (held-out pairs included), cached."""
    net = FactorNet().to(dev)
    if os.path.exists(CLS_PATH):
        net.load_state_dict(torch.load(CLS_PATH, map_location=dev)["state"])
        return net.eval()
    rng = np.random.RandomState(0)
    idx = rng.choice(len(all_arrays["x"]), n + 5000, replace=False)
    tr = GpuImages(Split(all_arrays["x"][idx[:n]]), dev, "cnn")
    lab = torch.from_numpy(all_arrays["labels"][idx]).to(dev)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    g = torch.Generator().manual_seed(0)
    for _ in range(epochs):
        for b in tr.batches(256, g, shuffle=True):
            loss = sum(F.cross_entropy(o, lab[b, j]) for j, o in enumerate(net(tr.get(b))))
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()
    te = GpuImages(Split(all_arrays["x"][idx[n:]]), dev, "cnn")
    with torch.no_grad():
        out = net(te.get(torch.arange(te.n, device=dev)))
    acc = [(o.argmax(1) == lab[n:, j]).float().mean().item() for j, o in enumerate(out)]
    torch.save({"state": net.state_dict(), "test_acc": acc}, CLS_PATH)
    return net


@torch.no_grad()
def _hue_shape_ok(net, x, lab):
    out = net(x.clamp(-1, 1))
    return ((out[HUE].argmax(1) == lab[:, HUE]) & (out[SHAPE].argmax(1) == lab[:, SHAPE])).float()


@torch.no_grad()
def composition(model, data, dev, seed, n_analogy=2000):
    net = factor_net(data.meta["all"], dev)
    out = {}
    codes = {}
    for name, sp in (("seen", data.test), ("unseen", data.meta["unseen"])):
        im = GpuImages(sp, dev, "cnn")
        lab = torch.from_numpy(sp.extra["labels"]).to(dev)
        mae, ok, ok_in = 0.0, 0.0, 0.0
        for b in im.batches(500):
            x = im.get(b)
            xr = model.decoder(M.mean_code(model, x))
            mae += ((xr - x).abs() / 2).mean((1, 2, 3)).sum().item()
            ok += _hue_shape_ok(net, xr, lab[b]).sum().item()
            ok_in += _hue_shape_ok(net, x, lab[b]).sum().item()
        out[name] = {"recon_mae": mae / im.n, "attr_acc": ok / im.n, "classifier_acc_on_inputs": ok_in / im.n}
        codes[name] = (M.mean_codes(model, im).cpu().numpy(), sp.extra["labels"])
    out["recon_gap"] = out["unseen"]["recon_mae"] / out["seen"]["recon_mae"]

    # linear probe: seen -> unseen
    tr = GpuImages(Split(data.train.x[:10000]), dev, "cnn")
    ztr, ltr = M.mean_codes(model, tr).cpu().numpy(), data.train.extra["labels"][:10000]
    for f, name in ((HUE, "hue"), (SHAPE, "shape")):
        clf = LogisticRegression(max_iter=2000).fit(ztr, ltr[:, f])
        out[f"probe_acc_{name}_seen"] = float(clf.score(codes["seen"][0], codes["seen"][1][:, f]))
        out[f"probe_acc_{name}_unseen"] = float(clf.score(codes["unseen"][0], codes["unseen"][1][:, f]))

    # analogies: every term a seen pair, same floor / wall / scale / orientation as the target
    allx, alll = data.meta["all"]["x"], data.meta["all"]["labels"]
    held = set(HELD_OUT)
    rng = np.random.RandomState(seed)
    ulab = data.meta["unseen"].extra["labels"]
    picks = rng.choice(len(ulab), min(n_analogy, len(ulab)), replace=False)
    tri, tgt = [], []
    for i in picks:
        t = ulab[i].copy()
        h, s = t[HUE], t[SHAPE]
        opts = [(h2, s2) for h2 in range(10) for s2 in range(4)
                if h2 != h and s2 != s and not {(h, s2), (h2, s2), (h2, s)} & held]
        h2, s2 = opts[rng.randint(len(opts))]
        terms = []
        for hh, ss in ((h, s2), (h2, s2), (h2, s)):
            v = t.copy()
            v[HUE], v[SHAPE] = hh, ss
            terms.append(np.ravel_multi_index(v, SHAPES3D_SIZES))
        tri.append(terms)
        tgt.append(np.ravel_multi_index(t, SHAPES3D_SIZES))
    tri, tgt = np.array(tri), np.array(tgt)
    assert (alll[tgt][:, HUE] == ulab[picks][:, HUE]).all()  # grid order check
    im = lambda idx: torch.from_numpy(allx[idx]).to(dev).float() / 127.5 - 1  # noqa: E731
    lab = torch.from_numpy(alll[tgt]).to(dev)
    res = {"hrr": [], "vector": [], "hrr_mae": [], "vector_mae": []}
    for j in range(0, len(tgt), 500):
        za, zb, zc = (M.mean_code(model, im(tri[j:j + 500, q])) for q in range(3))
        x = im(tgt[j:j + 500])
        for name, z in (("hrr", V.unbind(V.bind(za, zc), zb)), ("vector", za - zb + zc)):
            xr = model.decoder(z)
            res[name] += _hue_shape_ok(net, xr, lab[j:j + 500]).tolist()
            res[name + "_mae"] += ((xr - x).abs() / 2).mean((1, 2, 3)).tolist()
    out["analogy"] = {k: float(np.mean(v)) for k, v in res.items()}
    return out

