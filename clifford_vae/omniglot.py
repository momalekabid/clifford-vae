# omniglot (lake et al. 2015), self-contained: the data, then two tests that run after training
# (train_omniglot.py; evals.run_all calls evaluate() through data.meta["evals"]).
#   fewshot      karunaratne et al. 2021's few-shot episodes, answered by an hrr memory. m-way n-shot
#                episodes on the evaluation alphabets (characters never seen in training), 32
#                queries, 1000 episodes. every class gets a random unitary label vector, each support
#                code is bound to its class label and all m * n pairs are bundled into one vector.
#                a query unbinds the memory with its own code (involution) and is cleaned up against
#                the m labels. for unitary codes (clifford) this is fhrr: binding adds phases.
#   composition  every character misses 4 of its 20 drawers in training. for a held-out
#                (character c, drawer p): a = (c, p'), b = (c', p'), c3 = (c', p), c' from the same
#                alphabet, all three seen in training. hrr unbind(a (*) c3, b) and vector
#                a - b + c3, against copying a (right character, wrong drawer). scored by cos to the
#                target's code, mae of the decoded image to the target, and a character classifier
#                (trained once on every background image) on the decoded image.
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import models as M
from . import vsa as V
from .data import CACHE, DATA, SPLIT_SEED, Data, GpuImages, Split, _cached, _val_split

# ---------------------------------------------------------------- data


def omniglot_arrays():
    """lake et al. 2015: 1623 characters from 50 alphabets, 20 drawers each. background (964
    characters, 30 alphabets) and evaluation (659, 20) sets, 105 -> 32px (area resize, as in
    karunaratne et al. 2021), inverted. labels: character (global id), alphabet, drawer (0-19)."""
    def build():
        from PIL import Image
        from torchvision.datasets import Omniglot

        out, n_char, n_alpha = {}, 0, 0
        for part, bg in (("bg", True), ("ev", False)):
            ds = Omniglot(os.path.join(DATA, "torchvision"), background=bg, download=True)
            xs, lab = [], []
            for a, alpha in enumerate(sorted(os.listdir(ds.target_folder))):
                for char in sorted(os.listdir(os.path.join(ds.target_folder, alpha))):
                    folder = os.path.join(ds.target_folder, alpha, char)
                    for f in sorted(x for x in os.listdir(folder) if x.endswith(".png")):
                        img = Image.open(os.path.join(folder, f)).convert("L").resize((32, 32), Image.BOX)
                        xs.append(255 - np.asarray(img, np.uint8)[None])
                        lab.append((n_char, n_alpha + a, int(f[-6:-4]) - 1))
                    n_char += 1
            n_alpha += a + 1
            out[f"x{part}"], out[f"l{part}"] = np.stack(xs), np.asarray(lab, np.int64)
        return out

    return _cached("omniglot32", build)


OMNIGLOT_FOLD = 5  # character c misses drawer p in training when (c + p) % 5 == 0: 4 of 20 drawers


def omniglot_held(lab):
    return (lab[:, 0] + lab[:, 2]) % OMNIGLOT_FOLD == 0


def load_omniglot():
    """train / val: background images of the seen character x drawer pairs. test: the evaluation
    alphabets (new characters; few-shot and vsa). meta["unseen"]: the held-out background pairs,
    meta["bg"]: every background image (character classifier, analogies). y = groups = character;
    extra["labels"] = character, alphabet, drawer."""
    a = omniglot_arrays()
    mk = lambda x, l: Split(x, l[:, 0], l[:, 0], {"labels": l})  # noqa: E731
    held = omniglot_held(a["lbg"])
    train, val = _val_split(mk(a["xbg"][~held], a["lbg"][~held]))
    ev = np.random.RandomState(SPLIT_SEED).permutation(len(a["xev"]))  # the vsa item memory spans many characters
    return Data("omniglot", "cnn", 1, 32, train, val, mk(a["xev"][ev], a["lev"][ev]), knn_pool=None,
                meta={"unseen": mk(a["xbg"][held], a["lbg"][held]), "bg": mk(a["xbg"], a["lbg"]),
                      "evals": evaluate, "skip": {"fid"}})


# ---------------------------------------------------------------- tests


FEWSHOT = [(5, 1), (5, 5), (20, 1), (20, 5), (100, 5)]
CLS_PATH = os.path.join(CACHE, "omniglot_char_net.pt")


@torch.no_grad()
def fewshot(model, data, dev, seed, problems=FEWSHOT, n_episodes=1000, n_query=32):
    Z = F.normalize(M.mean_codes(model, GpuImages(data.test, dev, "cnn")).cpu(), dim=-1)
    y = data.test.y
    by_class = {c: np.nonzero(y == c)[0] for c in np.unique(y)}
    out = {}
    for m, n in problems:
        classes = np.array([c for c, v in by_class.items() if len(v) > n])  # all 659 on the full set
        if len(classes) < m:
            continue
        rng = np.random.RandomState(seed)
        eps = []
        for _ in range(n_episodes):
            sup, pool = [], []
            for j, c in enumerate(rng.choice(classes, m, replace=False)):
                idx = rng.permutation(by_class[c])
                sup.append(idx[:n])
                pool += [(i, j) for i in idx[n:]]
            q = [pool[i] for i in rng.choice(len(pool), min(n_query, len(pool)), replace=False)]
            eps.append((np.concatenate(sup), np.array([i for i, _ in q]), np.array([j for _, j in q])))
        g = torch.Generator().manual_seed(seed)
        acc = []
        for s, qi, qy in eps:
            labels = V.unitary_init(m, Z.shape[1], g)
            memory = V.bundle(V.bind(labels.repeat_interleave(n, 0), Z[s]))  # support ordered by class
            read = V.unbind(memory.expand(len(qi), -1), Z[qi])
            acc.append((V.normalize(read) @ labels.T).argmax(1).numpy() == qy)
        out[f"{m}way{n}shot"] = float(np.mean(np.concatenate(acc)))
    return out


class CharNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(),
                               nn.MaxPool2d(2), nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                               nn.Linear(256, n_classes))

    def forward(self, x):
        return self.f(x)


@torch.enable_grad()  # called from the no_grad tests
def char_net(bg, dev, epochs=30):
    """character classifier on every background image (held-out pairs included), with random
    shifts, cached. its accuracy on the held-out inputs is reported next to the decoded scores."""
    n_cls = int(bg.y.max()) + 1
    net = CharNet(n_cls).to(dev)
    if os.path.exists(CLS_PATH):
        net.load_state_dict(torch.load(CLS_PATH, map_location=dev)["state"])
        return net.eval()
    im = GpuImages(bg, dev, "cnn")
    lab = torch.from_numpy(bg.y).to(dev)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    g = torch.Generator().manual_seed(0)
    for _ in range(epochs):
        for b in im.batches(256, g, shuffle=True):
            x = im.get(b)
            sh = torch.randint(-2, 3, (2,), generator=g).tolist()
            loss = F.cross_entropy(net(torch.roll(x, sh, (2, 3))), lab[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    torch.save({"state": net.state_dict()}, CLS_PATH)
    return net.eval()


@torch.no_grad()
def composition(model, data, dev, seed, n_analogy=2000):
    bg = data.meta["bg"]
    lab = bg.extra["labels"]
    net = char_net(bg, dev)
    im = lambda idx: torch.from_numpy(bg.x[idx]).to(dev).float() / 127.5 - 1  # noqa: E731
    held = omniglot_held(lab)
    key = {(c, p): i for i, (c, _, p) in enumerate(lab)}
    train = {(c, p) for c, _, p in data.train.extra["labels"]}
    chars_of = {}
    for c, a, _ in lab:
        chars_of.setdefault(a, set()).add(c)

    out = {}
    hidx = np.nonzero(held)[0]
    mae, ok, ok_in = [], [], []
    for j in range(0, len(hidx), 500):
        b = hidx[j:j + 500]
        x, yb = im(b), torch.from_numpy(lab[b, 0]).to(dev)
        xr = model.decoder(M.mean_code(model, x))
        mae += ((xr - x).abs() / 2).mean((1, 2, 3)).tolist()
        ok += (net(xr.clamp(-1, 1)).argmax(1) == yb).float().tolist()
        ok_in += (net(x).argmax(1) == yb).float().tolist()
    out["unseen"] = {"recon_mae": float(np.mean(mae)), "char_acc": float(np.mean(ok)),
                     "classifier_acc_on_inputs": float(np.mean(ok_in))}

    rng = np.random.RandomState(seed)
    tri, tgt = [], []
    for i in rng.permutation(hidx):
        c, a, p = lab[i]
        opts = [(c2, p2) for c2 in chars_of[a] if c2 != c for p2 in range(20)
                if p2 != p and {(c, p2), (c2, p2), (c2, p)} <= train]
        if not opts:
            continue
        c2, p2 = opts[rng.randint(len(opts))]
        tri.append([key[(c, p2)], key[(c2, p2)], key[(c2, p)]])
        tgt.append(i)
        if len(tgt) == n_analogy:
            break
    tri, tgt = np.array(tri), np.array(tgt)
    res = {f"{m}_{s}": [] for m in ("hrr", "vector", "copy") for s in ("cos", "mae", "char_acc")}
    for j in range(0, len(tgt), 500):
        za, zb, zc = (M.mean_code(model, im(tri[j:j + 500, q])) for q in range(3))
        x = im(tgt[j:j + 500])
        zt, yb = M.mean_code(model, x), torch.from_numpy(lab[tgt[j:j + 500], 0]).to(dev)
        for name, z in (("hrr", V.unbind(V.bind(za, zc), zb)), ("vector", za - zb + zc), ("copy", za)):
            xr = model.decoder(z)
            res[name + "_cos"] += F.cosine_similarity(z, zt, dim=-1).tolist()
            res[name + "_mae"] += ((xr - x).abs() / 2).mean((1, 2, 3)).tolist()
            res[name + "_char_acc"] += (net(xr.clamp(-1, 1)).argmax(1) == yb).float().tolist()
    out["analogy"] = {k: float(np.mean(v)) for k, v in res.items()}
    out["analogy"]["n"] = int(len(tgt))
    return out


def evaluate(model, data, dev, seed):
    return {"fewshot": fewshot(model, data, dev, seed), "composition": composition(model, data, dev, seed)}
