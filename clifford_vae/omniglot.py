# omniglot (lake et al. 2015), self-contained: the data, then two tests that run after training
# (train_omniglot.py; evals.run_all calls evaluate() through data.meta["evals"]).
#   fewshot      karunaratne et al. 2021's few-shot episodes with their read-out. m-way n-shot
#                episodes on the evaluation alphabets (characters never seen in training), 32
#                queries, 1000 episodes. every support code is kept as a row of a key memory, and a
#                query takes the class with the largest summed cosine (sum-argmax), on real codes
#                (_cos) and on their sign (_cos_sign, the bipolar memory of their hardware).
# training uses 16 of the 20 drawings of every background character (omniglot_held picks the other 4);
# the split is kept from the runs that also scored character x drawer analogies, a test since removed.
import os

import numpy as np
import torch
import torch.nn.functional as F

from . import models as M
from .data import DATA, SPLIT_SEED, Data, GpuImages, Split, _cached, _val_split

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
    meta["bg"]: every background image. y = groups = character;
    extra["labels"] = character, alphabet, drawer."""
    a = omniglot_arrays()
    mk = lambda x, l: Split(x, l[:, 0], l[:, 0], {"labels": l})  # noqa: E731
    held = omniglot_held(a["lbg"])
    train, val = _val_split(mk(a["xbg"][~held], a["lbg"][~held]))
    ev = np.random.RandomState(SPLIT_SEED).permutation(len(a["xev"]))  # the vsa item memory spans many characters
    return Data("omniglot", "cnn", 1, 32, train, val, mk(a["xev"][ev], a["lev"][ev]), knn_pool=None,
                meta={"unseen": mk(a["xbg"][held], a["lbg"][held]), "bg": mk(a["xbg"], a["lbg"]),
                      "evals": evaluate, "skip": {"fid"}, "likelihood": "bernoulli"})


# ---------------------------------------------------------------- tests


FEWSHOT = [(5, 1), (5, 5), (20, 1), (20, 5), (100, 5)]


@torch.no_grad()
def fewshot(model, data, dev, seed, problems=FEWSHOT, n_episodes=1000, n_query=32):
    Z = F.normalize(M.mean_codes(model, GpuImages(data.test, dev, "cnn")).cpu(), dim=-1)
    Zs = F.normalize(torch.sign(Z), dim=-1)
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
        cos, sign = [], []
        for s, qi, qy in eps:
            for A, res in ((Z, cos), (Zs, sign)):  # support is ordered by class
                res.append((A[qi] @ A[s].T).view(len(qi), m, n).sum(-1).argmax(1).numpy() == qy)
        out[f"{m}way{n}shot_cos"] = float(np.mean(np.concatenate(cos)))
        out[f"{m}way{n}shot_cos_sign"] = float(np.mean(np.concatenate(sign)))
    return out


def evaluate(model, data, dev, seed):
    return {"fewshot": fewshot(model, data, dev, seed)}
