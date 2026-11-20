"""the wason card task (neuromatch neuroai w2d2 tutorial 2, furlong & eliasmith) on the codes of
trained vaes. the notebook's algorithm, with its concept vectors replaced by image codes.

a rule "if X then Y" is ANT (*) X + RELATION (*) IMPLIES + CONS (*) Y, and the cards that can break
it are X + NOT (*) Y. a transform T is learned from two rules with the notebook's update
T <- T - lr (T - answer (*) ~rule), then applied to a third rule over two new concepts. the trial
is solved when the two most similar actions are X and NOT_Y. an mlp trained on the same two rules
is the notebook's foil. structural symbols (ANT, RELATION, IMPLIES, CONS, NOT) are random unit
vectors, as in the notebook's vocabulary; each concept is the mean code of one test image of a
class (3d shapes: of an object hue or of a shape). reads raw/codes.npz, adds "wason" to each
run's metrics.json, and the random-symbol version (the notebook's own setup) to
<root>/_aggregate/wason_reference.json.

    python wason.py                      # every finished run under runs/
    python wason.py --root results --datasets cifar10,fashionmnist --force
"""
import argparse
import glob
import os

import numpy as np
from sklearn.neural_network import MLPRegressor

from clifford_vae import common

N_TRIALS, MLP_TRIALS, NUM_ITERS, LR = 100, 10, 500, 1e-1


def bind(a, b):
    return np.fft.ifft(np.fft.fft(a) * np.fft.fft(b)).real


def inv(a):  # the notebook's ~ (involution)
    return np.concatenate([a[:1], a[1:][::-1]])


def normalized(a):
    return a / np.linalg.norm(a)


def concepts(codes, y, ds):
    """{concept: indices of its images}. classes, or for 3d shapes the 10 object hues and 4 shapes
    (y is the hue x shape pair index)."""
    if ds == "shapes3d":
        return {**{f"hue{h}": np.nonzero(y // 4 == h)[0] for h in range(10)},
                **{f"shape{s}": np.nonzero(y % 4 == s)[0] for s in range(4)}}
    return {f"class{c}": np.nonzero(y == c)[0] for c in np.unique(y)}


def trial(vecs, sym, mlp=False):
    """vecs: 6 concept vectors. rules (0 -> 1) and (2 -> 3) train, (4 -> 5) tests, like the notebook's
    BLUE -> EVEN, ODD -> GREEN and RED -> PRIME."""
    action_space = np.array(list(vecs) + [bind(sym["NOT"], v) for v in vecs])
    rule = lambda x, y: normalized(bind(sym["ANT"], x) + bind(sym["RELATION"], sym["IMPLIES"]) + bind(sym["CONS"], y))  # noqa: E731
    answer = lambda x, y: normalized(x + bind(sym["NOT"], y))  # noqa: E731
    rules = [rule(vecs[0], vecs[1]), rule(vecs[2], vecs[3])]
    answers = [answer(vecs[0], vecs[1]), answer(vecs[2], vecs[3])]
    transform = np.zeros(len(vecs[0]))
    for _ in range(NUM_ITERS):
        for r, a in zip(rules, answers):
            transform -= LR * (transform - bind(a, inv(r)))
            transform = transform / np.linalg.norm(transform)
    new_rule = rule(vecs[4], vecs[5])
    target = {4, 6 + 5}  # X and NOT_Y

    def score(a_hat):
        top2 = set(np.argsort(action_space @ a_hat)[-2:])
        return float(top2 == target), float(4 in top2), float(11 in top2)

    out = {"transform": score(bind(transform, new_rule))}
    if mlp:
        regr = MLPRegressor(random_state=1, hidden_layer_sizes=(1024, 1024), max_iter=1000).fit(np.array(rules), np.array(answers))
        out["mlp"] = score(regr.predict(new_rule[None, :])[0])
    return out


def run(codes, y, ds, seed, mlp):
    rng = np.random.default_rng(seed)
    cs = concepts(codes, y, ds)
    names = sorted(cs)
    codes = codes / np.linalg.norm(codes, axis=1, keepdims=True)
    res = {"transform": [], "mlp": []}
    for t in range(N_TRIALS):
        sym = {k: normalized(rng.standard_normal(codes.shape[1])) for k in ("ANT", "RELATION", "IMPLIES", "CONS", "NOT")}
        pick = rng.choice(names, 6, replace=False)
        vecs = [codes[rng.choice(cs[c])] for c in pick]
        r = trial(vecs, sym, mlp and t < MLP_TRIALS)
        for k, v in r.items():
            res[k].append(v)
    out = {}
    for k, v in res.items():
        if v:
            v = np.array(v)
            out[k], out[f"{k}_ant"], out[f"{k}_cons"] = float(v[:, 0].mean()), float(v[:, 1].mean()), float(v[:, 2].mean())
            out[f"{k}_trials"] = len(v)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=common.RESULTS)
    p.add_argument("--datasets", default="mnist,fashionmnist,cifar10,shapes3d")
    p.add_argument("--force", action="store_true", help="redo runs that already have results")
    a = p.parse_args()
    dims = set()
    for ds in a.datasets.split(","):
        for f in sorted(glob.glob(os.path.join(a.root, ds, "*", "seed*", "raw", "codes.npz"))):
            rd = os.path.dirname(os.path.dirname(f))
            mpath = os.path.join(rd, "metrics.json")
            m = common.read_json(mpath, {})
            if m.get("status") != "done" or ("wason" in m and not a.force):
                continue
            z = np.load(f)
            seed = int(os.path.basename(rd)[4:])
            m["wason"] = run(z["codes"].astype(np.float64), z["y"], ds, seed, mlp=seed == 1)
            common.write_json(mpath, m)
            dims.add(z["codes"].shape[1])
            print(rd, {k: round(v, 3) for k, v in m["wason"].items() if not k.endswith("trials")}, flush=True)
    # the notebook's own setup: every concept a random unit vector, at each code length seen
    path = os.path.join(a.root, "_aggregate", "wason_reference.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ref = common.read_json(path, {})
    for D in sorted(dims):
        if str(D) not in ref:
            rng = np.random.default_rng(0)
            ref[str(D)] = run(rng.standard_normal((1000, D)), np.arange(1000) % 10, "random", 0, mlp=False)
            common.write_json(path, ref)
    print(f"reference: {path}")


if __name__ == "__main__":
    main()
