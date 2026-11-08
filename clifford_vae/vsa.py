# hrr primitives and the vsa benchmarks of schlegel, neubert & protzel (2022, sec. 3.1-3.3), with
# encoded images as the item memory. every test l2-normalises the items, averages 10 trials per
# point and takes a seed, so results are reproducible.
#
#   bundle_capacity        3.1  bundle k items, top-k nearest in memory, fraction that were bundled
#   unbinding_depth        3.2  bind v with r_1..r_m, unbind r_m..r_1, (cos(v, result) + 1) / 2
#   role_filler            3.3  k image roles, k image fillers, never sharing a class / identity
#   role_filler_random_roles    random unitary roles, image fillers (the submission's variant)
#   role_filler_random_fillers  image roles, random unitary fillers (clean-up over the atoms)
#   role_filler_cross           image roles from one dataset, fillers from another (offline,
#                               figures.py; needs two runs' saved codes)
#   two_bundle                  the submission's bundle test (appendix only, chance 0.5)
import math

import numpy as np
import torch
import torch.nn.functional as F

K_RANGE = list(range(2, 51))
N_ITEMS = 1000
N_TRIALS = 10
DEPTH = 40


def hrr_init(n, d):
    return torch.randn(n, d) / math.sqrt(d)


def unitary_init(n, d, gen=None):
    """n random real vectors of length d with |F{x}_k| = 1 for every k."""
    m = (d - 1) // 2
    phi = (torch.rand(n, m, generator=gen) * 2 - 1) * math.pi
    f = torch.ones(n, d, dtype=torch.complex64)
    f[:, 1:m + 1] = torch.exp(1j * phi)
    f[:, d - m:] = torch.flip(torch.conj(f[:, 1:m + 1]), (-1,))
    return torch.fft.ifft(f, dim=-1).real


def normalize(x):
    return F.normalize(x, dim=-1)


def bind(a, b):
    # torch 2.7's cpu (mkl) fft rejects expanded, stride-0 inputs
    a, b = a.contiguous(), b.contiguous()
    return torch.fft.ifft(torch.fft.fft(a, dim=-1) * torch.fft.fft(b, dim=-1), dim=-1).real


def involution(a):
    """b* = (b_0, b_{n-1}, ..., b_1): the approximate inverse, exact when b is unitary."""
    return torch.cat([a[..., :1], torch.flip(a[..., 1:], (-1,))], -1)


def unbind(ab, b, method="inv"):
    """inv: ab (*) b*.   dagger: deconvolution, F^-1{F(ab) / F(b)} (exact inverse, unstable
    when b has small fourier coefficients)."""
    if method == "inv":
        return bind(ab, involution(b))
    f = torch.fft.fft(ab.contiguous(), dim=-1) / (torch.fft.fft(b.contiguous(), dim=-1) + 1e-12)
    return torch.fft.ifft(f, dim=-1).real


def bundle(v):
    """sum, scaled by 1/sqrt(k)."""
    return v.sum(0) / math.sqrt(len(v))


def _g(seed):
    return torch.Generator().manual_seed(int(seed))


def _curve(k_range, fn, n_trials):
    out = {"k": [], "accuracy": [], "std": [], "trials": []}
    for k in k_range:
        acc = [fn(k) for _ in range(n_trials)]
        if any(a is None for a in acc):
            break
        out["k"].append(k)
        out["accuracy"].append(float(np.mean(acc)))
        out["std"].append(float(np.std(acc)))
        out["trials"].append([float(a) for a in acc])
    return out


def _cleanup(rec, memory):
    return torch.argmax(normalize(rec) @ memory.T, dim=-1)


def bundle_capacity(items, seed=0, k_range=K_RANGE, n_trials=N_TRIALS):
    g, u = _g(seed), normalize(items)

    def trial(k):
        idx = torch.randperm(len(u), generator=g)[:k]
        top = torch.topk(u @ normalize(bundle(u[idx])), k).indices
        return torch.isin(top, idx).float().mean().item()

    return _curve(k_range, trial, n_trials)


def unbinding_depth(items, seed=0, method="inv", depth=DEPTH, n_trials=N_TRIALS):
    g, u = _g(seed), normalize(items)
    per_m = [[] for _ in range(depth)]
    for _ in range(n_trials):
        idx = torch.randperm(len(u), generator=g)[:depth + 1]
        v, rs = u[idx[:1]], u[idx[1:]]
        for m in range(1, depth + 1):
            s = v
            for i in range(m):
                s = bind(s, rs[i:i + 1])
            for i in range(m - 1, -1, -1):
                s = unbind(s, rs[i:i + 1], method)
            per_m[m - 1].append((F.cosine_similarity(s, v, dim=-1).item() + 1) / 2)
    return {"m": list(range(1, depth + 1)), "similarity": [float(np.mean(x)) for x in per_m],
            "std": [float(np.std(x)) for x in per_m], "trials": per_m}


def _disjoint_pick(groups, k, g):
    """k role items and k filler items, distinct, with role groups and filler groups disjoint:
    the group set is split in half at random each trial. None if a half runs out of items."""
    gs = torch.unique(groups)
    perm = gs[torch.randperm(len(gs), generator=g)]
    ra, fa = perm[: len(gs) // 2], perm[len(gs) // 2:]
    ri = torch.nonzero(torch.isin(groups, ra)).squeeze(1)
    fi = torch.nonzero(torch.isin(groups, fa)).squeeze(1)
    if len(ri) < k or len(fi) < k:
        return None
    return ri[torch.randperm(len(ri), generator=g)[:k]], fi[torch.randperm(len(fi), generator=g)[:k]]


def role_filler(items, groups=None, seed=0, k_range=K_RANGE, n_trials=N_TRIALS):
    """schlegel 3.3 with image roles and image fillers that never share a group (class or
    identity). bind k pairs, bundle, then unbind every role (expect its filler) and every filler
    (expect its role); clean up by nearest neighbour over the whole item memory.
    accuracy = correctly recovered / 2k. groups=None: plain random picks (random atoms)."""
    g, u = _g(seed), normalize(items)
    groups = torch.arange(len(u)) if groups is None else torch.as_tensor(groups)

    def trial(k):
        pick = _disjoint_pick(groups, k, g)
        if pick is None:
            return None
        ri, fi = pick
        r = bundle(bind(u[ri], u[fi]))
        keys, want = torch.cat([u[ri], u[fi]]), torch.cat([fi, ri])
        got = _cleanup(unbind(r.expand(2 * k, -1), keys), u)
        return (got == want).float().mean().item()

    return _curve(k_range, trial, n_trials)


def role_filler_random_roles(items, seed=0, k_range=K_RANGE, n_trials=N_TRIALS):
    """roles are random unitary vectors (symbols), fillers are images; unbind each role, clean
    up over the item memory, score the k fillers."""
    g, u = _g(seed), normalize(items)

    def trial(k):
        idx = torch.randperm(len(u), generator=g)[:k]
        roles = normalize(unitary_init(k, u.shape[1], g))
        got = _cleanup(unbind(bundle(bind(roles, u[idx])).expand(k, -1), roles), u)
        return (got == idx).float().mean().item()

    return _curve(k_range, trial, n_trials)


def role_filler_random_fillers(items, seed=0, k_range=K_RANGE, n_trials=N_TRIALS, n_atoms=N_ITEMS):
    """roles are images, fillers random unitary atoms (a clean-up memory of n_atoms symbols);
    unbind each image role, clean up over the atoms, score the k fillers. isolates how good the
    image codes are as keys, with no semantic overlap between roles and fillers possible."""
    g, u = _g(seed), normalize(items)
    atoms = normalize(unitary_init(n_atoms, u.shape[1], g))

    def trial(k):
        ri = torch.randperm(len(u), generator=g)[:k]
        fi = torch.randperm(n_atoms, generator=g)[:k]
        got = _cleanup(unbind(bundle(bind(u[ri], atoms[fi])).expand(k, -1), u[ri]), atoms)
        return (got == fi).float().mean().item()

    return _curve(k_range, trial, n_trials)


def role_filler_cross(roles, fillers, seed=0, k_range=K_RANGE, n_trials=N_TRIALS):
    """roles from one item memory (e.g. fashionmnist codes), fillers from another (cifar-10
    codes of a model with the same prior and d). clean-up over the union of both memories;
    score both directions as in role_filler."""
    g, ur, uf = _g(seed), normalize(roles), normalize(fillers)
    mem = torch.cat([ur, uf])

    def trial(k):
        ri = torch.randperm(len(ur), generator=g)[:k]
        fi = torch.randperm(len(uf), generator=g)[:k]
        r = bundle(bind(ur[ri], uf[fi]))
        keys, want = torch.cat([ur[ri], uf[fi]]), torch.cat([fi + len(ur), ri])
        got = _cleanup(unbind(r.expand(2 * k, -1), keys), mem)
        return (got == want).float().mean().item()

    return _curve(k_range, trial, n_trials)


def two_bundle(items, seed=0, k_range=K_RANGE, n_trials=N_TRIALS):
    """the submission's test: bundle k items (C1) and k others (C2); fraction of the first k
    closer to C1 than to C2. chance 0.5."""
    g, u = _g(seed), normalize(items)

    def trial(k):
        if 2 * k > len(u):
            return None
        idx = torch.randperm(len(u), generator=g)[:2 * k]
        x = u[idx[:k]]
        return (x @ normalize(u[idx[:k]].sum(0)) > x @ normalize(u[idx[k:]].sum(0))).float().mean().item()

    return _curve(k_range, trial, n_trials)


def evaluate(items, groups=None, seed=0):
    """every in-run test on one item memory (encoded images or random atoms)."""
    items = items.cpu().float()
    return {
        "dim": int(items.shape[1]),
        "n_items": int(items.shape[0]),
        "bundle_capacity": bundle_capacity(items, seed),
        "unbinding_depth_inv": unbinding_depth(items, seed, "inv"),
        "unbinding_depth_dagger": unbinding_depth(items, seed, "dagger"),
        "role_filler": role_filler(items, groups, seed),
        "role_filler_random_roles": role_filler_random_roles(items, seed),
        "role_filler_random_fillers": role_filler_random_fillers(items, seed),
        "two_bundle": two_bundle(items, seed),
    }


def reference(dim, seed=0, n=N_ITEMS):
    """the random baseline: a memory of n random unitary vectors at the item dimension (for
    clifford with d angles that is 2d)."""
    return evaluate(unitary_init(n, dim, _g(seed + 10_000)), None, seed)
