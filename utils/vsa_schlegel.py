"""schlegel, neubert & protzel (2022) benchmarks, implemented exactly (sec. 3.1-3.3).
check against their table 2: random hrr at D=510 gives ~99% for 15 bundled vectors and 15 pairs.
every trial function takes a torch.Generator so results are reproducible."""
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .vsa import bind, bundle, hrr_init, normalize_vectors, unbind, unitary_init


# ---------------------------------------------------------------------------
# benchmarks from schlegel, neubert & protzel (2022), "a comparison of vector
# symbolic architectures", artif. intell. rev. 55:4523-4555, sec. 3.1-3.3.
# the only change from the paper is where the atoms come from: items can be
# encoded images instead of random vectors.
# ---------------------------------------------------------------------------

SCHLEGEL_K_RANGE = list(range(2, 51))  # k = 2...50 in sec. 3.1 and 3.3
SCHLEGEL_N_ITEMS = 1000  # item memory size N
SCHLEGEL_N_TRIALS = 10  # "we repeat each experiment 10 times and report means"
SCHLEGEL_DEPTH = 40  # n = 40 sequential bindings in sec. 3.2


def _gen(seed: Optional[int]) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(0 if seed is None else int(seed))
    return g


def bundle_capacity_trials(items: torch.Tensor, k_range, n_trials: int, seed: int = 0) -> Dict:
    """sec. 3.1: pick k items without replacement, bundle them, query the item memory
    for the k most similar vectors (cosine). accuracy = fraction of the k returned
    vectors that were in the bundle."""
    g = _gen(seed)
    unit = normalize_vectors(items.cpu())
    n = len(unit)
    results = {"k": [], "accuracy": [], "std": []}
    for k in k_range:
        accs = []
        for _ in range(n_trials):
            idx = torch.randperm(n, generator=g)[:k]
            b = bundle(unit[idx], normalize=True)
            sims = unit @ normalize_vectors(b)
            top = torch.topk(sims, k).indices
            accs.append(torch.isin(top, idx).float().mean().item())
        results["k"].append(k)
        results["accuracy"].append(float(np.mean(accs)))
        results["std"].append(float(np.std(accs)))
    return results


def role_filler_trials(items: torch.Tensor, k_range, n_trials: int, unbind_method: str = "inv",
                       seed: int = 0) -> Dict:
    """sec. 3.3: pick 2k items, bind them into k role-filler pairs, bundle the pairs.
    then try to recover all 2k items: unbind each role (expect its filler) and each
    filler (expect its role), clean up by nearest neighbour in the item memory.
    accuracy = correctly recovered / 2k."""
    g = _gen(seed)
    unit = normalize_vectors(items.cpu())
    n = len(unit)
    results = {"k": [], "accuracy": [], "std": []}
    for k in k_range:
        accs = []
        for _ in range(n_trials):
            idx = torch.randperm(n, generator=g)[: 2 * k]
            roles, fillers = unit[idx[:k]], unit[idx[k:]]
            r = bundle(bind(roles, fillers), normalize=True)
            keys = torch.cat([roles, fillers])
            want = torch.cat([idx[k:], idx[:k]])
            rec = unbind(r.unsqueeze(0).expand(2 * k, -1), keys, method=unbind_method)
            got = torch.argmax(normalize_vectors(rec) @ unit.T, dim=-1)
            accs.append((got == want).float().mean().item())
        results["k"].append(k)
        results["accuracy"].append(float(np.mean(accs)))
        results["std"].append(float(np.std(accs)))
    return results


def symbolic_role_filler_trials(items: torch.Tensor, k_range, n_trials: int, unbind_method: str = "inv",
                                seed: int = 0) -> Dict:
    """not in schlegel et al.: roles are random unitary vectors (symbols), fillers are
    items (percepts). bind, bundle, unbind each role, clean up over the item memory,
    score the k fillers. this is the variant the first submission used."""
    g = _gen(seed)
    unit = normalize_vectors(items.cpu())
    n, d = unit.shape
    results = {"k": [], "accuracy": [], "std": []}
    for k in k_range:
        accs = []
        for _ in range(n_trials):
            idx = torch.randperm(n, generator=g)[:k]
            torch.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=g)))
            roles = normalize_vectors(unitary_init(k, d))
            r = bundle(bind(roles, unit[idx]), normalize=True)
            rec = unbind(r.unsqueeze(0).expand(k, -1), roles, method=unbind_method)
            got = torch.argmax(normalize_vectors(rec) @ unit.T, dim=-1)
            accs.append((got == idx).float().mean().item())
        results["k"].append(k)
        results["accuracy"].append(float(np.mean(accs)))
        results["std"].append(float(np.std(accs)))
    return results


def unbinding_depth_trials(items: torch.Tensor, n_bind: int, n_trials: int, unbind_method: str = "inv",
                           seed: int = 0) -> Dict:
    """sec. 3.2: bind a vector v with r_1 ... r_n in turn, then unbind r_n ... r_1.
    similarity of v and the result, rescaled from [-1, 1] to [0, 1]. v and the r_i
    are distinct items from the memory. reported for every depth m = 1...n."""
    g = _gen(seed)
    unit = normalize_vectors(items.cpu())
    n = len(unit)
    assert n > n_bind, f"need more than {n_bind} items"
    per_m = [[] for _ in range(n_bind)]
    for _ in range(n_trials):
        idx = torch.randperm(n, generator=g)[: n_bind + 1]
        v, rs = unit[idx[:1]], unit[idx[1:]]
        for m in range(1, n_bind + 1):
            s = v
            for i in range(m):
                s = bind(s, rs[i : i + 1])
            for i in range(m - 1, -1, -1):
                s = unbind(s, rs[i : i + 1], method=unbind_method)
            cos = F.cosine_similarity(s, v, dim=-1).item()
            per_m[m - 1].append((cos + 1) / 2)
    return {
        "m": list(range(1, n_bind + 1)),
        "similarity": [float(np.mean(x)) for x in per_m],
        "std": [float(np.std(x)) for x in per_m],
    }


def evaluate_items(items: torch.Tensor, n_trials: int = SCHLEGEL_N_TRIALS, seed: int = 0,
                   k_range=None, n_bind: int = SCHLEGEL_DEPTH) -> Dict:
    """all benchmarks on one item memory (encoded images or random atoms).
    items are l2-normalised inside every test."""
    k_range = k_range or SCHLEGEL_K_RANGE
    out = {
        "dim": int(items.shape[-1]),
        "n_items": int(items.shape[0]),
        "bundle_capacity": bundle_capacity_trials(items, k_range, n_trials, seed),
        "role_filler": role_filler_trials(items, k_range, n_trials, "inv", seed),
        "symbolic_role_filler": symbolic_role_filler_trials(items, k_range, n_trials, "inv", seed),
    }
    for name, method in [("unbinding_depth_inv", "inv"), ("unbinding_depth_dagger", "†")]:
        out[name] = unbinding_depth_trials(items, n_bind, n_trials, method, seed)
    return out


def reference_results(kind: str, d: int, n_items: int = SCHLEGEL_N_ITEMS, seed: int = 0, **kw) -> Dict:
    """same benchmarks on random atoms: kind in {"hrr", "unitary"}. seed sets the atoms."""
    torch.manual_seed(seed)
    init_fn = {"hrr": hrr_init, "unitary": unitary_init}[kind]
    return evaluate_items(init_fn(n_items, d), seed=seed, **kw)
