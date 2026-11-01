"""same vsa tests and k-NN as the sweep, but on one posterior SAMPLE per image instead of the
mean code (the submission used samples). also the decode error of mean vs sampled codes, to
show the sampled codes still carry the image. writes vsa_sampled.json next to metrics.json.

    REVISION_RESULTS=/workspace/results_rev python revision/vsa_sampled.py --datasets cifar10,fashionmnist
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from revision import common  # noqa: E402
from revision import data as D  # noqa: E402
from revision import evals as E  # noqa: E402
from revision import models as M  # noqa: E402
import cnn.models as cm  # noqa: E402


def set_kappa_mode(a):
    cm.CLIFFORD_PER_CIRCLE_KAPPA = a.get("clifford_kappa", "shared") == "per_circle"
    cm.KAPPA_CAP_FIXED = 10.0 if a.get("kappa_cap", "scaled") == "fixed" else None
    cm.CLIFFORD_KAPPA_CAP_FIXED = 10.0 if a.get("clifford_kappa_cap", "fixed") == "fixed" else None


@torch.no_grad()
def decode_l1(model, x, z):
    # mean absolute error per pixel on the [0, 1] scale, as evals.recon_error
    return ((model.decoder(z) - x).abs() / 2).mean().item()


@torch.no_grad()
def run(model, data, dev, seed):
    te = D.GpuImages(data.test.x, dev, data.arch)
    pool = D.GpuImages(data.knn_pool.x, dev, data.arch)
    x = te.get(torch.arange(E.VSA_ITEMS))
    mean = model.mean_code(x)
    if model.prior_name == "N_L2":
        mean = F.normalize(mean, dim=-1)
    torch.manual_seed(seed)
    samp = model.get_flat_latent(x)
    out = {"decode_l1_mean": decode_l1(model, x, mean), "decode_l1_sample": decode_l1(model, x, samp),
           "cos_sample_to_mean": F.cosine_similarity(samp, mean, dim=-1).mean().item()}

    # swap the mean code for a seeded sample so evals.vsa / evals.knn see samples
    model.mean_code = model.get_flat_latent
    torch.manual_seed(seed)
    out["vsa"], _ = E.vsa(model, te, seed)
    torch.manual_seed(seed)
    out["knn"] = E.knn(model, pool, data.knn_pool.y, te, data.test.y, seed)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default="cifar10,fashionmnist")
    p.add_argument("--seeds", default="1-10")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    lo, hi = map(int, a.seeds.split("-"))
    dev = common.device()
    cache = {}
    for ds in a.datasets.split(","):
        for mpath in sorted(glob.glob(os.path.join(common.RESULTS_ROOT, ds, "*", "seed*", "metrics.json"))):
            rd = os.path.dirname(mpath)
            out = os.path.join(rd, "vsa_sampled.json")
            seed = int(os.path.basename(rd).replace("seed", ""))
            if not lo <= seed <= hi or (os.path.exists(out) and not a.force):
                continue
            try:
                if json.load(open(mpath)).get("status") != "done":
                    continue
            except Exception:
                continue
            cfg = json.load(open(os.path.join(rd, "config.json")))["args"]
            set_kappa_mode(cfg)
            if ds not in cache:
                cache[ds] = D.load(ds)
            model = M.build(cfg["prior"], cfg["d"], cache[ds], dev)
            # pod fp32 best.pt is dropped once the fp16 copy is synced; load_state_dict upcasts
            ck = os.path.join(rd, "best.pt")
            if not os.path.exists(ck):
                ck = os.path.join(rd, "best_fp16.pt")
            model.load_state_dict(torch.load(ck, map_location=dev)["state"])
            model.eval()
            common.write_json(out, run(model, cache[ds], dev, seed))
            print(rd, flush=True)


if __name__ == "__main__":
    main()
