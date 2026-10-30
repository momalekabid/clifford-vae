"""recompute k-NN (cosine for every prior, euclidean logged) for finished runs from best.pt.
runs whose metrics.json already has knn metric "cosine" and euc_acc_1000 are skipped.

    REVISION_RESULTS=/workspace/results_rev python revision/recompute_knn.py
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from revision import common  # noqa: E402
from revision import data as D  # noqa: E402
from revision import evals as E  # noqa: E402
from revision import models as M  # noqa: E402
import cnn.models as cm  # noqa: E402
import mnist.mlp_vae as mm  # noqa: E402


def main():
    dev = common.device()
    cache = {}
    for mpath in sorted(glob.glob(os.path.join(common.RESULTS_ROOT, "*", "*", "seed*", "metrics.json"))):
        m = json.load(open(mpath))
        if m.get("status") != "done" or "knn" not in m:
            continue
        if m["knn"].get("metric") == "cosine" and "euc_acc_1000" in m["knn"]:
            continue
        rd = os.path.dirname(mpath)
        a = json.load(open(os.path.join(rd, "config.json")))["args"]
        for mod in (cm, mm):
            mod.CLIFFORD_PER_CIRCLE_KAPPA = a.get("clifford_kappa", "shared") == "per_circle"
            mod.KAPPA_CAP_FIXED = 10.0 if a.get("kappa_cap", "scaled") == "fixed" else None
            mod.CLIFFORD_KAPPA_CAP_FIXED = 10.0 if a.get("clifford_kappa_cap", "fixed") == "fixed" else None
        if a["dataset"] not in cache:
            cache[a["dataset"]] = D.load(a["dataset"])
        data = cache[a["dataset"]]
        model = M.build(a["prior"], a["d"], data, dev)
        model.load_state_dict(torch.load(os.path.join(rd, "best.pt"), map_location=dev)["state"])
        model.eval()
        pool = data.knn_pool
        pool_imgs = D.GpuImages(pool.x, dev, data.arch, binary=pool.extra.get("binary"))
        te = D.GpuImages(data.test.x, dev, data.arch, binary=data.test.extra.get("binary"))
        m["knn_old"] = m["knn"]
        m["knn"] = E.knn(model, pool_imgs, pool.y, te, data.test.y, a["seed"])
        common.write_json(mpath, m)
        print(rd, m["knn_old"]["acc_100"], "->", m["knn"]["acc_100"], flush=True)


if __name__ == "__main__":
    main()
