"""re-run the evaluations of finished runs from their best checkpoint, without training. use it after an
eval changes, or to add the raw dumps to runs that don't have them yet.

    python evaluate.py runs/cifar10/C-d256/seed1
    python evaluate.py runs --only vsa,raw            # every run under a root
    python evaluate.py runs --only vsa --skip_existing
    python evaluate.py runs --include_crashed         # runs whose training finished but whose evals crashed
"""
import argparse
import glob
import os
import time

import torch

from clifford_vae import common, evals as E, models as M
from clifford_vae import data as D
from clifford_vae.train import load_weights

PARTS = ["elbo", "knn", "vsa", "sampled", "fid", "raw"]
KEY = {"elbo": "test_elbo", "sampled": "vsa_sampled"}  # the metrics.json key that marks a part as done


def run_dirs(path):
    if os.path.exists(os.path.join(path, "metrics.json")):
        return [path]
    return sorted(os.path.dirname(m) for m in glob.glob(os.path.join(path, "*", "*", "seed*", "metrics.json")))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path")
    p.add_argument("--only", default=",".join(PARTS), help=f"comma list of {PARTS}")
    p.add_argument("--skip_existing", action="store_true", help="skip a part already in metrics.json / raw/")
    p.add_argument("--include_crashed", action="store_true",
                   help="also runs not marked done that have a best checkpoint (training ended, an eval crashed). "
                        "runs whose epoch log changed in the last 15 min count as still training and are skipped")
    a = p.parse_args()
    only = set(a.only.split(","))
    dev = common.device()
    cache = {}
    for rd in run_dirs(a.path):
        m = common.read_json(os.path.join(rd, "metrics.json"), {})
        cfg = common.read_json(os.path.join(rd, "config.json"), {})
        ck = common.find_ckpt(rd, "best")
        if ck is None or (m.get("status") != "done" and not a.include_crashed):
            continue
        crashed = m.get("status") != "done"
        ep = os.path.join(rd, "epoch_log.csv")
        if crashed and os.path.exists(ep) and time.time() - os.path.getmtime(ep) < 15 * 60:
            continue  # still training
        ds = cfg.get("dataset") or cfg["args"]["dataset"]
        args = cfg["args"]
        if ds not in cache:
            cache = {ds: D.load(ds)}
        data = cache[ds]
        shared = args["kappa_shared"] if "kappa_shared" in args else M.kappa_shared_in(ck)  # older runs: from the checkpoint
        model = M.build(args["prior"], args["d"], data, dev, args.get("kappa_cap"), args.get("kappa_floor"), shared)
        load_weights(model, ck)
        model.eval()
        seed = args["seed"]
        common.seed_everything(seed)
        todo = {x for x in only if not (a.skip_existing and (KEY.get(x, x) in m or (x == "raw" and os.path.isdir(os.path.join(rd, "raw")))))}
        skip = set(PARTS) - todo
        if not todo:
            continue
        print(f"{rd}: {sorted(todo)}", flush=True)
        with torch.no_grad():
            new = E.run_all(model, data, rd, seed, dev, skip)
        if "elbo" not in todo:
            for k in ("test_elbo", "latency"):
                new.pop(k, None)
        m.update(new)
        if crashed:
            m["status"], m["note"] = "done", "evals re-run with evaluate.py after they crashed"
        common.write_json(os.path.join(rd, "metrics.json"), m)


if __name__ == "__main__":
    main()
