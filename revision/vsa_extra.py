"""revision extras on saved test codes (no retraining): l2 vs polar (projected) bundles for
bundle capacity / role-filler / random-role role-filler, and binary-tree decoding with standard
vs fourier-space permutations. writes vsa_extra.json next to each run's metrics.json, and
random unitary / hrr references at every item-memory dimension to <root>/_aggregate/.

    REVISION_RESULTS=/workspace/results_rev python revision/vsa_extra.py --datasets cifar10
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from revision import common  # noqa: E402
from utils.vsa import hrr_init, unitary_init  # noqa: E402
from utils.vsa_schlegel import SCHLEGEL_N_ITEMS, evaluate_extra  # noqa: E402

REF_SEEDS = range(3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default="cifar10")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    dims = set()
    for ds in a.datasets.split(","):
        for mpath in sorted(glob.glob(os.path.join(common.RESULTS_ROOT, ds, "*", "seed*", "metrics.json"))):
            rd = os.path.dirname(mpath)
            out = os.path.join(rd, "vsa_extra.json")
            codes = os.path.join(rd, "codes_test1000.pt")
            try:
                done = json.load(open(mpath)).get("status") == "done"
            except Exception:
                done = False
            if not done or not os.path.exists(codes):
                continue
            items = torch.load(codes, map_location="cpu")["codes"].float()
            dims.add(items.shape[-1])
            if os.path.exists(out) and not a.force:
                continue
            seed = int(os.path.basename(rd).replace("seed", ""))
            common.write_json(out, evaluate_extra(items, seed=seed))
            print(rd, flush=True)

    ref_path = os.path.join(common.RESULTS_ROOT, "_aggregate", "vsa_extra_references.json")
    os.makedirs(os.path.dirname(ref_path), exist_ok=True)
    refs = json.load(open(ref_path)) if os.path.exists(ref_path) else {}
    for D in sorted(dims):
        for kind, init in (("unitary", unitary_init), ("hrr", hrr_init)):
            key = f"{kind}-{D}"
            if key in refs:
                continue
            res = []
            for s in REF_SEEDS:
                torch.manual_seed(s)
                res.append(evaluate_extra(init(SCHLEGEL_N_ITEMS, D), seed=s))
            refs[key] = res
            common.write_json(ref_path, refs)
            print("ref", key, flush=True)


if __name__ == "__main__":
    main()
