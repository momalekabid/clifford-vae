"""clean cost numbers for the r2 cost table: one trained checkpoint per cell, run alone on an
idle gpu (sweep timings were measured with 12 runs sharing the gpu). per cell: seconds per
training epoch (same loop as train.py, first epoch dropped as warm-up), peak training memory,
encode / decode latency (evals.latency) and posterior sampling time. nothing is saved back to
the checkpoint. writes <root>/_aggregate/timing.json.

    REVISION_RESULTS=/workspace/results_rev python revision/timing.py --datasets cifar10,fashionmnist
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from revision import common  # noqa: E402
from revision import data as D  # noqa: E402
from revision import evals as E  # noqa: E402
from revision import models as M  # noqa: E402
from revision.train import DEFAULTS, GRAD_CLIP, WEIGHT_DECAY  # noqa: E402
from revision.vsa_sampled import set_kappa_mode  # noqa: E402

CNN_DIMS = [128, 256, 512, 1024, 2048, 4096]
PRIORS = ["N", "N_L2", "S", "C"]


def checkpoint(rd):
    for name in ("best.pt", "best_fp16.pt"):
        if os.path.exists(os.path.join(rd, name)):
            return os.path.join(rd, name)
    return None


def time_cell(model, data, dev, epochs, seed):
    sync = torch.cuda.synchronize
    gen = torch.Generator().manual_seed(seed)
    tr = D.GpuImages(data.train.x, dev, data.arch)
    te = D.GpuImages(data.test.x, dev, data.arch)
    hp = DEFAULTS["cnn"]
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=WEIGHT_DECAY)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model.train()
    times = []
    for _ in range(epochs + 1):
        sync()
        t = time.perf_counter()
        for idx in tr.batches(hp["batch_size"], gen, shuffle=True):
            total, _, _, _ = M.loss(model, tr.get(idx, gen), 1.0)
            opt.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
        sync()
        times.append(time.perf_counter() - t)
    out = {"s_per_epoch": sum(times[1:]) / epochs, "s_per_epoch_all": times,
           "peak_train_mem_mb": torch.cuda.max_memory_allocated() / 2**20}

    model.eval()
    out.update(E.latency(model, te, dev))
    # posterior sampling alone (encoder output fixed), per 1000 images
    with torch.no_grad():
        x = te.get(torch.arange(1000, device=te.x.device))
        mu, params = model.encoder(x)
        model.reparameterize(mu, params)
        sync()
        t = time.perf_counter()
        for _ in range(10):
            model.reparameterize(mu, params)
        sync()
    out["sample_ms_per_1k"] = (time.perf_counter() - t) / 10 * 1000
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default="cifar10,fashionmnist")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--epochs", type=int, default=3)
    a = p.parse_args()
    dev = common.device()
    assert dev.type == "cuda", "timing needs the gpu"
    out_path = os.path.join(common.RESULTS_ROOT, "_aggregate", "timing.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    res = json.load(open(out_path)) if os.path.exists(out_path) else {}
    for ds in a.datasets.split(","):
        data = D.load(ds)
        for d in CNN_DIMS:
            for prior in PRIORS:
                key = f"{ds}/{prior}-d{d}"
                rd = os.path.join(common.RESULTS_ROOT, ds, f"{prior}-d{d}", f"seed{a.seed}")
                ck = checkpoint(rd)
                if key in res or ck is None:
                    if ck is None:
                        print("no checkpoint", rd, flush=True)
                    continue
                set_kappa_mode(json.load(open(os.path.join(rd, "config.json")))["args"])
                model = M.build(prior, d, data, dev)
                model.load_state_dict(torch.load(ck, map_location=dev)["state"])
                res[key] = time_cell(model, data, dev, a.epochs, a.seed)
                res[key]["n_params"] = M.n_params(model)
                res[key]["gpu"] = torch.cuda.get_device_name()
                common.write_json(out_path, res)
                print(key, {k: round(v, 2) for k, v in res[key].items() if isinstance(v, float)}, flush=True)
                del model
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
