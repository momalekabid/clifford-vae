"""clean cost numbers (reviewer 2): each cell timed alone on an idle gpu, from its seed-1 checkpoint.
seconds per training epoch (same loop as training, first epoch dropped as warm-up), peak training
memory, encode / decode latency. writes <root>/_aggregate/timing.json for tables.py.

    python timing.py --datasets cifar10,fashionmnist --epochs 3
"""
import argparse
import os
import time

import torch

from clifford_vae import common, data as D, evals as E, models as M
from clifford_vae.train import DEFAULTS, GRAD_CLIP, WEIGHT_DECAY, load_weights
from run_experiments import CELLS


def time_cell(model, data, dev, epochs, seed):
    hp = DEFAULTS[data.arch]
    gen = torch.Generator().manual_seed(seed)
    tr = D.GpuImages(data.train, dev, data.arch)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=WEIGHT_DECAY)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model.train()
    times = []
    for _ in range(epochs + 1):
        torch.cuda.synchronize()
        t = time.perf_counter()
        for idx in tr.batches(hp["batch_size"], gen, shuffle=True):
            total, _, _ = model.loss(tr.get(idx, gen, dynamic=True), 1.0)
            opt.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
    out = {"s_per_epoch": sum(times[1:]) / epochs, "peak_mem_mb": torch.cuda.max_memory_allocated() / 2**20,
           "n_params": M.n_params(model)}
    model.eval()
    out.update(E.latency(model, D.GpuImages(data.test, dev, data.arch), dev))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", default="cifar10,fashionmnist")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--root", default=common.RESULTS)
    a = p.parse_args()
    assert torch.cuda.is_available(), "timing needs a gpu"
    dev = torch.device("cuda")
    path = os.path.join(a.root, "_aggregate", "timing.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = common.read_json(path, {})
    out["_gpu"] = torch.cuda.get_device_name(0)
    for ds in a.datasets.split(","):
        data = D.load(ds)
        dims, priors = CELLS[ds]
        for d in dims:
            for pr in priors:
                rd = common.run_dir(ds, pr, d, 1, root=a.root)
                ck = common.find_ckpt(rd, "best")
                if ck is None:
                    continue
                model = M.build(pr, d, data, dev, kappa_shared=M.kappa_shared_in(ck))
                load_weights(model, ck)
                out[f"{ds}/{pr}-d{d}"] = time_cell(model, data, dev, a.epochs, 1)
                print(ds, pr, d, {k: round(v, 2) for k, v in out[f"{ds}/{pr}-d{d}"].items()}, flush=True)
                common.write_json(path, out)
                del model


if __name__ == "__main__":
    main()
