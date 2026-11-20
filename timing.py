"""clean cost numbers (reviewer 2): each cell timed alone on an idle gpu (from its seed-1 checkpoint when there is one).
seconds per training epoch (same loop as training, first epoch dropped as warm-up), peak training
memory, encode / decode latency. writes <root>/_aggregate/timing.json for tables.py.

    python timing.py --datasets cifar10,fashionmnist --epochs 3 --repeats 5
"""
import argparse
import os
import time

import torch

from clifford_vae import common, data as D, evals as E, models as M
from clifford_vae.train import DEFAULTS, GRAD_CLIP, WEIGHT_DECAY, load_weights
from run_experiments import CELLS


def time_cell(model, data, dev, epochs, seed, repeats=1):
    """repeats independent timings of one cell. each repeat: one warm-up epoch, then `epochs` timed epochs,
    then the latency test. returns the mean of every quantity and, under <name>_runs, the per-repeat values."""
    hp = DEFAULTS[data.arch]
    tr = D.GpuImages(data.train, dev, data.arch)
    te = D.GpuImages(data.test, dev, data.arch)
    runs = []
    for r in range(repeats):
        gen = torch.Generator().manual_seed(seed + r)
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
        one = {"s_per_epoch": sum(times[1:]) / epochs, "peak_mem_mb": torch.cuda.max_memory_allocated() / 2**20}
        model.eval()
        one.update(E.latency(model, te, dev))
        runs.append(one)
    out = {k: sum(x[k] for x in runs) / len(runs) for k in runs[0]}
    out["n_params"] = M.n_params(model)
    if repeats > 1:
        out.update({k + "_runs": [x[k] for x in runs] for k in runs[0]})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", default="cifar10,fashionmnist")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--repeats", type=int, default=1, help="independent timings per cell (error bars in the table)")
    p.add_argument("--root", default=common.RESULTS)
    p.add_argument("--interleave", action="store_true", help="time the cells round-robin, one repeat per cell per round, so slow "
                   "drift in the host's load falls on every cell alike")
    p.add_argument("--dims", default=None, help="comma list: only these latent dimensions")
    p.add_argument("--priors", default=None, help="comma list: only these priors")
    a = p.parse_args()
    assert torch.cuda.is_available(), "timing needs a gpu"
    dev = torch.device("cuda")
    path = os.path.join(a.root, "_aggregate", "timing.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = common.read_json(path, {})
    out["_gpu"] = torch.cuda.get_device_name(0)
    if a.interleave:
        for ds in a.datasets.split(","):
            data = D.load(ds)
            dims, priors = CELLS[ds]
            dims = [d for d in dims if not a.dims or str(d) in a.dims.split(",")]
            priors = [q for q in priors if not a.priors or q in a.priors.split(",")]
            models = {f"{ds}/{pr}-d{d}": M.build(pr, d, data, dev) for d in dims for pr in priors}
            time_cell(next(iter(models.values())), data, dev, 1, 1, 1)  # unrecorded start-up
            runs = {k: [] for k in models}
            for r in range(a.repeats):
                for k, model in models.items():
                    one = time_cell(model, data, dev, a.epochs, 1 + r, 1)
                    runs[k].append(one)
                    cell = {q: sum(x[q] for x in runs[k]) / len(runs[k]) for q in one}
                    cell.update({q + "_runs": [x[q] for x in runs[k]] for q in one if q != "n_params"})
                    out[k] = cell
                common.write_json(path, out)
                print("round", r + 1, {k.split("/")[1]: round(runs[k][-1]["s_per_epoch"], 2) for k in models}, flush=True)
        return
    warmed = False
    for ds in a.datasets.split(","):
        data = D.load(ds)
        dims, priors = CELLS[ds]
        dims = [d for d in dims if not a.dims or str(d) in a.dims.split(",")]
        priors = [q for q in priors if not a.priors or q in a.priors.split(",")]
        for d in dims:
            for pr in priors:
                rd = common.run_dir(ds, pr, d, 1, root=a.root)
                ck = common.find_ckpt(rd, "best")
                # cost does not depend on the weights: without a checkpoint the cell is timed from a fresh model
                model = M.build(pr, d, data, dev, kappa_shared=M.kappa_shared_in(ck) if ck else False)
                if ck:
                    load_weights(model, ck)
                if not warmed:  # the process's first epochs carry one-off start-up cost: run them unrecorded
                    time_cell(model, data, dev, 1, 1, 1)
                    warmed = True
                out[f"{ds}/{pr}-d{d}"] = time_cell(model, data, dev, a.epochs, 1, a.repeats)
                r = out[f"{ds}/{pr}-d{d}"]
                print(ds, pr, d, {k: round(v, 2) for k, v in r.items() if not k.endswith("_runs")},
                      "epoch times", [round(x, 2) for x in r.get("s_per_epoch_runs", [])], flush=True)
                common.write_json(path, out)
                del model


if __name__ == "__main__":
    main()
