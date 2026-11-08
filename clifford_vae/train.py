# one training loop for every dataset and prior, then every evaluation on the val-selected
# checkpoint. the per-dataset scripts (train_*.py) only set defaults and call main().
#
# beta schedules (--schedule):
#   cyclical  cnn default. linear warm-up 0 -> beta_max over `warmup` epochs, then cycles of
#             `cycle` epochs that ramp beta_min -> beta_max over the first half and hold
#             beta_max for the second (fu et al. 2019). checkpoint selection and patience only
#             look at the beta_max plateau, where the val elbo is the objective being trained.
#   constant  mnist default (davidson et al. 2018 setup). warm-up, then beta_max.
#
# two checkpoints per run, named after it (common.ckpt_path): <run>_half.pt (epoch epochs//2, with
# optimiser state: a crashed run resumes from it) and <run>_best.pt (val-selected, every test runs
# on it). each stores its tag, epoch and val elbo; after the evals both are stored as fp16 weights.
import argparse
import os
import time

import numpy as np
import torch

from . import common
from . import data as D
from . import evals as E
from . import models as M

GRAD_CLIP = 1.0
WEIGHT_DECAY = 0.01

DEFAULTS = {
    "mlp": dict(optim="adam", lr=1e-3, batch_size=128, epochs=500, patience=50, warmup=100,
                schedule="constant", beta_min=1.0, beta_max=1.0, cycle=100),
    "cnn": dict(optim="adamw", lr=3e-4, batch_size=256, epochs=500, patience=100, warmup=100,
                schedule="cyclical", beta_min=0.0, beta_max=1.0, cycle=100),
}


def beta_at(epoch, a):
    if epoch < a.warmup:
        return a.beta_max * (epoch + 1) / a.warmup
    if a.schedule == "constant":
        return a.beta_max
    pos = ((epoch - a.warmup) % a.cycle) / a.cycle
    return a.beta_min + (a.beta_max - a.beta_min) * min(1.0, 2.0 * pos)


def parser(dataset, dims, priors, overrides=None):
    p = argparse.ArgumentParser(description=f"train + evaluate one {dataset} vae")
    p.add_argument("--prior", required=True, choices=priors)
    p.add_argument("--d", type=int, required=True, help=f"latent dimension, paper: {dims}")
    p.add_argument("--seed", type=int, required=True)
    arch = "mlp" if dataset == "mnist" else "cnn"
    for k, v in {**DEFAULTS[arch], **(overrides or {})}.items():
        p.add_argument(f"--{k}", type=type(v), default=v)
    p.add_argument("--kappa_cap", type=float, default=None, help="clifford: max kappa (default 10, inf = no cap)")
    p.add_argument("--kappa_floor", type=float, default=None, help="clifford: min kappa (default 0.01 cnn, 0.03 mlp)")
    p.add_argument("--kappa_shared", action="store_true", help="clifford: one kappa per image (the submission's setup)")
    p.add_argument("--log_every", type=int, default=10, help="steps between curve rows")
    p.add_argument("--skip", default="", help="comma list of evals to skip: knn,vsa,fid,raw")
    p.add_argument("--root", default=None, help="results root (default runs/ or $CVAE_RESULTS)")
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--force", action="store_true", help="re-run a finished run")
    return p


def _save(path, **obj):
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)


def _to_fp16(path):
    if os.path.exists(path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        ck = {"state": {k: v.half() if v.is_floating_point() else v for k, v in ck["state"].items()},
              **{k: ck.get(k) for k in ("tag", "run", "epoch", "val")}, "dtype": "float16"}
        _save(path, **ck)


def load_weights(model, path):
    ck = torch.load(path, map_location=next(model.parameters()).device, weights_only=False)
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ck["state"].items()})
    return ck


def main(dataset, dims, priors, overrides=None, argv=None):
    a = parser(dataset, dims, priors, overrides).parse_args(argv)
    rd = common.run_dir(dataset, a.prior, a.d, a.seed, a.beta_max, a.root)
    os.makedirs(rd, exist_ok=True)
    mpath = os.path.join(rd, "metrics.json")
    if common.get(common.read_json(mpath, {}), "status") == "done" and not a.force:
        print(f"skip (done): {rd}")
        return

    gen = common.seed_everything(a.seed)
    dev = common.device()
    data = D.load(dataset)
    model = M.build(a.prior, a.d, data, dev, a.kappa_cap, a.kappa_floor, a.kappa_shared)
    opt = (torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=WEIGHT_DECAY) if a.optim == "adamw"
           else torch.optim.Adam(model.parameters(), lr=a.lr))

    state = {"epoch": 0, "step": 0, "best": float("inf"), "best_epoch": -1, "bad": 0, "epoch_times": []}
    half, best = common.ckpt_path(rd, "half"), common.ckpt_path(rd, "best")
    run = os.path.relpath(rd, a.root or common.RESULTS)
    if os.path.exists(half) and not a.force:
        ck = torch.load(half, map_location=dev, weights_only=False)
        if "opt" in ck:
            model.load_state_dict(ck["state"])
            opt.load_state_dict(ck["opt"])
            state = ck["loop"]
            gen.set_state(ck["gen"])
            torch.set_rng_state(ck["torch_rng"])
            print(f"resumed from {os.path.basename(half)} at epoch {state['epoch']}")

    cfg = {"dataset": dataset, "args": vars(a), "run_dir": rd, "n_params": M.n_params(model), "grad_clip": GRAD_CLIP,
           "weight_decay": WEIGHT_DECAY if a.optim == "adamw" else 0.0, "split_seed": D.SPLIT_SEED,
           "n_train": len(data.train.x), "n_val": len(data.val.x), "n_test": len(data.test.x), **common.provenance()}
    common.write_json(os.path.join(rd, "config.json"), cfg)
    common.write_json(mpath, {"status": "running"})

    wb = None
    if not a.no_wandb:
        import wandb

        cell = os.path.basename(os.path.dirname(rd))
        wb = wandb.init(project=common.WANDB_PROJECT, name=f"{dataset}-{cell}-s{a.seed}", group=f"{dataset}-{cell}",
                        config=cfg, dir=rd, reinit=True, tags=[dataset, a.prior, f"d{a.d}"])

    tr = D.GpuImages(data.train, dev, data.arch)
    va = D.GpuImages(data.val, dev, data.arch)
    steps = common.CurveLogger(os.path.join(rd, "train_log.csv"),
                               ["step", "epoch", "loss", "recon", "kl", "grad_norm", "lr", "beta"], wb)
    epochs = common.CurveLogger(os.path.join(rd, "epoch_log.csv"),
                                ["epoch", "beta", "train_loss", "val_recon", "val_kl", "val_neg_elbo", "best_val",
                                 "on_plateau", "kappa_mean", "kappa_at_cap", "kappa_at_floor", "epoch_time_s"], wb)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    status, t_train = "done", time.time()
    for epoch in range(state["epoch"], a.epochs):
        t0 = time.time()
        beta = beta_at(epoch, a)
        model.train()
        run, nb = 0.0, 0
        for idx in tr.batches(a.batch_size, gen, shuffle=True):
            total, rec, kl = model.loss(tr.get(idx, gen, dynamic=True), beta)
            if not torch.isfinite(total):
                status = "failed_nonfinite_loss"
                break
            opt.zero_grad(set_to_none=True)
            total.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)  # pre-clip norm
            opt.step()
            state["step"] += 1
            run, nb = run + total.item(), nb + 1
            if state["step"] % a.log_every == 0:
                steps.log({"step": state["step"], "epoch": epoch, "loss": total.item(), "recon": rec.item(),
                           "kl": kl.item(), "grad_norm": gn.item(), "lr": opt.param_groups[0]["lr"], "beta": beta},
                          state["step"], "train/")
        if status != "done":
            break

        model.eval()
        v = E.test_elbo(model, va, seed=epoch)
        if not np.isfinite(v["neg_elbo"]):
            status = "failed_nonfinite_val"
            break
        on_plateau = epoch >= a.warmup - 1 and beta >= a.beta_max
        if on_plateau and v["neg_elbo"] < state["best"]:
            state.update(best=v["neg_elbo"], best_epoch=epoch, bad=0)
            _save(best, state=model.state_dict(), tag="best", run=run, epoch=epoch, val=v)
        elif on_plateau and epoch >= a.warmup:
            state["bad"] += 1  # patience only counts on the plateau, after the warm-up
        state["epoch_times"].append(time.time() - t0)
        state["epoch"] = epoch + 1
        with torch.no_grad():
            kap = M.kappa_stats(model, va.get(torch.arange(min(1000, va.n), device=dev)))
        epochs.log({"epoch": epoch, "beta": beta, "train_loss": run / max(nb, 1), "val_recon": v["recon"],
                    "val_kl": v["kl"], "val_neg_elbo": v["neg_elbo"], "best_val": state["best"],
                    "on_plateau": int(on_plateau), "epoch_time_s": state["epoch_times"][-1], **kap},
                   state["step"], "val/")
        steps.flush()
        epochs.flush()
        if epoch + 1 == a.epochs // 2:
            _save(half, state=model.state_dict(), opt=opt.state_dict(), loop=state, gen=gen.get_state(),
                  torch_rng=torch.get_rng_state(), tag="half", run=run, epoch=epoch, val=v)
        if state["bad"] >= a.patience:
            break
    steps.close()
    epochs.close()

    cost = {"train_time_s": time.time() - t_train, "epochs_run": state["epoch"], "best_epoch": state["best_epoch"],
            "mean_epoch_time_s": float(np.mean(state["epoch_times"])) if state["epoch_times"] else None,
            "peak_mem_mb": torch.cuda.max_memory_allocated() / 2**20 if dev.type == "cuda" else None,
            "n_params": M.n_params(model)}
    metrics = {"status": status, "cost": cost,
               "checkpoints": {t: os.path.basename(p) for t, p in (("best", best), ("half", half)) if os.path.exists(p)}}
    if state["best_epoch"] < 0:
        metrics["status"] = status if status != "done" else "failed_no_checkpoint"
    else:
        load_weights(model, best)
        common.seed_everything(a.seed)
        metrics.update(E.run_all(model, data, rd, a.seed, dev, set(filter(None, a.skip.split(",")))))
        for p in (best, half):
            _to_fp16(p)
    metrics["time_end"] = time.strftime("%Y-%m-%d %H:%M:%S")
    common.write_json(mpath, metrics)
    if wb is not None:
        flat = common.flatten_scalars({k: v for k, v in metrics.items() if k != "vsa"}, "test/")
        for name, r in metrics.get("vsa", {}).items():
            if isinstance(r, dict) and "k" in r:
                for k in (10, 20, 30):
                    if k in r["k"]:
                        flat[f"test/vsa/{name}_k{k}"] = r["accuracy"][r["k"].index(k)]
        wb.log(flat, step=state["step"])
        wb.summary.update(flat | {"status": metrics["status"]})
        wb.finish()
    print(f"done: {rd} status={metrics['status']} best_epoch={state['best_epoch']}")
