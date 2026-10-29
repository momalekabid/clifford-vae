"""one training run for the revision: one dataset, one prior, one latent dim, one seed.

    python revision/train.py --dataset cifar10 --prior C --d 128 --seed 1

everything the paper needs is saved in the run dir (results_rev/<dataset>/<prior>-d<d>/seed<s>/):
config.json (args, seed, git sha, versions), pip_freeze.txt, train_log.csv (per-step),
epoch_log.csv, best.pt (val-selected), codes_test1000.pt (mean codes of the first 1000 test
images), metrics.json (status, every eval, cost). logs to wandb natively.
a finished run (metrics.json with status "done") is skipped, so the queue is resumable.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from revision import common  # noqa: E402  (sets cublas env before torch touches cuda)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from revision import data as D  # noqa: E402
from revision import evals as E  # noqa: E402
from revision import models as M  # noqa: E402

# paper table 6 (hyperparameters). mnist follows davidson et al. 2018.
DEFAULTS = {
    "mlp": dict(optim="adam", lr=1e-3, batch_size=128, epochs=500, patience=50, warmup=100,
                schedule="warmup", beta_min=1.0, beta_max=1.0, cycle=250),
    "cnn": dict(optim="adamw", lr=3e-4, batch_size=256, epochs=500, patience=50, warmup=100,
                schedule="cyclical", beta_min=0.1, beta_max=1.0, cycle=250),
}
GRAD_CLIP = 1.0
WEIGHT_DECAY = 0.01  # adamw default, stated in methods


def beta_at(epoch, a):
    """linear warm-up 0 -> beta_max over `warmup` epochs, then either constant beta_max or a
    triangular cycle that STARTS at beta_max and goes down to beta_min and back (no jump)."""
    if epoch < a.warmup:
        return a.beta_max * (epoch + 1) / a.warmup
    if a.schedule != "cyclical":
        return a.beta_max
    pos = ((epoch - a.warmup) % a.cycle) / a.cycle
    return a.beta_min + (a.beta_max - a.beta_min) * abs(1.0 - 2.0 * pos)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(D.LOADERS))
    p.add_argument("--prior", required=True, choices=M.PRIORS)
    p.add_argument("--d", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    for k in ["optim", "schedule"]:
        p.add_argument(f"--{k}", type=str, default=None)
    for k in ["lr", "beta_min", "beta_max"]:
        p.add_argument(f"--{k}", type=float, default=None)
    for k in ["batch_size", "epochs", "patience", "warmup", "cycle"]:
        p.add_argument(f"--{k}", type=int, default=None)
    p.add_argument("--log_every", type=int, default=10, help="steps between curve rows")
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--skip_evals", type=str, default="", help="comma list, e.g. fid,vsa")
    p.add_argument("--slot_query", action="store_true", help="on hold: slot-query task (off by default)")
    p.add_argument("--force", action="store_true", help="re-run even if finished")
    p.add_argument("--out", type=str, default=None, help="override run dir")
    a = p.parse_args()
    arch = "mlp" if a.dataset == "mnist" else "cnn"
    for k, v in DEFAULTS[arch].items():
        if getattr(a, k) is None:
            setattr(a, k, v)
    return a


def main():
    a = parse()
    rd = a.out or common.run_dir(a.dataset, a.prior, a.d, a.seed, a.beta_max)
    os.makedirs(rd, exist_ok=True)
    mpath = os.path.join(rd, "metrics.json")
    if os.path.exists(mpath) and not a.force:
        import json

        if json.load(open(mpath)).get("status") == "done":
            print(f"skip (done): {rd}")
            return

    gen = common.seed_everything(a.seed)
    dev = common.device()
    data = D.load(a.dataset)
    model = M.build(a.prior, a.d, data, dev)

    cfg = {"args": vars(a), "run_dir": rd, "n_params": M.n_params(model), "grad_clip": GRAD_CLIP,
           "weight_decay": WEIGHT_DECAY if a.optim == "adamw" else 0.0, "split_seed": D.SPLIT_SEED,
           "n_train": len(data.train.x), "n_val": len(data.val.x), "n_test": len(data.test.x),
           **common.provenance()}
    common.write_json(os.path.join(rd, "config.json"), cfg)
    common.pip_freeze(os.path.join(rd, "pip_freeze.txt"))
    common.write_json(mpath, {"status": "running"})

    wb = None
    if not a.no_wandb:
        import wandb

        cell = os.path.basename(os.path.dirname(rd))
        wb = wandb.init(project=common.WANDB_PROJECT, name=f"{a.dataset}-{cell}-s{a.seed}",
                        group=f"{a.dataset}-{cell}", config=cfg, dir=rd, reinit=True,
                        tags=[a.dataset, a.prior, f"d{a.d}", "revision"])

    binar = data.arch == "mlp"
    tr = D.GpuImages(data.train.x, dev, data.arch)
    va = D.GpuImages(data.val.x, dev, data.arch, binary=data.val.extra.get("binary"))
    te = D.GpuImages(data.test.x, dev, data.arch, binary=data.test.extra.get("binary"))

    if a.optim == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=WEIGHT_DECAY)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=a.lr)

    fields = ["step", "epoch", "loss", "recon", "kl", "grad_norm", "lr", "beta"]
    curve = common.CurveLogger(os.path.join(rd, "train_log.csv"), fields, wb)
    efields = ["epoch", "beta", "train_loss", "val_recon", "val_kl", "val_neg_elbo", "best_val",
               "kappa_mean", "kappa_at_cap", "epoch_time_s"]
    ecurve = common.CurveLogger(os.path.join(rd, "epoch_log.csv"), efields, None)

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    best, best_epoch, bad, step, status = float("inf"), -1, 0, 0, "done"
    epoch_times = []
    t_train = time.time()
    for epoch in range(a.epochs):
        t0 = time.time()
        beta = beta_at(epoch, a)
        model.train()
        run_loss, nb = 0.0, 0
        for idx in tr.batches(a.batch_size, gen, shuffle=True):
            x = tr.get(idx, gen, dynamic_binarize=binar)
            total, rec, kl, _ = M.loss(model, x, beta)
            if not torch.isfinite(total):
                status = "failed_nonfinite_loss"
                break
            opt.zero_grad(set_to_none=True)
            total.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)  # pre-clip norm
            opt.step()
            step += 1
            run_loss += total.item()
            nb += 1
            if step % a.log_every == 0:
                curve.log({"step": step, "epoch": epoch, "loss": total.item(), "recon": rec.item(),
                           "kl": kl.item(), "grad_norm": gnorm.item(), "lr": opt.param_groups[0]["lr"],
                           "beta": beta}, step)
        if status != "done":
            break

        model.eval()
        v = E.test_elbo(model, va, 1000, seed=epoch)
        kap = M.concentration_stats(model, va.get(torch.arange(min(1000, va.n), device=dev)))
        if not np.isfinite(v["neg_elbo"]):
            status = "failed_nonfinite_val"
            break
        if v["neg_elbo"] < best:
            best, best_epoch, bad = v["neg_elbo"], epoch, 0
            torch.save({"state": model.state_dict(), "epoch": epoch, "val": v}, os.path.join(rd, "best.pt.tmp"))
            os.replace(os.path.join(rd, "best.pt.tmp"), os.path.join(rd, "best.pt"))
        elif epoch >= a.warmup:
            # patience only counts once beta has finished warming up
            bad += 1
        epoch_times.append(time.time() - t0)
        row = {"epoch": epoch, "beta": beta, "train_loss": run_loss / max(nb, 1), "val_recon": v["recon"],
               "val_kl": v["kl"], "val_neg_elbo": v["neg_elbo"], "best_val": best, "epoch_time_s": epoch_times[-1],
               **kap}
        ecurve.log(row, step)
        if wb is not None:
            wb.log({f"val/{k}": val for k, val in row.items() if k != "epoch"} | {"epoch": epoch}, step=step)
        curve.flush()
        ecurve.flush()
        if bad >= a.patience:
            break
    curve.close()
    ecurve.close()

    cost = {"train_time_s": time.time() - t_train, "epochs_run": len(epoch_times), "best_epoch": best_epoch,
            "mean_epoch_time_s": float(np.mean(epoch_times)) if epoch_times else None,
            "peak_mem_mb": torch.cuda.max_memory_allocated() / 2**20 if dev.type == "cuda" else None,
            "n_params": M.n_params(model)}
    metrics = {"status": status, "cost": cost}
    if best_epoch < 0:
        metrics["status"] = status if status != "done" else "failed_no_checkpoint"
        common.write_json(mpath, metrics)
        if wb is not None:
            wb.summary["status"] = metrics["status"]
            wb.finish()
        return

    # ---------------- evaluation on the val-selected checkpoint, test set touched once
    ck = torch.load(os.path.join(rd, "best.pt"), map_location=dev, weights_only=False)
    model.load_state_dict(ck["state"])
    model.eval()
    skip = set(filter(None, a.skip_evals.split(",")))
    common.seed_everything(a.seed)
    t_ev = time.time()

    def tick(name):
        nonlocal_t = time.time()
        print(f"eval {name}: {nonlocal_t - t_ev:.1f}s since start of evals", flush=True)

    metrics["test_elbo"] = E.test_elbo(model, te, 1000, seed=a.seed)
    metrics["recon_mae"] = E.recon_error(model, te)
    metrics["cost"].update(E.latency(model, te, dev))
    tick("elbo+recon+latency")
    pool = data.knn_pool
    if pool is not None and pool.y is not None and "knn" not in skip and a.dataset != "shapes2d":
        pool_imgs = D.GpuImages(pool.x, dev, data.arch, binary=pool.extra.get("binary"))
        metrics["knn"] = E.knn(model, pool_imgs, pool.y, te, data.test.y, a.seed)
        tick("knn")
        if a.slot_query:
            metrics["slot_query"] = E.slot_query(model, pool_imgs, pool.y, te, data.test.y, a.seed)
    if "vsa" not in skip and a.dataset != "shapes2d":
        metrics["vsa"], items = E.vsa(model, te, a.seed)
        torch.save({"codes": items, "labels": torch.from_numpy(data.test.y[: len(items)])},
                   os.path.join(rd, "codes_test1000.pt"))
        tick("vsa")
    if data.arch == "cnn" and "fid" not in skip and a.dataset != "shapes2d":
        metrics["fid"] = E.generation_fid(model, te, a.seed)
        tick("fid")
    if a.dataset == "shapes2d":
        un = D.GpuImages(data.extra_tests["unseen"].x, dev, "cnn")
        metrics["composition"] = E.composition(model, te, un, data.test.extra["labels"],
                                               data.extra_tests["unseen"].extra["labels"], dev)
    metrics["time_end"] = time.strftime("%Y-%m-%d %H:%M:%S")
    common.write_json(mpath, metrics)
    if wb is not None:
        flat = common.flatten_scalars({k: v for k, v in metrics.items() if k != "vsa"}, "test/")
        if "vsa" in metrics:
            for name in ["bundle_capacity", "role_filler", "symbolic_role_filler"]:
                r = metrics["vsa"][name]
                for k in (10, 20, 30):
                    if k in r["k"]:
                        flat[f"test/vsa/{name}_k{k}"] = r["accuracy"][r["k"].index(k)]
            flat["test/vsa/depth_inv_m40"] = metrics["vsa"]["unbinding_depth_inv"]["similarity"][-1]
            flat["test/vsa/dim"] = metrics["vsa"]["dim"]
        wb.log(flat, step=step)
        for k, val in flat.items():
            wb.summary[k] = val
        wb.summary["status"] = metrics["status"]
        wb.save(mpath, base_path=rd, policy="now")
        wb.finish()
    print(f"done: {rd} status={metrics['status']} best_epoch={best_epoch}")


if __name__ == "__main__":
    main()
