"""every paper cell x runs (one seed per run), resumable. workers claim runs with a lock file, so several copies can
share one results root (one per gpu, or several machines on a shared disk). seed-major: seed 1 of
every cell finishes before any seed 2.

    python run_experiments.py --list                                       # what's done
    python run_experiments.py --workers 8                                  # everything, 20 runs per cell
    python run_experiments.py --datasets cifar10 --runs 5 --gpus 0,1 --workers 8
    python run_experiments.py --datasets cifar10 --runs 6-20               # runs 6 to 20 (e.g. a second machine)
    python run_experiments.py --datasets mnist --priors C,N --dims 2,10    # a subset
    python run_experiments.py --datasets shapes3d --workers 4              # disentangle test, 3 runs
    python run_experiments.py --datasets omniglot --workers 4              # few-shot + held-out pairs, 5 runs
any unknown flag goes through to the train script (e.g. --epochs 2 --no_wandb).
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import time
from collections import Counter
from multiprocessing import Process

from clifford_vae import common

SCRIPTS = {"mnist": "train_mnist.py", "fashionmnist": "train_fashionmnist.py", "cifar10": "train_cifar10.py",
           "celeba": "train_celeba.py", "shapes3d": "train_disentangle.py",
           "omniglot": "train_omniglot.py"}
CNN = ([128, 256, 512, 1024, 2048, 4096], ["N", "N_L2", "S", "C"])
CELLS = {"cifar10": CNN, "fashionmnist": CNN, "mnist": ([2, 5, 10, 20, 40, 128], ["N", "N_L2", "S", "V", "C"]),
         "celeba": ([128, 256, 512], ["N", "N_L2", "S", "C"]), "shapes3d": ([128], ["N", "N_L2", "S", "C"]),
         "omniglot": ([256, 512], ["N", "N_L2", "C"])}
# cells whose d depends on the prior. omniglot: every code is 512 long, as in karunaratne et al.
# 2021 (clifford's code is 2d long, so C runs at d=256)
PRIOR_DIM = {"omniglot": {"N": 512, "N_L2": 512, "C": 256}}
BETAS = {"shapes3d": [1.0, 4.0], "omniglot": [1.0, 4.0]}  # beta_max per cell (beta-vae for the disentangle tests)
ORDER = ["cifar10", "fashionmnist", "mnist", "celeba"]  # shapes3d and omniglot are standalone: --datasets ...
RUNS = {"celeba": "5", "shapes3d": "3", "omniglot": "5"}  # runs per cell when --runs is not given (else 20)


def run_ids(s):
    """--runs: a count ("5" = runs 1-5), a range ("6-20") or a list ("1,4,7"). run i uses seed i."""
    return list(range(1, int(s) + 1)) if s.isdigit() else parse_list(s, int)


def parse_list(s, cast=str):
    if s is None:
        return None
    if "-" in s and cast is int:
        lo, hi = s.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [cast(x) for x in s.split(",")]


def jobs(a):
    out = []
    seeds = {ds: run_ids(a.runs or RUNS.get(ds, "20")) for ds in parse_list(a.datasets)}
    for seed in sorted(set().union(*seeds.values())):
        for ds in [x for x in parse_list(a.datasets) if seed in seeds[x]]:
            dims, priors = CELLS[ds]
            pairs = [(PRIOR_DIM[ds][p], p) for p in priors] if ds in PRIOR_DIM else [(d, p) for d in dims for p in priors]
            for d, p in pairs:
                for b in BETAS.get(ds, [1.0]):
                    if (a.dims is None or d in parse_list(a.dims, int)) and (a.priors is None or p in parse_list(a.priors)):
                        out.append((ds, p, d, seed, b))
    return out


def rdir(job, root):
    ds, p, d, seed, b = job
    return common.run_dir(ds, p, d, seed, b, root=root)


def status(job, root):
    return common.get(common.read_json(os.path.join(rdir(job, root), "metrics.json"), {}), "status", "todo")


def _alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def claim(job, root):
    lock = os.path.join(rdir(job, root), ".lock")
    os.makedirs(os.path.dirname(lock), exist_ok=True)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:  # stale lock from a dead worker on this host: take it over
            info = json.load(open(lock))
            if info["host"] == socket.gethostname() and not _alive(info["pid"]):
                os.remove(lock)
                return claim(job, root)
        except Exception:
            pass
        return False
    with os.fdopen(fd, "w") as f:
        json.dump({"host": socket.gethostname(), "pid": os.getpid(), "time": time.time()}, f)
    return True


def worker(wid, joblist, a, passthrough):
    env = dict(os.environ)
    if a.gpus:
        gpus = parse_list(a.gpus)
        env["CUDA_VISIBLE_DEVICES"] = gpus[wid % len(gpus)]
    tries = Counter()
    while True:
        job = next((j for j in joblist if status(j, a.root) != "done" and tries[j] <= a.max_retries and claim(j, a.root)),
                   None)
        if job is None:
            return
        ds, p, d, seed, b = job
        cmd = [sys.executable, os.path.join(common.REPO, SCRIPTS[ds]), "--prior", p, "--d", str(d), "--seed", str(seed),
               *(["--beta_max", str(b)] if b != 1.0 else []),
               *(["--root", a.root] if a.root else []), *passthrough]
        try:
            with open(os.path.join(rdir(job, a.root), "stdout.log"), "a") as f:
                f.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n")
                f.flush()
                rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=common.REPO, env=env)
            tries[job] += 1  # any exit that isn't "done" (crash, failure) counts as a try
            print(f"[w{wid}] {rdir(job, a.root)} rc={rc} status={status(job, a.root)}", flush=True)
        finally:
            try:
                os.remove(os.path.join(rdir(job, a.root), ".lock"))
            except FileNotFoundError:
                pass


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", default=",".join(ORDER))
    p.add_argument("--runs", default=None, help="runs per cell: a count (5), a range (6-20) or a list (1,4,7); "
                                                "default 20, celeba 5, shapes3d 3, omniglot 5")
    p.add_argument("--dims", default=None, help="comma list (default: every paper dim)")
    p.add_argument("--priors", default=None, help="comma list (default: every prior)")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--gpus", default=None, help="comma list of gpu ids, workers are spread over them")
    p.add_argument("--max_retries", type=int, default=1)
    p.add_argument("--root", default=None)
    p.add_argument("--list", action="store_true")
    a, passthrough = p.parse_known_args()
    joblist = jobs(a)
    if a.list:
        c = Counter((j[0], status(j, a.root)) for j in joblist)
        for (ds, st), n in sorted(c.items()):
            print(f"{ds:14s} {st:24s} {n}")
        print("total", len(joblist))
        return
    procs = [Process(target=worker, args=(i, joblist, a, passthrough)) for i in range(a.workers)]
    for pr in procs:
        pr.start()
    for pr in procs:
        pr.join()


if __name__ == "__main__":
    main()
