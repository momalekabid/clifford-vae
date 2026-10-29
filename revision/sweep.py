"""run the revision sweep. resumable, safe to start several workers on one or more gpus.

    python revision/sweep.py --workers 4                # all groups, seeds 1-10
    python revision/sweep.py --workers 4 --seeds 11-20  # second pass, adds seeds
    python revision/sweep.py --list                     # show the job order and status

order: shapes2d, then cifar10 / fashionmnist / stl10, then mnist, and the mnist vMF runs
last. within a group the order is seed-major, so every cell has seeds 1..n before any cell
gets seed n+1. a job is claimed with an exclusive lock file in its run dir; finished runs
(metrics.json status "done") are skipped. stdout/stderr of each run go to its run dir.
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import time
from multiprocessing import Process

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from revision import common  # noqa: E402

CNN_DIMS = [128, 256, 512, 1024, 2048, 4096]
MNIST_DIMS = [2, 5, 10, 20, 40, 128]
CNN_PRIORS = ["N", "N_L2", "S", "C"]

GROUPS = {
    # name: list of (dataset, prior, d, extra args) cells
    "shapes2d": [("shapes2d", p, 128, {"beta_max": b}) for b in (1.0, 4.0) for p in CNN_PRIORS],
    "cifar10": [("cifar10", p, d, {}) for d in CNN_DIMS for p in CNN_PRIORS],
    "fashionmnist": [("fashionmnist", p, d, {}) for d in CNN_DIMS for p in CNN_PRIORS],
    "stl10": [("stl10", p, d, {}) for d in (128, 256) for p in CNN_PRIORS],
    "mnist": [("mnist", p, d, {}) for d in MNIST_DIMS for p in ["N", "N_L2", "S", "C"]],
    "mnist_vmf": [("mnist", "V", d, {}) for d in MNIST_DIMS],
}
ORDER = ["shapes2d", "cifar10", "fashionmnist", "stl10", "mnist", "mnist_vmf"]


def parse_seeds(s):
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def jobs(groups, seeds):
    # seed-major across groups too, so seed 1 of every group finishes before any seed 2.
    # mnist_vmf still goes last.
    out = []
    main = [g for g in groups if g != "mnist_vmf"]
    for seed in seeds:
        for g in main:
            for ds, prior, d, extra in GROUPS[g]:
                out.append((g, ds, prior, d, seed, extra))
    if "mnist_vmf" in groups:
        for seed in seeds:
            for ds, prior, d, extra in GROUPS["mnist_vmf"]:
                out.append(("mnist_vmf", ds, prior, d, seed, extra))
    return out


def rdir(job):
    _, ds, prior, d, seed, extra = job
    return common.run_dir(ds, prior, d, seed, extra.get("beta_max", 1.0))


def status(job):
    m = os.path.join(rdir(job), "metrics.json")
    if not os.path.exists(m):
        return "todo"
    try:
        return json.load(open(m)).get("status", "?")
    except Exception:
        return "?"


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def claim(job):
    d = rdir(job)
    os.makedirs(d, exist_ok=True)
    lock = os.path.join(d, ".lock")
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        # stale lock from a dead worker on this host: take it over
        try:
            info = json.load(open(lock))
            if info["host"] == socket.gethostname() and not _pid_alive(info["pid"]):
                os.remove(lock)
                return claim(job)
        except Exception:
            pass
        return False
    with os.fdopen(fd, "w") as f:
        json.dump({"host": socket.gethostname(), "pid": os.getpid(), "time": time.time()}, f)
    return True


def release(job):
    try:
        os.remove(os.path.join(rdir(job), ".lock"))
    except FileNotFoundError:
        pass


def run_job(job, extra_args):
    g, ds, prior, d, seed, extra = job
    cmd = [sys.executable, os.path.join(common.REPO, "revision", "train.py"), "--dataset", ds, "--prior", prior,
           "--d", str(d), "--seed", str(seed)]
    for k, v in extra.items():
        cmd += [f"--{k}", str(v)]
    cmd += extra_args
    with open(os.path.join(rdir(job), "stdout.log"), "a") as f:
        f.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n")
        f.flush()
        return subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=common.REPO)


def worker(wid, joblist, extra_args, max_retries):
    tries = {}
    while True:
        picked = None
        for job in joblist:
            st = status(job)
            if st == "done" or (st.startswith("failed") and tries.get(job, 0) >= max_retries):
                continue
            if claim(job):
                picked = job
                break
        if picked is None:
            return
        try:
            rc = run_job(picked, extra_args)
            tries[picked] = tries.get(picked, 0) + 1
            print(f"[w{wid}] {rdir(picked)} rc={rc} status={status(picked)}", flush=True)
        finally:
            release(picked)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--groups", default=",".join(ORDER))
    p.add_argument("--seeds", default="1-10")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--max_retries", type=int, default=1, help="re-runs of a failed run (fresh, same seed)")
    p.add_argument("--priors", default=None, help="comma list, e.g. N,N_L2 (default: all)")
    p.add_argument("--list", action="store_true")
    a, extra_args = p.parse_known_args()
    joblist = jobs(a.groups.split(","), parse_seeds(a.seeds))
    if a.priors:
        joblist = [j for j in joblist if j[2] in a.priors.split(",")]
    if a.list:
        from collections import Counter

        c = Counter((j[0], status(j)) for j in joblist)
        for (g, st), n in sorted(c.items()):
            print(f"{g:14s} {st:24s} {n}")
        print("total", len(joblist))
        return
    procs = [Process(target=worker, args=(i, joblist, extra_args, a.max_retries)) for i in range(a.workers)]
    for pr in procs:
        pr.start()
    for pr in procs:
        pr.join()


if __name__ == "__main__":
    main()
