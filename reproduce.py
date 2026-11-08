"""everything in one go: train + evaluate every run (run_experiments.py), then the clean timings on
the now idle gpu (timing.py), then every table and figure (tables.py, figures.py). resumable: runs
that are done are skipped, so stop it and start it again any time. flags other than the ones below
go to run_experiments.py (--runs, --dims, --priors, --workers, --gpus, --max_retries).

    python reproduce.py --datasets cifar10,fashionmnist,mnist --workers 8 --gpus 0,1
    python reproduce.py --datasets cifar10 --runs 5 --dims 128,256 --workers 4
    python reproduce.py --report_only              # no training: timing, tables, figures from runs/
"""
import argparse
import subprocess
import sys

import torch

from clifford_vae import common
from run_experiments import ORDER

TIMED = ("cifar10", "fashionmnist", "celeba")  # the cost tables are for the cnn datasets


def step(name, cmd):
    print(f"\n=== {name}: {' '.join(cmd)}", flush=True)
    r = subprocess.run([sys.executable, *cmd])
    if r.returncode:
        print(f"=== {name} exited with {r.returncode}; carrying on", flush=True)
    return r.returncode


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", default=",".join(ORDER))
    p.add_argument("--root", default=common.RESULTS)
    p.add_argument("--report_only", action="store_true", help="skip training, only timing, tables, figures")
    p.add_argument("--skip_timing", action="store_true")
    a, rest = p.parse_known_args()
    root = ["--root", a.root]
    if not a.report_only:
        step("experiments", ["run_experiments.py", "--datasets", a.datasets, *root, *rest])
    timed = [d for d in a.datasets.split(",") if d in TIMED]
    if timed and not a.skip_timing:
        if torch.cuda.is_available():
            step("timing", ["timing.py", "--datasets", ",".join(timed), *root])
        else:
            print("\n=== timing skipped: no cuda gpu", flush=True)
    step("tables", ["tables.py", *root])
    step("figures", ["figures.py", *root])
    print(f"\ndone. tables in {a.root}/_aggregate/tables, figures in {a.root}/_aggregate/figures", flush=True)


if __name__ == "__main__":
    main()
