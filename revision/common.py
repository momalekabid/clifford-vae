# shared run infrastructure for the revision: seeding, run dirs, provenance, logging.
import csv
import json
import os
import platform
import random
import socket
import subprocess
import sys
import time

# must be set before cuda initialises for deterministic cublas
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
VMF_PATH = os.path.join(REPO, "vmf")
if VMF_PATH not in sys.path:
    # the patched vmf copy (float64 log-bessel), never the installed egg
    sys.path.insert(0, VMF_PATH)

RESULTS_ROOT = os.environ.get("REVISION_RESULTS", os.path.join(REPO, "results_rev"))
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "clifford-vae-revision")


def seed_everything(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def device() -> torch.device:
    if os.environ.get("REVISION_DEVICE"):
        return torch.device(os.environ["REVISION_DEVICE"])
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _git(*args):
    try:
        return subprocess.check_output(["git", "-C", REPO, *args], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def provenance() -> dict:
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "git_sha": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": gpu,
        "time_start": time.strftime("%Y-%m-%d %H:%M:%S"),
        "argv": sys.argv,
    }


def pip_freeze(path: str):
    try:
        out = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], stderr=subprocess.DEVNULL).decode()
    except Exception as e:
        out = f"pip freeze failed: {e}\n"
    with open(path, "w") as f:
        f.write(out)


def run_dir(dataset: str, prior: str, d: int, seed: int, beta_max: float = 1.0) -> str:
    cell = f"{prior}-d{d}" + ("" if beta_max == 1.0 else f"-b{beta_max:g}")
    return os.path.join(RESULTS_ROOT, dataset, cell, f"seed{seed}")


def write_json(path: str, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=_jsonable)
    os.replace(tmp, path)


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if torch.is_tensor(o):
        return o.tolist()
    return str(o)


class CurveLogger:
    """per-step curve to a local csv and to wandb (if a run is active)."""

    def __init__(self, path: str, fields, wandb_run=None):
        self.f = open(path, "w", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=fields, extrasaction="ignore")
        self.w.writeheader()
        self.wandb_run = wandb_run

    def log(self, row: dict, step: int):
        self.w.writerow(row)
        if self.wandb_run is not None:
            self.wandb_run.log(row, step=step)

    def flush(self):
        self.f.flush()

    def close(self):
        self.f.close()


def flatten_scalars(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten_scalars(v, key + "/"))
        elif isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool):
            out[key] = float(v)
    return out
