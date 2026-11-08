# run infrastructure: seeding, device, run dirs, provenance, json and csv logging.
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
# several runs often share one machine; torch would otherwise grab every core for the small
# cpu ops in the vsa tests, which is slower than a few threads
N_THREADS = int(os.environ.get("CVAE_THREADS", "4"))
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, str(N_THREADS))

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(N_THREADS)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.environ.get("CVAE_RESULTS", os.path.join(REPO, "runs"))
DATA = os.environ.get("CVAE_DATA", os.path.join(REPO, "data"))
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "clifford-vae")


def seed_everything(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if os.environ.get("CVAE_DETERMINISTIC", "1") == "1":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True
    return torch.Generator().manual_seed(seed)


def device() -> torch.device:
    if os.environ.get("CVAE_DEVICE"):
        return torch.device(os.environ["CVAE_DEVICE"])
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_dir(dataset: str, prior: str, d: int, seed: int, beta_max: float = 1.0, root: str = None) -> str:
    cell = f"{prior}-d{d}" + ("" if beta_max == 1.0 else f"-b{beta_max:g}")
    return os.path.join(root or RESULTS, dataset, cell, f"seed{seed}")


def ckpt_path(rd: str, tag: str) -> str:
    """a run's checkpoint, named after the run so it still says what it is when copied out, e.g.
    runs/cifar10/C-d256/seed1/cifar10_C-d256_seed1_best.pt"""
    ds, cell, seed = os.path.normpath(rd).split(os.sep)[-3:]
    return os.path.join(rd, f"{ds}_{cell}_{seed}_{tag}.pt")


def find_ckpt(rd: str, tag: str = "best"):
    """the checkpoint path, or None. also finds the names older runs used (best.pt, best_fp16.pt)."""
    for p in (ckpt_path(rd, tag), os.path.join(rd, f"{tag}.pt"), os.path.join(rd, f"{tag}_fp16.pt")):
        if os.path.exists(p):
            return p
    return None


def _git(*args):
    try:
        return subprocess.check_output(["git", "-C", REPO, *args], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def provenance() -> dict:
    return {
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "time_start": time.strftime("%Y-%m-%d %H:%M:%S"),
        "argv": sys.argv,
    }


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray) or torch.is_tensor(o):
        return o.tolist()
    return str(o)


def write_json(path: str, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=_jsonable)
    os.replace(tmp, path)


def read_json(path: str, default=None):
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


def get(d, path, default=None):
    """nested lookup with a dotted path, e.g. get(m, "knn.acc_100")."""
    for p in path.split("."):
        if not isinstance(d, dict) or p not in d:
            return default
        d = d[p]
    return d


class CurveLogger:
    """rows to a local csv (survives without wandb) and to wandb when a run is active.
    append mode, so a resumed run keeps its earlier rows."""

    def __init__(self, path: str, fields, wandb_run=None):
        new = not os.path.exists(path)
        self.f = open(path, "a", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=fields, extrasaction="ignore")
        if new:
            self.w.writeheader()
        self.wandb_run = wandb_run

    def log(self, row: dict, step: int, prefix: str = ""):
        self.w.writerow(row)
        if self.wandb_run is not None:
            self.wandb_run.log({prefix + k: v for k, v in row.items()}, step=step)

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
