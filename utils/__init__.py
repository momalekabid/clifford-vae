from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch

from .wandb_utils import (
    WandbLogger,
    test_self_binding,
    compute_class_means,
    evaluate_mean_vector_cosine,
    vsa_bind,
    vsa_unbind,
    vsa_invert,
    test_vsa_operations,
)


DEFAULT_WANDB_PROJECT: str = "clifford-vae"
DEFAULT_RESULTS_DIR: str = "results"


def get_default_device(prefer_mps: bool = True) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if (
        prefer_mps
        and getattr(torch.backends, "mps", None)
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def set_global_seeds(seed: int = 42, deterministic_cudnn: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(deterministic_cudnn)
    except Exception:
        pass
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = deterministic_cudnn
        torch.backends.cudnn.benchmark = not deterministic_cudnn


__all__ = [
    # exports
    "WandbLogger",
    "test_self_binding",
    "compute_class_means",
    "evaluate_mean_vector_cosine",
    "vsa_bind",
    "vsa_unbind",
    "vsa_invert",
    "test_vsa_operations",
    # defaults/helpers
    "DEFAULT_WANDB_PROJECT",
    "DEFAULT_RESULTS_DIR",
    "get_default_device",
    "ensure_dir",
    "set_global_seeds",
]
