"""Device & runtime environment detection.

Single source of truth used by config.py and any module that needs to know
"am I on CUDA / MPS / CPU?" or "am I running inside Colab?".
"""
from __future__ import annotations

import os
import platform
import sys


def is_colab() -> bool:
    """True iff running inside a Google Colab notebook."""
    if "google.colab" in sys.modules:
        return True
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return "COLAB_GPU" in os.environ or "COLAB_RELEASE_TAG" in os.environ


def is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64")


def detect_device() -> str:
    """Return 'cuda' | 'mps' | 'cpu'. torch import is deferred so this module
    can be imported before torch is installed (e.g. from setup.py)."""
    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def detect_env() -> str:
    """Return 'colab' | 'cuda' | 'mac' | 'cpu' — used by setup.py to pick a
    requirements file. Distinct from detect_device() because Mac on MPS still
    needs a different install command than Linux/Windows CPU."""
    if is_colab():
        return "colab"
    if is_apple_silicon():
        return "mac"
    if detect_device() == "cuda":
        return "cuda"
    # No torch installed yet? Probe nvidia-smi as a hint.
    try:
        import subprocess
        subprocess.run(
            ["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            check=True, timeout=3,
        )
        return "cuda"
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return "cpu"
