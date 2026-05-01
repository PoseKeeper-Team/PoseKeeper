"""PoseKeeper environment setup.

Detects the runtime environment (CUDA / Apple Silicon / CPU / Colab) and
installs the matching requirements file. Run from the project root:

    python setup.py                  # auto-detect, install, run test_env.py
    python setup.py --env cpu        # force a specific environment
    python setup.py --no-install     # detect only, skip pip
    python setup.py --skip-test      # install but don't run test_env.py

Requires Python 3.10.x. MediaPipe wheels are most stable on 3.10; on 3.11+
the install may succeed but is unsupported by the team. On 3.9- it will
refuse to run.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# We import device detection from src.utils so there is exactly one detector.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.utils.device import detect_env, is_apple_silicon  # noqa: E402


REQUIRED_PY = (3, 10)
PROJECT_ROOT = Path(__file__).resolve().parent
REQ_DIR = PROJECT_ROOT / "requirements"


def check_python_version() -> None:
    major, minor = sys.version_info[:2]
    if (major, minor) < REQUIRED_PY:
        print(f"[FAIL] Python {major}.{minor} detected — PoseKeeper requires 3.10.x.")
        sys.exit(1)
    if (major, minor) != REQUIRED_PY:
        print(
            f"[WARN] Python {major}.{minor} detected. The team standard is 3.10.x; "
            "MediaPipe wheels are most stable there. Continuing anyway."
        )


def pip_install(req_file: Path) -> int:
    cmd = [sys.executable, "-m", "pip", "install", "-r", str(req_file)]
    print(f"[INFO] Running: {' '.join(cmd)}")
    return subprocess.call(cmd)


def install_for_env(env: str) -> None:
    req_file = REQ_DIR / f"{env}.txt"
    if not req_file.exists():
        print(f"[FAIL] {req_file} not found.")
        sys.exit(1)

    rc = pip_install(req_file)
    if rc != 0:
        # Apple Silicon: mediapipe sometimes lacks a wheel and falls through.
        # mediapipe-silicon is a community wheel that fills the gap.
        if env == "mac" and is_apple_silicon():
            print("[WARN] Initial install failed. Retrying mediapipe via "
                  "'mediapipe-silicon' fallback...")
            rc2 = subprocess.call(
                [sys.executable, "-m", "pip", "install", "mediapipe-silicon"]
            )
            if rc2 != 0:
                print("[FAIL] mediapipe-silicon also failed. Try Python 3.10 in "
                      "a fresh venv: `python3.10 -m venv .venv && source "
                      ".venv/bin/activate`.")
                sys.exit(1)
        else:
            print(f"[FAIL] pip install returned {rc}. See output above.")
            sys.exit(rc)


def run_env_check() -> int:
    test_script = PROJECT_ROOT / "test_env.py"
    if not test_script.exists():
        print("[WARN] test_env.py not found. Skipping post-install check.")
        return 0
    print("\n[INFO] Running test_env.py to verify the install...\n")
    return subprocess.call([sys.executable, str(test_script)])


def main() -> None:
    parser = argparse.ArgumentParser(description="PoseKeeper environment setup")
    parser.add_argument(
        "--env",
        choices=["auto", "cuda", "cpu", "mac", "colab"],
        default="auto",
        help="Force a specific environment (default: auto-detect).",
    )
    parser.add_argument("--no-install", action="store_true",
                        help="Detect only; do not run pip install.")
    parser.add_argument("--skip-test", action="store_true",
                        help="Skip running test_env.py after install.")
    args = parser.parse_args()

    check_python_version()

    env = detect_env() if args.env == "auto" else args.env
    print(f"[INFO] Target environment: {env}")

    if args.no_install:
        print("[INFO] --no-install set. Exiting after detection.")
        return

    install_for_env(env)
    print("\n[OK] Install complete.")

    if not args.skip_test:
        rc = run_env_check()
        sys.exit(rc)


if __name__ == "__main__":
    main()
