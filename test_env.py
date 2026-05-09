"""Post-install environment check.

Each check prints one line:  [OK] / [FAIL] / [SKIP] <name>: <detail>
A non-zero exit code means at least one [FAIL].

Webcam, pystray, plyer, and tkinter checks are SKIPPED on Colab — the
runtime has no display server or USB device, and the training-only Colab
flow doesn't need them.
"""
from __future__ import annotations

import importlib
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.utils.device import detect_device, is_colab  # noqa: E402


PASSED: list[str] = []
FAILED: list[str] = []
SKIPPED: list[str] = []


def ok(name: str, detail: str = "") -> None:
    PASSED.append(name)
    print(f"[OK]   {name}{f' - {detail}' if detail else ''}")


def fail(name: str, err: BaseException) -> None:
    FAILED.append(name)
    print(f"[FAIL] {name} - {type(err).__name__}: {err}")


def skip(name: str, reason: str) -> None:
    SKIPPED.append(name)
    print(f"[SKIP] {name} - {reason}")


def check_import(module: str, *, attr: str | None = None) -> None:
    try:
        m = importlib.import_module(module)
        version = getattr(m, "__version__", "?")
        if attr:
            getattr(m, attr)
        ok(f"import {module}", f"v{version}")
    except Exception as e:  # noqa: BLE001
        fail(f"import {module}", e)


def check_device() -> None:
    try:
        import torch
        dev = detect_device()
        x = torch.tensor([1.0]).to(dev)
        _ = (x * 2).cpu().item()
        ok(f"torch device={dev}",
           f"cuda_available={torch.cuda.is_available()} "
           f"mps_available={getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available()}")
    except Exception as e:  # noqa: BLE001
        fail("torch device", e)


def check_mediapipe_inference() -> None:
    """Actually instantiate Pose + FaceMesh and run them on a dummy frame."""
    try:
        import numpy as np
        import mediapipe.python.solutions.pose as mp_pose
        import mediapipe.python.solutions.face_mesh as mp_face

        dummy = np.zeros((240, 320, 3), dtype=np.uint8)
        with mp_pose.Pose(model_complexity=0) as pose:
            pose.process(dummy)
        with mp_face.FaceMesh(refine_landmarks=False) as fm:
            fm.process(dummy)
        ok("MediaPipe Pose + FaceMesh", "instantiated and ran on dummy frame")
    except Exception as e:  # noqa: BLE001
        fail("MediaPipe Pose + FaceMesh", e)


def check_webcam() -> None:
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            fail("webcam VideoCapture(0)", RuntimeError("camera did not open"))
            return
        try:
            grabbed, frame = cap.read()
        finally:
            cap.release()
        if not grabbed or frame is None:
            fail("webcam VideoCapture(0)", RuntimeError("no frame returned"))
            return
        h, w = frame.shape[:2]
        ok("webcam VideoCapture(0)", f"frame {w}x{h}")
    except Exception as e:  # noqa: BLE001
        fail("webcam VideoCapture(0)", e)


def check_tkinter() -> None:
    try:
        import tkinter
        root = tkinter.Tk()
        root.withdraw()
        root.destroy()
        ok("tkinter", f"Tcl/Tk {tkinter.TkVersion}")
    except Exception as e:  # noqa: BLE001
        fail("tkinter", e)


def main() -> int:
    print(f"\n=== PoseKeeper environment check ===\n")
    check_import("numpy")
    check_import("pandas")
    check_import("torch")
    check_import("cv2")
    check_import("mediapipe")
    check_import("matplotlib")
    check_import("sqlite3")
    check_device()
    check_mediapipe_inference()

    if is_colab():
        skip("pystray", "not used on Colab")
        skip("plyer", "not used on Colab")
        skip("tkinter", "no display on Colab")
        skip("webcam VideoCapture(0)", "no webcam on Colab — use uploaded video instead")
    else:
        check_import("pystray")
        check_import("plyer")
        check_tkinter()
        check_webcam()

    total = len(PASSED) + len(FAILED) + len(SKIPPED)
    print(f"\n[Summary] {len(PASSED)}/{total} passed, "
          f"{len(FAILED)} failed, {len(SKIPPED)} skipped")
    return 1 if FAILED else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
