"""PoseKeeper 진입점 — 시스템 트레이 앱 실행.

    python main.py

추론은 백그라운드 스레드에서 돌고, 트레이 아이콘이 메인 스레드를 잡고 있음.
사용자가 트레이 메뉴 -> "대시보드 열기" 누를 때만 Tkinter 창이 뜸.
"""
from __future__ import annotations

import logging
import os
import platform
import sys

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("src").setLevel(logging.DEBUG)


def main() -> int:
    # macOS에서 SSL 인증서 관련 에러(MediaPipe 모델 다운로드 실패) 방지
    if platform.system() == "Darwin":
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context

    # 트레이 앱 import가 실패하면 의존성 또는 코드 경로 문제가 있는 상태다.
    try:
        from src.app.tray import run_tray
    except ImportError as exc:
        print(f"[ERROR] 트레이 앱 import 실패: {exc}")
        print("        `python setup.py` 와 `python test_env.py` 로 환경을 먼저 점검하세요.")
        return 0

    if platform.system() == "Darwin":
        print("\n" + "="*50)
        print("[INFO] macOS에서 실행 중입니다.")
        print("[INFO] 앱이 실행되면 화면 상단 '메뉴바'의 아이콘을 확인하세요.")
        print("[INFO] 아이콘을 클릭하고 '대시보드 열기'를 누르면 창이 뜹니다.")
        print("="*50 + "\n")

    return run_tray()


if __name__ == "__main__":
    sys.exit(main())
