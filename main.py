"""PoseKeeper 진입점 — 시스템 트레이 앱 실행.

    python main.py

추론은 백그라운드 스레드에서 돌고, 트레이 아이콘이 메인 스레드를 잡고 있음.
사용자가 트레이 메뉴 -> "대시보드 열기" 누를 때만 Tkinter 창이 뜸.
"""
from __future__ import annotations

import sys


def main() -> int:
    # 실제 실행 로직은 src/app/tray.py 가 완성되면 여기서 호출.
    try:
        from src.app.tray import run_tray
    except ImportError:
        print("[INFO] src/app/tray.py 의 run_tray() 가 아직 구현되지 않았습니다.")
        print("       먼저 `python setup.py` 와 `python test_env.py` 로 환경을 검증하세요.")
        return 0
    return run_tray()


if __name__ == "__main__":
    sys.exit(main())
