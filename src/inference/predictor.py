"""실시간 추론 통합 모듈.

3개 모델(MLP/LSTM/AE)을 동시에 lazy-load 해서 한 번의 MediaPipe 처리 결과를
공유하도록 구성. config.BG_* 값을 따라가며, 대시보드 모드로 전환되면
config.DASHBOARD_* 로 스위치.
"""
