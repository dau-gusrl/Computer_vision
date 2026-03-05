"""
실시간 추론 및 FPS 측정 (Real-time Inference and FPS Measurement)

학습 목표:
- FPS(Frames Per Second) 계산 방법
- 처리 시간 병목 지점 파악
- 최적화 전략 (리사이즈, 스킵 프레임, 해상도 조절)
- 경량 모델의 필요성 이해
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import time
import collections


class FPSCounter:
    """롤링 윈도우 기반 FPS 측정기.

    단순한 프레임 간 시간 측정보다 안정적인 FPS를 제공합니다.
    """

    def __init__(self, window_size: int = 30):
        """
        Args:
            window_size: 평균 계산에 사용할 최근 프레임 수
        """
        self.timestamps = collections.deque(maxlen=window_size)
        self.window_size = window_size

    def tick(self) -> None:
        """현재 타임스탬프를 기록합니다."""
        self.timestamps.append(time.perf_counter())

    def get_fps(self) -> float:
        """현재 FPS를 반환합니다.

        Returns:
            측정된 FPS (측정 데이터 부족 시 0.0)
        """
        if len(self.timestamps) < 2:
            return 0.0
        elapsed = self.timestamps[-1] - self.timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self.timestamps) - 1) / elapsed


class ProcessingTimer:
    """코드 블록의 처리 시간을 측정하는 컨텍스트 매니저.

    Example:
        with ProcessingTimer("Detection") as t:
            result = detect(frame)
        print(f"처리 시간: {t.elapsed_ms:.1f}ms")
    """

    def __init__(self, label: str = ""):
        self.label = label
        self.elapsed_ms = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000


def measure_processing_time(func, *args, n_runs: int = 10, **kwargs) -> dict:
    """함수의 처리 시간을 반복 측정합니다.

    Args:
        func: 측정할 함수
        *args: 함수 인수
        n_runs: 반복 횟수
        **kwargs: 함수 키워드 인수

    Returns:
        {'mean_ms': float, 'min_ms': float, 'max_ms': float, 'std_ms': float}
    """
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        func(*args, **kwargs)
        times.append((time.perf_counter() - start) * 1000)

    times_arr = np.array(times)
    return {
        "mean_ms": float(np.mean(times_arr)),
        "min_ms": float(np.min(times_arr)),
        "max_ms": float(np.max(times_arr)),
        "std_ms": float(np.std(times_arr)),
    }


def draw_fps_overlay(image: np.ndarray, fps: float,
                     processing_ms: float = None) -> np.ndarray:
    """FPS와 처리 시간을 이미지에 표시합니다.

    Args:
        image: BGR 이미지
        fps: FPS 값
        processing_ms: 처리 시간 (밀리초, 선택)

    Returns:
        오버레이가 추가된 이미지
    """
    result = image.copy()

    # FPS 색상: 30fps 이상=초록, 15-30=노랑, 15 미만=빨강
    if fps >= 30:
        color = (0, 255, 0)
    elif fps >= 15:
        color = (0, 255, 255)
    else:
        color = (0, 0, 255)

    fps_text = f"FPS: {fps:.1f}"
    cv.putText(result, fps_text, (10, 30),
               cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
    cv.putText(result, fps_text, (10, 30),
               cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    if processing_ms is not None:
        ms_text = f"Processing: {processing_ms:.1f}ms"
        cv.putText(result, ms_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv.putText(result, ms_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    return result


def benchmark_resolution_impact() -> None:
    """이미지 해상도가 처리 속도에 미치는 영향을 측정합니다."""
    # 기준 이미지 생성
    base_image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

    resolutions = [
        ("1920×1080 (Full HD)", (1920, 1080)),
        ("1280×720 (HD)", (1280, 720)),
        ("640×480 (VGA)", (640, 480)),
        ("320×240 (QVGA)", (320, 240)),
    ]

    def process_frame(img: np.ndarray) -> np.ndarray:
        """대표적인 처리 파이프라인: 그레이스케일 → 블러 → 캐니"""
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        blurred = cv.GaussianBlur(gray, (5, 5), 0)
        return cv.Canny(blurred, 50, 150)

    print("=== 해상도별 처리 속도 비교 ===")
    print(f"{'해상도':<25} {'평균(ms)':<12} {'FPS 추정':<12}")
    print("-" * 50)

    results = []
    for label, (w, h) in resolutions:
        img = cv.resize(base_image, (w, h))
        timing = measure_processing_time(process_frame, img, n_runs=20)
        fps_estimate = 1000 / timing["mean_ms"]
        results.append((label, timing["mean_ms"], fps_estimate))
        print(f"{label:<25} {timing['mean_ms']:<12.2f} {fps_estimate:<12.1f}")

    # 시각화
    labels = [r[0] for r in results]
    fps_values = [r[2] for r in results]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(range(len(labels)), fps_values,
                   color=["red" if f < 15 else "yellow" if f < 30 else "green"
                          for f in fps_values])
    plt.xticks(range(len(labels)), labels, rotation=10, ha="right")
    plt.ylabel("추정 FPS")
    plt.title("해상도별 처리 속도\n(빨강: <15fps, 노랑: 15-30fps, 초록: >30fps)")
    plt.axhline(y=30, color="green", linestyle="--", alpha=0.5, label="30fps 기준")
    plt.axhline(y=15, color="red", linestyle="--", alpha=0.5, label="15fps 기준")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def simulate_realtime_processing() -> None:
    """실시간 처리 루프를 시뮬레이션합니다.

    실제 카메라 처리 루프 패턴:
        cap = cv.VideoCapture(0)
        fps_counter = FPSCounter()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            with ProcessingTimer("process") as t:
                result = process_frame(frame)
            fps_counter.tick()
            result = draw_fps_overlay(result, fps_counter.get_fps(), t.elapsed_ms)
            cv.imshow("Result", result)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv.destroyAllWindows()
    """
    print("=== 실시간 처리 시뮬레이션 ===\n")

    # 합성 프레임으로 처리 루프 시뮬레이션
    fps_counter = FPSCounter(window_size=30)
    n_frames = 60

    processing_times = []
    fps_history = []

    for i in range(n_frames):
        # 프레임 생성 (카메라 캡처 시뮬레이션)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # 처리 시간 측정
        with ProcessingTimer("frame_processing") as t:
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            blurred = cv.GaussianBlur(gray, (5, 5), 0)
            edges = cv.Canny(blurred, 50, 150)

        fps_counter.tick()
        processing_times.append(t.elapsed_ms)
        fps_history.append(fps_counter.get_fps())

    avg_fps = np.mean(fps_history[10:])  # 초기 측정 제외
    avg_ms = np.mean(processing_times)

    print(f"처리 결과 ({n_frames}프레임):")
    print(f"  평균 처리 시간: {avg_ms:.2f}ms")
    print(f"  평균 FPS: {avg_fps:.1f}")
    print(f"  최소/최대 처리 시간: {min(processing_times):.2f}ms / {max(processing_times):.2f}ms")

    # 처리 시간과 FPS 그래프
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("실시간 처리 성능 모니터링", fontsize=14)

    axes[0].plot(processing_times, color="blue", alpha=0.7)
    axes[0].axhline(y=avg_ms, color="red", linestyle="--",
                    label=f"평균: {avg_ms:.1f}ms")
    axes[0].set_xlabel("프레임")
    axes[0].set_ylabel("처리 시간 (ms)")
    axes[0].set_title("프레임별 처리 시간")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(fps_history, color="green", alpha=0.7)
    axes[1].axhline(y=30, color="green", linestyle="--", alpha=0.5, label="30fps")
    axes[1].axhline(y=avg_fps, color="red", linestyle="--",
                    label=f"평균: {avg_fps:.1f}fps")
    axes[1].set_xlabel("프레임")
    axes[1].set_ylabel("FPS")
    axes[1].set_title("FPS 히스토리")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def demo_realtime_inference() -> None:
    """실시간 추론 개념을 시연합니다."""
    benchmark_resolution_impact()
    simulate_realtime_processing()

    print("""
=== 실시간 처리 최적화 전략 ===

1. 해상도 축소
   처리 전: frame = cv.resize(frame, (640, 480))
   효과: 처리 시간 ∝ 픽셀 수 = W × H

2. 스킵 프레임 (Skip Frame)
   frame_count += 1
   if frame_count % 3 == 0:  # 3프레임마다 처리
       result = detect(frame)
   효과: FPS 3배 향상, 탐지 빈도 감소

3. 멀티스레딩
   별도 스레드에서 캡처/처리/표시 분리
   효과: I/O 대기 시간 최소화

4. 경량 모델 사용
   MobileNet, YOLOv5n (nano), EfficientDet-Lite 등
   효과: 정확도 소폭 감소, 속도 대폭 향상

5. 하드웨어 가속
   GPU: cv.dnn.DNN_TARGET_CUDA
   NPU: 전용 추론 하드웨어
""")


if __name__ == "__main__":
    demo_realtime_inference()
