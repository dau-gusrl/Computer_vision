"""
히스토그램 처리 (Histogram Processing)

학습 목표:
- 히스토그램의 의미와 계산 방법
- 히스토그램 평활화(Histogram Equalization)의 원리
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- 히스토그램을 이용한 이미지 분석
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def compute_histogram(image: np.ndarray, channel: int = 0) -> np.ndarray:
    """단일 채널의 히스토그램을 계산합니다.

    Args:
        image: 그레이스케일 또는 BGR 이미지
        channel: 계산할 채널 인덱스 (그레이스케일은 0)

    Returns:
        히스토그램 배열 (shape: [256, 1])

    Note:
        cv.calcHist() 파라미터:
            images   : 이미지 리스트
            channels : 계산할 채널 번호 리스트
            mask     : 관심 영역 마스크 (None이면 전체 이미지)
            histSize : 빈(bin) 수 리스트
            ranges   : 값 범위 리스트
    """
    return cv.calcHist([image], [channel], None, [256], [0, 256])


def equalize_histogram(image: np.ndarray) -> np.ndarray:
    """히스토그램 평활화를 적용합니다.

    원리:
        CDF(누적 분포 함수)를 이용하여 히스토그램을 균일하게 분산
        어두운 이미지의 대비(contrast)를 향상시키는 데 효과적

    Args:
        image: 그레이스케일 이미지 (uint8)

    Returns:
        평활화된 이미지

    Note:
        컬러 이미지의 경우 HSV의 V 채널에만 적용 후 변환 권장
        (RGB 채널에 각각 적용하면 색상이 왜곡될 수 있음)
    """
    if image.ndim == 3:
        # 컬러: HSV V 채널에만 적용
        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        hsv[:, :, 2] = cv.equalizeHist(hsv[:, :, 2])
        return cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    return cv.equalizeHist(image)


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0,
                tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """CLAHE(Contrast Limited Adaptive Histogram Equalization)를 적용합니다.

    원리:
        이미지를 타일로 나누어 각 타일에 히스토그램 평활화 적용
        clip_limit로 노이즈 증폭을 제한

    특징 (vs 일반 평활화):
        - 지역적 대비 향상으로 세부 묘사 보존
        - clip_limit로 과도한 노이즈 증폭 방지
        - 의료 이미지, 위성 이미지에 특히 효과적

    Args:
        image: 그레이스케일 또는 BGR 이미지
        clip_limit: 노이즈 증폭 제한 임계값 (낮을수록 약한 효과)
        tile_grid_size: 타일 분할 크기 (더 작으면 더 지역적)

    Returns:
        CLAHE가 적용된 이미지
    """
    clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    if image.ndim == 3:
        # 컬러: LAB의 L 채널에만 적용 (색상 보존)
        lab = cv.cvtColor(image, cv.COLOR_BGR2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv.cvtColor(lab, cv.COLOR_LAB2BGR)
    return clahe.apply(image)


def adjust_brightness_contrast(image: np.ndarray, alpha: float = 1.0,
                                beta: int = 0) -> np.ndarray:
    """밝기와 대비를 선형 변환으로 조정합니다.

    변환식: output = alpha * input + beta

    Args:
        image: 입력 이미지
        alpha: 대비 조정 (1.0 = 변경 없음, >1.0 = 대비 증가)
        beta: 밝기 조정 (0 = 변경 없음, >0 = 밝게, <0 = 어둡게)

    Returns:
        조정된 이미지

    Note:
        cv.convertScaleAbs()는 자동으로 [0, 255] 범위로 클리핑합니다.
    """
    return cv.convertScaleAbs(image, alpha=alpha, beta=beta)


def plot_histogram(image: np.ndarray, title: str = "Histogram") -> None:
    """이미지의 히스토그램을 그립니다."""
    plt.figure(figsize=(10, 4))

    if image.ndim == 2:
        # 그레이스케일
        hist = compute_histogram(image)
        plt.plot(hist, color="gray")
        plt.fill_between(range(256), hist.ravel(), alpha=0.5, color="gray")
    else:
        # 컬러: 채널별 히스토그램
        colors = [("b", "Blue"), ("g", "Green"), ("r", "Red")]
        for i, (color, label) in enumerate(colors):
            hist = compute_histogram(image, channel=i)
            plt.plot(hist, color=color, label=label, alpha=0.7)
        plt.legend()

    plt.xlabel("픽셀 값 (0–255)")
    plt.ylabel("빈도수")
    plt.title(title)
    plt.xlim([0, 256])
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def compare_equalization_methods(image: np.ndarray) -> None:
    """히스토그램 평활화 방법을 비교합니다."""
    if image.ndim == 3:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        gray = image

    equalized = cv.equalizeHist(gray)
    clahe_result = apply_clahe(gray, clip_limit=2.0, tile_grid_size=(8, 8))

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("히스토그램 평활화 비교", fontsize=14)

    images = [gray, equalized, clahe_result]
    titles = ["원본", "Histogram Equalization", "CLAHE (clip=2.0)"]

    for col, (img, title) in enumerate(zip(images, titles)):
        # 이미지 표시
        axes[0, col].imshow(img, cmap="gray")
        axes[0, col].set_title(title)
        axes[0, col].axis("off")

        # 히스토그램 표시
        hist = compute_histogram(img)
        axes[1, col].plot(hist, color="gray")
        axes[1, col].fill_between(range(256), hist.ravel(), alpha=0.5, color="gray")
        axes[1, col].set_xlim([0, 256])
        axes[1, col].set_xlabel("픽셀 값")
        axes[1, col].set_ylabel("빈도수")
        axes[1, col].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def demo_with_sample_image() -> None:
    """샘플 이미지로 히스토그램 처리를 시연합니다."""
    # 어두운 이미지 생성 (평활화 효과 확인용)
    sample = np.zeros((300, 400), dtype=np.uint8)
    for r in range(300):
        for c in range(400):
            sample[r, c] = int(50 + 80 * (r / 300) * (c / 400))
    # 도형 추가
    cv.rectangle(sample, (50, 50), (150, 150), 120, -1)
    cv.circle(sample, (300, 150), 70, 100, -1)

    print("=== 히스토그램 처리 시연 ===")
    print(f"원본 이미지 평균 밝기: {sample.mean():.1f}")
    print(f"원본 이미지 표준편차: {sample.std():.1f}")

    equalized = cv.equalizeHist(sample)
    print(f"\n평활화 후 평균 밝기: {equalized.mean():.1f}")
    print(f"평활화 후 표준편차: {equalized.std():.1f}")

    compare_equalization_methods(sample)


if __name__ == "__main__":
    demo_with_sample_image()
