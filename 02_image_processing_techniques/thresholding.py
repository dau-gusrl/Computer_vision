"""
임계값 처리 (Thresholding)

학습 목표:
- Binary thresholding의 원리와 종류
- Adaptive thresholding의 필요성과 원리
- Otsu's thresholding (자동 임계값 결정)
- 실제 활용 시나리오 (문서 이진화, 객체 분리)
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def global_threshold(image: np.ndarray, threshold: int = 127,
                     thresh_type: str = "binary") -> tuple:
    """전역 임계값 처리를 적용합니다.

    Args:
        image: 그레이스케일 이미지
        threshold: 임계값 (0–255)
        thresh_type: 임계값 유형

    Returns:
        (사용된 임계값, 이진화된 이미지) 튜플

    임계값 유형:
        binary      : pixel > threshold → 255, else → 0
        binary_inv  : pixel > threshold → 0, else → 255
        trunc       : pixel > threshold → threshold, else → pixel
        tozero      : pixel > threshold → pixel, else → 0
        tozero_inv  : pixel > threshold → 0, else → pixel
    """
    type_map = {
        "binary": cv.THRESH_BINARY,
        "binary_inv": cv.THRESH_BINARY_INV,
        "trunc": cv.THRESH_TRUNC,
        "tozero": cv.THRESH_TOZERO,
        "tozero_inv": cv.THRESH_TOZERO_INV,
    }
    if thresh_type not in type_map:
        raise ValueError(f"thresh_type은 {list(type_map.keys())} 중 하나여야 합니다.")

    if image.ndim == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    ret, binary = cv.threshold(image, threshold, 255, type_map[thresh_type])
    return ret, binary


def otsu_threshold(image: np.ndarray) -> tuple:
    """Otsu 방법으로 최적 임계값을 자동으로 결정합니다.

    원리:
        클래스 내 분산을 최소화(= 클래스 간 분산을 최대화)하는 임계값 선택
        두 개의 픽셀 그룹(배경/전경)이 뚜렷이 구분될 때 효과적

    Returns:
        (자동 결정된 임계값, 이진화된 이미지) 튜플

    Note:
        단봉형(unimodal) 히스토그램에는 효과가 떨어질 수 있습니다.
        이중봉형(bimodal) 히스토그램에 최적입니다.
    """
    if image.ndim == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # 가우시안 블러 후 Otsu 적용 권장 (노이즈 감소)
    blurred = cv.GaussianBlur(image, (5, 5), 0)
    ret, binary = cv.threshold(blurred, 0, 255,
                               cv.THRESH_BINARY + cv.THRESH_OTSU)
    return ret, binary


def adaptive_threshold(image: np.ndarray, block_size: int = 11,
                       c_value: int = 2, method: str = "gaussian") -> np.ndarray:
    """적응형 임계값 처리를 적용합니다.

    전역 임계값의 한계:
        조명이 불균일한 이미지에서는 단일 임계값으로 전체 이미지를 이진화하기 어려움

    원리:
        각 픽셀 주변의 블록(block_size × block_size) 내 가중 평균으로
        지역적 임계값을 계산 → 불균일 조명에 강건

    Args:
        image: 그레이스케일 이미지
        block_size: 지역 임계값 계산 블록 크기 (홀수, 3 이상)
        c_value: 평균에서 뺄 상수 (노이즈 제어)
        method: 'gaussian'(가우시안 가중 평균) 또는 'mean'(단순 평균)

    Returns:
        이진화된 이미지
    """
    if image.ndim == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    adaptive_method = (cv.ADAPTIVE_THRESH_GAUSSIAN_C if method == "gaussian"
                       else cv.ADAPTIVE_THRESH_MEAN_C)

    return cv.adaptiveThreshold(
        image, 255, adaptive_method,
        cv.THRESH_BINARY, block_size, c_value
    )


def compare_threshold_methods(image: np.ndarray) -> None:
    """다양한 임계값 처리 방법을 비교합니다."""
    if image.ndim == 3:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        gray = image

    _, binary_global = global_threshold(gray, 127)
    otsu_val, binary_otsu = otsu_threshold(gray)
    binary_adaptive_mean = adaptive_threshold(gray, method="mean")
    binary_adaptive_gauss = adaptive_threshold(gray, method="gaussian")

    print(f"Otsu 자동 결정 임계값: {otsu_val:.1f}")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("임계값 처리 방법 비교", fontsize=14)

    items = [
        (gray, "원본 그레이스케일"),
        (binary_global, "전역 임계값 (127)"),
        (binary_otsu, f"Otsu 임계값 ({otsu_val:.0f})"),
        (None, ""),  # 빈 칸
        (binary_adaptive_mean, "Adaptive (Mean)"),
        (binary_adaptive_gauss, "Adaptive (Gaussian)"),
    ]

    for ax, (img, title) in zip(axes.flat, items):
        if img is not None:
            ax.imshow(img, cmap="gray")
            ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def demo_with_sample_image() -> None:
    """불균일 조명 이미지로 임계값 처리를 시연합니다."""
    # 불균일 조명 시뮬레이션
    sample = np.zeros((300, 400), dtype=np.uint8)

    # 배경 그라디언트 (불균일 조명 효과)
    for r in range(300):
        for c in range(400):
            sample[r, c] = int(80 * (c / 400))

    # 텍스트/도형 추가
    cv.putText(sample, "OpenCV", (30, 100), cv.FONT_HERSHEY_SIMPLEX, 1.5, 200, 3)
    cv.putText(sample, "Threshold", (30, 170), cv.FONT_HERSHEY_SIMPLEX, 1.2, 200, 3)
    cv.rectangle(sample, (200, 200), (350, 270), 180, 3)

    # 노이즈 추가
    noise = np.random.randint(-10, 10, sample.shape)
    sample = np.clip(sample.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    print("=== 임계값 처리 시연 ===")
    print("불균일 조명 이미지에서 전역 임계값 vs 적응형 임계값 비교")

    compare_threshold_methods(sample)


if __name__ == "__main__":
    demo_with_sample_image()
