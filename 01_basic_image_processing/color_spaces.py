"""
색상 공간 변환 (Color Space Conversion)

학습 목표:
- RGB, BGR, Grayscale, HSV 각 색상 공간의 목적과 특징 이해
- cv.cvtColor() 플래그와 사용 방법
- 역변환 가능 여부 및 정보 손실 이해
- 채널 순서의 중요성 (OpenCV = BGR, matplotlib = RGB)
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def explain_color_spaces() -> None:
    """각 색상 공간의 특징을 설명합니다."""
    info = {
        "BGR": "OpenCV 기본 색상 공간. Blue-Green-Red 순서. 일반 이미지 처리.",
        "RGB": "사람에게 직관적인 표현. matplotlib 등 대부분의 라이브러리가 사용.",
        "Grayscale": "단일 채널(밝기만). 메모리 효율적. 색상 정보 필요 없는 처리에 사용.",
        "HSV": "Hue(색상)-Saturation(채도)-Value(명도). 색상 기반 마스킹/추적에 유리.",
        "LAB": "인간 시각에 균일한 색상 공간. 색상 비교 및 색상 전이에 활용.",
        "YCrCb": "밝기(Y)와 색차(Cr, Cb) 분리. 이미지 압축(JPEG)에 활용.",
    }
    print("=== 색상 공간 설명 ===")
    for space, desc in info.items():
        print(f"  {space:10s}: {desc}")


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """BGR 이미지를 RGB로 변환합니다.

    Note:
        OpenCV는 BGR을 기본으로 사용하지만 matplotlib은 RGB를 사용합니다.
        표시 전 반드시 변환이 필요합니다.
    """
    return cv.cvtColor(image, cv.COLOR_BGR2RGB)


def bgr_to_grayscale(image: np.ndarray) -> np.ndarray:
    """BGR 이미지를 그레이스케일로 변환합니다.

    Note:
        변환 공식: Gray = 0.114·B + 0.587·G + 0.299·R
        사람의 눈은 녹색(G)에 가장 민감하므로 가중치가 가장 큽니다.
        역변환(Grayscale → BGR) 시 색상 정보는 복원되지 않습니다.
    """
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)


def bgr_to_hsv(image: np.ndarray) -> np.ndarray:
    """BGR 이미지를 HSV로 변환합니다.

    Note:
        OpenCV HSV 범위:
          H (Hue)        : 0–179  (일반적인 0–360을 절반으로 압축)
          S (Saturation) : 0–255
          V (Value)      : 0–255
    """
    return cv.cvtColor(image, cv.COLOR_BGR2HSV)


def hsv_color_mask(image: np.ndarray, lower_hsv: np.ndarray, upper_hsv: np.ndarray) -> np.ndarray:
    """HSV 범위를 기반으로 특정 색상을 마스킹합니다.

    Args:
        image: BGR 입력 이미지
        lower_hsv: HSV 하한값 배열 (예: np.array([35, 50, 50]) for 초록색)
        upper_hsv: HSV 상한값 배열 (예: np.array([85, 255, 255]) for 초록색)

    Returns:
        마스킹된 이미지 (BGR)

    Example:
        # 초록색 객체 추출
        lower = np.array([35, 50, 50])
        upper = np.array([85, 255, 255])
        result = hsv_color_mask(image, lower, upper)
    """
    hsv = bgr_to_hsv(image)
    mask = cv.inRange(hsv, lower_hsv, upper_hsv)
    result = cv.bitwise_and(image, image, mask=mask)
    return result


def split_channels(image: np.ndarray) -> tuple:
    """이미지의 채널을 분리합니다.

    Returns:
        BGR 이미지: (B, G, R) 튜플
    """
    return cv.split(image)


def merge_channels(b: np.ndarray, g: np.ndarray, r: np.ndarray) -> np.ndarray:
    """개별 채널을 병합하여 BGR 이미지를 만듭니다."""
    return cv.merge([b, g, r])


def visualize_color_spaces(image: np.ndarray) -> None:
    """다양한 색상 공간으로 변환한 결과를 matplotlib으로 시각화합니다."""
    gray = bgr_to_grayscale(image)
    hsv = bgr_to_hsv(image)
    image_rgb = bgr_to_rgb(image)

    b, g, r = split_channels(image)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("색상 공간 시각화", fontsize=14)

    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title("원본 (RGB 표시)")

    axes[0, 1].imshow(gray, cmap="gray")
    axes[0, 1].set_title("Grayscale")

    axes[0, 2].imshow(hsv[:, :, 0], cmap="hsv")
    axes[0, 2].set_title("HSV - H (색상)")

    axes[0, 3].imshow(hsv[:, :, 1], cmap="gray")
    axes[0, 3].set_title("HSV - S (채도)")

    axes[1, 0].imshow(b, cmap="Blues")
    axes[1, 0].set_title("Blue 채널")

    axes[1, 1].imshow(g, cmap="Greens")
    axes[1, 1].set_title("Green 채널")

    axes[1, 2].imshow(r, cmap="Reds")
    axes[1, 2].set_title("Red 채널")

    axes[1, 3].imshow(hsv[:, :, 2], cmap="gray")
    axes[1, 3].set_title("HSV - V (명도)")

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def demo_with_sample_image() -> None:
    """샘플 이미지로 색상 공간 변환을 시연합니다."""
    explain_color_spaces()
    print()

    # 컬러 그라디언트 샘플 이미지 생성
    sample = np.zeros((300, 400, 3), dtype=np.uint8)
    for i in range(400):
        # BGR로 색상 그라디언트 생성
        hue = int(i / 400 * 179)
        color_hsv = np.uint8([[[hue, 255, 200]]])
        color_bgr = cv.cvtColor(color_hsv, cv.COLOR_HSV2BGR)[0, 0]
        sample[:, i] = color_bgr

    print("=== 색상 공간 변환 시연 ===")

    gray = bgr_to_grayscale(sample)
    hsv = bgr_to_hsv(sample)

    print(f"원본 shape: {sample.shape}, dtype: {sample.dtype}")
    print(f"Grayscale shape: {gray.shape}  ← 채널 1개로 줄어듦")
    print(f"HSV shape: {hsv.shape}  ← 동일한 shape, 값의 의미만 다름")
    print(f"\nHSV H 채널 범위: {hsv[:,:,0].min()} ~ {hsv[:,:,0].max()} (OpenCV: 0–179)")

    # 초록색 마스킹 예시
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    masked = hsv_color_mask(sample, lower_green, upper_green)
    print(f"\n초록색 마스킹 완료 (HSV H: {lower_green[0]}–{upper_green[0]})")

    visualize_color_spaces(sample)


if __name__ == "__main__":
    demo_with_sample_image()
