"""
모폴로지 연산 (Morphological Operations)

학습 목표:
- Erosion, Dilation의 원리와 효과
- Opening, Closing 연산 (Erosion/Dilation 조합)
- 다양한 구조 요소(SE) 형태 이해
- 실제 활용 시나리오 (노이즈 제거, 윤곽선 추출)
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def create_structuring_element(shape: str = "rect", ksize: int = 5) -> np.ndarray:
    """구조 요소(Structuring Element)를 생성합니다.

    Args:
        shape: 'rect'(사각형), 'ellipse'(타원), 'cross'(십자) 중 하나
        ksize: 커널 크기

    Returns:
        구조 요소 행렬

    Note:
        구조 요소 형태에 따라 모폴로지 결과가 달라집니다.
        - rect   : 모든 방향에 균등하게 적용
        - ellipse: 원형 객체에 적합
        - cross  : 수평/수직 방향으로만 적용
    """
    shape_map = {
        "rect": cv.MORPH_RECT,
        "ellipse": cv.MORPH_ELLIPSE,
        "cross": cv.MORPH_CROSS,
    }
    if shape not in shape_map:
        raise ValueError(f"shape는 'rect', 'ellipse', 'cross' 중 하나여야 합니다. 입력값: {shape}")
    return cv.getStructuringElement(shape_map[shape], (ksize, ksize))


def erode(image: np.ndarray, ksize: int = 5, iterations: int = 1) -> np.ndarray:
    """침식(Erosion) 연산을 적용합니다.

    원리: 구조 요소가 완전히 포함되는 위치만 1로 유지
    효과:
        - 흰색 영역 축소 (검은 영역 확장)
        - 작은 흰색 노이즈 제거
        - 객체들 사이의 연결 분리

    Args:
        image: 이진 이미지 (또는 그레이스케일)
        ksize: 구조 요소 크기
        iterations: 반복 횟수 (많을수록 효과 강화)
    """
    kernel = create_structuring_element("rect", ksize)
    return cv.erode(image, kernel, iterations=iterations)


def dilate(image: np.ndarray, ksize: int = 5, iterations: int = 1) -> np.ndarray:
    """팽창(Dilation) 연산을 적용합니다.

    원리: 구조 요소가 한 픽셀이라도 포함되면 1로 설정
    효과:
        - 흰색 영역 확장 (검은 영역 축소)
        - 작은 구멍(hole) 메우기
        - 끊어진 에지 연결

    Args:
        image: 이진 이미지 (또는 그레이스케일)
        ksize: 구조 요소 크기
        iterations: 반복 횟수
    """
    kernel = create_structuring_element("rect", ksize)
    return cv.dilate(image, kernel, iterations=iterations)


def opening(image: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Opening 연산을 적용합니다 (Erosion → Dilation).

    효과:
        - 작은 흰색 노이즈 제거 (foreground 기준)
        - 객체 크기와 형태를 대체로 보존

    Note:
        Opening = Erosion 후 Dilation
        cv.morphologyEx(image, cv.MORPH_OPEN, kernel) 과 동일
    """
    kernel = create_structuring_element("rect", ksize)
    return cv.morphologyEx(image, cv.MORPH_OPEN, kernel)


def closing(image: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Closing 연산을 적용합니다 (Dilation → Erosion).

    효과:
        - 객체 내부의 작은 구멍 메우기
        - 끊어진 윤곽선 연결

    Note:
        Closing = Dilation 후 Erosion
        cv.morphologyEx(image, cv.MORPH_CLOSE, kernel) 과 동일
    """
    kernel = create_structuring_element("rect", ksize)
    return cv.morphologyEx(image, cv.MORPH_CLOSE, kernel)


def morphological_gradient(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """모폴로지 그래디언트를 계산합니다 (Dilation - Erosion).

    효과: 객체의 윤곽선(경계) 추출
    """
    kernel = create_structuring_element("rect", ksize)
    return cv.morphologyEx(image, cv.MORPH_GRADIENT, kernel)


def top_hat(image: np.ndarray, ksize: int = 15) -> np.ndarray:
    """Top Hat 변환을 적용합니다 (원본 - Opening).

    효과:
        - 구조 요소보다 작은 밝은 영역 강조
        - 불균일한 배경에서 밝은 객체 검출에 유용
    """
    kernel = create_structuring_element("rect", ksize)
    return cv.morphologyEx(image, cv.MORPH_TOPHAT, kernel)


def black_hat(image: np.ndarray, ksize: int = 15) -> np.ndarray:
    """Black Hat 변환을 적용합니다 (Closing - 원본).

    효과:
        - 구조 요소보다 작은 어두운 영역 강조
        - 불균일한 배경에서 어두운 객체 검출에 유용
    """
    kernel = create_structuring_element("rect", ksize)
    return cv.morphologyEx(image, cv.MORPH_BLACKHAT, kernel)


def visualize_morphology(binary_image: np.ndarray) -> None:
    """모폴로지 연산 결과를 비교 시각화합니다."""
    results = {
        "원본 이진 이미지": binary_image,
        "Erosion (침식)": erode(binary_image, ksize=5),
        "Dilation (팽창)": dilate(binary_image, ksize=5),
        "Opening (노이즈 제거)": opening(binary_image, ksize=5),
        "Closing (구멍 메우기)": closing(binary_image, ksize=5),
        "Gradient (윤곽선)": morphological_gradient(binary_image, ksize=3),
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("모폴로지 연산 비교", fontsize=14)

    for ax, (title, img) in zip(axes.flat, results.items()):
        ax.imshow(img, cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def demo_with_sample_image() -> None:
    """샘플 이진 이미지로 모폴로지 연산을 시연합니다."""
    # 노이즈가 있는 이진 이미지 생성
    binary = np.zeros((300, 400), dtype=np.uint8)

    # 기본 도형
    cv.rectangle(binary, (50, 50), (180, 180), 255, -1)
    cv.circle(binary, (300, 150), 80, 255, -1)

    # 구멍(hole) 추가
    cv.circle(binary, (115, 115), 20, 0, -1)
    cv.rectangle(binary, (270, 120), (330, 180), 0, -1)

    # 노이즈 추가 (작은 점들)
    rng = np.random.default_rng(42)
    noise_coords = rng.integers(0, [400, 300], size=(200, 2))
    for x, y in noise_coords:
        binary[y, x] = 255

    # 객체 주변의 끊어진 에지 시뮬레이션
    cv.line(binary, (200, 0), (200, 300), 255, 2)
    erode_mask = np.random.randint(0, 2, binary.shape, dtype=np.uint8) * 255
    binary = cv.bitwise_and(binary, erode_mask)
    binary = cv.bitwise_or(binary, cv.rectangle(np.zeros_like(binary), (50, 50), (180, 180), 255, -1))
    binary = cv.bitwise_or(binary, cv.circle(np.zeros_like(binary), (300, 150), 80, 255, -1))

    # 재생성 (명확한 예시를 위해)
    binary = np.zeros((300, 400), dtype=np.uint8)
    cv.rectangle(binary, (50, 50), (180, 180), 255, -1)
    cv.circle(binary, (300, 150), 80, 255, -1)
    cv.circle(binary, (115, 115), 20, 0, -1)   # 내부 구멍
    # 랜덤 노이즈 추가
    noise = np.zeros_like(binary)
    rng = np.random.default_rng(42)
    noise_pts = rng.integers(0, [400, 300], size=(300, 2))
    for x, y in noise_pts:
        noise[y, x] = 255
    binary_noisy = cv.bitwise_or(binary, noise)

    print("=== 모폴로지 연산 시연 ===")
    print(f"이진 이미지 shape: {binary_noisy.shape}, dtype: {binary_noisy.dtype}")
    print("Opening: Erosion → Dilation (작은 노이즈 제거)")
    print("Closing: Dilation → Erosion (내부 구멍 메우기)")

    visualize_morphology(binary_noisy)


if __name__ == "__main__":
    demo_with_sample_image()
