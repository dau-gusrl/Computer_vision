"""
필터링 (Filtering - Blur, Sobel, Canny Edge Detection)

학습 목표:
- 컨볼루션 연산의 원리
- 다양한 블러 필터 비교 (Box, Gaussian, Median, Bilateral)
- 소벨(Sobel) 필터를 이용한 에지 검출
- 캐니(Canny) 에지 검출 알고리즘 이해
- 커널 크기와 결과의 관계
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def apply_box_blur(image: np.ndarray, ksize: int = 5) -> np.ndarray:
    """박스(평균) 블러를 적용합니다.

    원리: 커널 영역의 평균값으로 중심 픽셀 대체
    커널 예시 (3×3):
        1/9 * [[1,1,1],
                [1,1,1],
                [1,1,1]]

    Args:
        image: 입력 이미지
        ksize: 커널 크기 (홀수여야 함)
    """
    return cv.blur(image, (ksize, ksize))


def apply_gaussian_blur(image: np.ndarray, ksize: int = 5, sigma: float = 0) -> np.ndarray:
    """가우시안 블러를 적용합니다.

    원리: 가우시안 분포 가중치로 주변 픽셀의 가중 평균 계산
    특징: 박스 블러보다 자연스러운 결과, 에지 검출 전 노이즈 제거에 주로 사용

    Args:
        image: 입력 이미지
        ksize: 커널 크기 (홀수여야 함)
        sigma: 표준편차 (0이면 ksize로 자동 계산)
    """
    return cv.GaussianBlur(image, (ksize, ksize), sigma)


def apply_median_blur(image: np.ndarray, ksize: int = 5) -> np.ndarray:
    """미디언 블러를 적용합니다.

    원리: 커널 영역 픽셀값들의 중앙값으로 대체
    특징: Salt-and-Pepper 노이즈 제거에 효과적, 에지를 비교적 잘 보존

    Args:
        image: 입력 이미지
        ksize: 커널 크기 (홀수여야 함)
    """
    return cv.medianBlur(image, ksize)


def apply_bilateral_filter(image: np.ndarray, d: int = 9,
                            sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
    """양방향(Bilateral) 필터를 적용합니다.

    원리: 공간적 거리와 색상 유사도를 모두 고려한 가중 평균
    특징: 에지를 보존하면서 노이즈 제거 (Edge-Preserving Smoothing)
    단점: 다른 블러에 비해 처리 속도가 느림

    Args:
        image: 입력 이미지
        d: 픽셀 이웃 지름
        sigma_color: 색상 공간 표준편차 (클수록 넓은 색상 범위 혼합)
        sigma_space: 좌표 공간 표준편차 (클수록 먼 픽셀도 영향)
    """
    return cv.bilateralFilter(image, d, sigma_color, sigma_space)


def apply_sobel(image: np.ndarray, direction: str = "both",
                ksize: int = 3) -> np.ndarray:
    """소벨 필터로 에지를 검출합니다.

    원리:
        Sobel-X (수평 에지, 수직 방향 변화):
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]]
        Sobel-Y (수직 에지, 수평 방향 변화):
            [[-1,-2,-1],
             [ 0, 0, 0],
             [ 1, 2, 1]]
        에지 크기: |Gx| + |Gy| 또는 sqrt(Gx² + Gy²)

    Args:
        image: BGR 또는 그레이스케일 입력 이미지
        direction: 'x', 'y', 'both' 중 하나
        ksize: 소벨 커널 크기 (1, 3, 5, 7)

    Returns:
        에지 이미지 (uint8)
    """
    if image.ndim == 3:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        gray = image

    # float64로 계산 후 절댓값 변환 (음수 에지도 검출)
    if direction in ("x", "both"):
        sobel_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=ksize)
        sobel_x = cv.convertScaleAbs(sobel_x)
    if direction in ("y", "both"):
        sobel_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=ksize)
        sobel_y = cv.convertScaleAbs(sobel_y)

    if direction == "x":
        return sobel_x
    if direction == "y":
        return sobel_y
    # both: 두 방향 합산
    return cv.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)


def apply_canny(image: np.ndarray, threshold1: float = 50,
                threshold2: float = 150) -> np.ndarray:
    """캐니 에지 검출을 적용합니다.

    알고리즘 단계:
        1. 가우시안 블러로 노이즈 제거
        2. 소벨 필터로 기울기(gradient) 크기와 방향 계산
        3. Non-Maximum Suppression: 에지 방향으로 극대점만 유지
        4. Double Thresholding:
            - 강한 에지: threshold2 이상
            - 약한 에지: threshold1 ~ threshold2
            - 노이즈: threshold1 미만
        5. Hysteresis: 강한 에지에 연결된 약한 에지만 최종 에지로 인정

    Args:
        image: BGR 또는 그레이스케일 입력 이미지
        threshold1: 히스테리시스 하한 임계값
        threshold2: 히스테리시스 상한 임계값 (threshold1의 2–3배 권장)

    Returns:
        이진 에지 이미지
    """
    if image.ndim == 3:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        gray = image

    # 캐니 전 가우시안 블러 권장 (노이즈 감소)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    return cv.Canny(blurred, threshold1, threshold2)


def compare_blur_filters(image: np.ndarray) -> None:
    """다양한 블러 필터를 비교 시각화합니다."""
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) if image.ndim == 3 else image

    results = {
        "원본": gray,
        "Box Blur (5×5)": apply_box_blur(gray, ksize=5),
        "Gaussian Blur (5×5)": apply_gaussian_blur(gray, ksize=5),
        "Median Blur (5)": apply_median_blur(gray, ksize=5),
        "Bilateral Filter": apply_bilateral_filter(gray),
    }

    fig, axes = plt.subplots(1, len(results), figsize=(20, 4))
    fig.suptitle("블러 필터 비교", fontsize=14)

    for ax, (title, img) in zip(axes, results.items()):
        ax.imshow(img, cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def compare_edge_detectors(image: np.ndarray) -> None:
    """소벨과 캐니 에지 검출 결과를 비교합니다."""
    results = {
        "원본 그레이스케일": cv.cvtColor(image, cv.COLOR_BGR2GRAY) if image.ndim == 3 else image,
        "Sobel X": apply_sobel(image, direction="x"),
        "Sobel Y": apply_sobel(image, direction="y"),
        "Sobel X+Y": apply_sobel(image, direction="both"),
        "Canny (50, 150)": apply_canny(image, 50, 150),
    }

    fig, axes = plt.subplots(1, len(results), figsize=(20, 4))
    fig.suptitle("에지 검출 비교", fontsize=14)

    for ax, (title, img) in zip(axes, results.items()):
        ax.imshow(img, cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def demo_with_sample_image() -> None:
    """샘플 이미지로 필터링을 시연합니다."""
    # 에지 검출에 적합한 기하학적 샘플 이미지 생성
    sample = np.zeros((300, 400, 3), dtype=np.uint8)
    cv.rectangle(sample, (50, 50), (180, 180), (200, 200, 200), -1)
    cv.circle(sample, (300, 150), 80, (150, 150, 150), -1)
    cv.line(sample, (0, 250), (400, 250), (255, 255, 255), 3)
    # 노이즈 추가 (필터 효과 비교를 위해)
    noise = np.random.randint(0, 30, sample.shape, dtype=np.uint8)
    sample = cv.add(sample, noise)

    print("=== 필터링 시연 ===")
    compare_blur_filters(sample)
    compare_edge_detectors(sample)


if __name__ == "__main__":
    demo_with_sample_image()
