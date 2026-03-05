"""
Harris 코너 검출 (Harris Corner Detection)

학습 목표:
- Harris 코너 검출의 수학적 원리
- 코너/에지/평탄 영역 구분
- cv.cornerHarris() 파라미터 이해
- Shi-Tomasi 코너 검출과의 비교
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def harris_corner_detection(image: np.ndarray, block_size: int = 2,
                             ksize: int = 3, k: float = 0.04) -> np.ndarray:
    """Harris 코너 검출을 수행합니다.

    수학적 원리:
        픽셀 (x, y)에서 작은 윈도우를 이동할 때의 강도 변화:
        E(u, v) = Σ w(x,y) · [I(x+u, y+v) - I(x,y)]²

        M = Σ w(x,y) · [[Ix², Ix·Iy],
                         [Ix·Iy, Iy²]]

        R = det(M) - k · trace(M)²
          = λ1·λ2 - k·(λ1 + λ2)²

        R 해석:
            R >> 0  → 코너 (두 고유값 모두 큼)
            R << 0  → 에지 (한 고유값만 큼)
            |R| ≈ 0 → 평탄 영역 (두 고유값 모두 작음)

    Args:
        image: 입력 이미지
        block_size: 코너 검출 이웃 크기
        ksize: 소벨 연산자 aperture 크기
        k: Harris 민감도 파라미터 (일반적으로 0.04–0.06)

    Returns:
        각 픽셀의 Harris 반응값 (float32 행렬)
    """
    if image.ndim == 3:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        gray = image

    gray_float = np.float32(gray)
    return cv.cornerHarris(gray_float, block_size, ksize, k)


def draw_harris_corners(image: np.ndarray, harris_response: np.ndarray,
                        threshold_ratio: float = 0.01) -> np.ndarray:
    """Harris 반응값 기반으로 코너를 이미지에 표시합니다.

    Args:
        image: 원본 이미지 (BGR)
        harris_response: cornerHarris() 반응값 행렬
        threshold_ratio: 최대 반응값 대비 임계값 비율

    Returns:
        코너가 표시된 이미지
    """
    result = image.copy()

    # 반응값 팽창으로 극대점 강조
    dilated = cv.dilate(harris_response, None)

    # 임계값 이상인 픽셀 = 코너
    threshold = threshold_ratio * harris_response.max()
    corners = (harris_response > threshold) & (harris_response == dilated)

    # 코너를 빨간 점으로 표시
    result[corners] = [0, 0, 255]  # BGR: 빨간색

    n_corners = np.sum(corners)
    print(f"검출된 코너 수: {n_corners}개 (임계값 비율: {threshold_ratio})")

    return result


def shi_tomasi_corner_detection(image: np.ndarray, max_corners: int = 50,
                                quality_level: float = 0.01,
                                min_distance: float = 10) -> list:
    """Shi-Tomasi 코너 검출을 수행합니다.

    Harris와의 차이:
        Harris: R = det(M) - k·trace(M)²
        Shi-Tomasi: R = min(λ1, λ2)
        → 더 안정적인 코너 검출

    Args:
        image: BGR 이미지
        max_corners: 검출할 최대 코너 수
        quality_level: 최대 코너 품질 대비 최소 비율 (0.0–1.0)
        min_distance: 코너 간 최소 거리 (픽셀)

    Returns:
        코너 좌표 리스트 [(x, y), ...]
    """
    if image.ndim == 3:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        gray = image

    corners = cv.goodFeaturesToTrack(
        gray, max_corners, quality_level, min_distance
    )

    if corners is not None:
        # shape: (N, 1, 2) → [(x, y), ...]
        return [(int(pt[0][0]), int(pt[0][1])) for pt in corners]
    return []


def draw_shi_tomasi_corners(image: np.ndarray, corners: list,
                             radius: int = 5, color: tuple = (0, 255, 0)) -> np.ndarray:
    """Shi-Tomasi 코너를 이미지에 그립니다."""
    result = image.copy()
    for x, y in corners:
        cv.circle(result, (x, y), radius, color, -1)
    return result


def visualize_harris_response(image: np.ndarray) -> None:
    """Harris 반응값 맵을 시각화합니다."""
    if image.ndim == 3:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
        image = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)

    response = harris_corner_detection(image)
    corners_img = draw_harris_corners(image, response, threshold_ratio=0.01)

    # 반응값 정규화 (시각화용)
    response_normalized = cv.normalize(response, None, 0, 255,
                                       cv.NORM_MINMAX, dtype=cv.CV_8U)

    st_corners = shi_tomasi_corner_detection(image, max_corners=50)
    st_img = draw_shi_tomasi_corners(image, st_corners)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle("Harris 코너 검출", fontsize=14)

    axes[0].imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    axes[0].set_title("원본 이미지")

    axes[1].imshow(response_normalized, cmap="hot")
    axes[1].set_title("Harris 반응값 맵\n(밝을수록 코너 가능성 높음)")

    axes[2].imshow(cv.cvtColor(corners_img, cv.COLOR_BGR2RGB))
    axes[2].set_title("Harris 코너 (빨간 점)")

    axes[3].imshow(cv.cvtColor(st_img, cv.COLOR_BGR2RGB))
    axes[3].set_title(f"Shi-Tomasi 코너\n({len(st_corners)}개, 초록 점)")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def demo_with_sample_image() -> None:
    """샘플 이미지로 Harris 코너 검출을 시연합니다."""
    # 코너가 풍부한 샘플 이미지 생성
    sample = np.zeros((300, 400, 3), dtype=np.uint8)
    cv.rectangle(sample, (50, 50), (180, 180), (200, 200, 200), 2)
    cv.rectangle(sample, (200, 60), (360, 160), (150, 150, 150), 2)
    cv.line(sample, (50, 230), (360, 230), (180, 180, 180), 2)
    cv.line(sample, (200, 50), (200, 270), (180, 180, 180), 2)
    cv.circle(sample, (100, 260), 30, (160, 160, 160), 2)

    print("=== Harris 코너 검출 시연 ===")
    print("R >> 0: 코너 | R << 0: 에지 | |R| ≈ 0: 평탄")

    visualize_harris_response(sample)


if __name__ == "__main__":
    demo_with_sample_image()
