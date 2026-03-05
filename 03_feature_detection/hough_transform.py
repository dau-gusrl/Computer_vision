"""
Hough 변환 (Hough Transform)

학습 목표:
- Hough 직선 검출의 수학적 원리
- 표준 Hough 변환 vs 확률적 Hough 변환
- Hough 원 검출 (HoughCircles)
- 파라미터 조정에 따른 결과 변화
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def hough_lines_standard(image: np.ndarray, rho: float = 1,
                          theta: float = np.pi / 180,
                          threshold: int = 100) -> np.ndarray:
    """표준 Hough 직선 변환으로 직선을 검출합니다.

    원리:
        이미지 공간 (x, y) → 파라미터 공간 (ρ, θ)
        각 에지 점에서 가능한 모든 직선을 파라미터 공간에 표현
        → 파라미터 공간에서 교점이 많은 (ρ, θ)가 실제 직선

        직선 방정식: ρ = x·cos(θ) + y·sin(θ)

    Args:
        image: BGR 또는 그레이스케일 이미지
        rho: 거리 분해능 (픽셀 단위)
        theta: 각도 분해능 (라디안 단위)
        threshold: 누산기 임계값 (교점 수 최소값)

    Returns:
        검출된 직선 배열 (ρ, θ) 또는 None
    """
    # Canny 에지 검출 후 Hough 변환 적용
    if image.ndim == 3:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        gray = image
    edges = cv.Canny(gray, 50, 150)
    return cv.HoughLines(edges, rho, theta, threshold)


def hough_lines_probabilistic(image: np.ndarray, rho: float = 1,
                               theta: float = np.pi / 180,
                               threshold: int = 50,
                               min_line_length: int = 50,
                               max_line_gap: int = 10) -> np.ndarray:
    """확률적 Hough 직선 변환으로 선분을 검출합니다.

    표준 Hough와의 차이:
        표준: 무한 직선 (ρ, θ) 반환
        확률적: 실제 선분의 시작/끝 좌표 (x1, y1, x2, y2) 반환
        처리 속도가 빠르고 선분 길이/간격 제어 가능

    Args:
        image: BGR 또는 그레이스케일 이미지
        rho: 거리 분해능
        theta: 각도 분해능
        threshold: 누산기 임계값
        min_line_length: 검출할 최소 선분 길이 (픽셀)
        max_line_gap: 이어진 선분으로 간주할 최대 간격 (픽셀)

    Returns:
        선분 배열 [(x1, y1, x2, y2), ...] 또는 None
    """
    if image.ndim == 3:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        gray = image
    edges = cv.Canny(gray, 50, 150)
    return cv.HoughLinesP(edges, rho, theta, threshold,
                          minLineLength=min_line_length,
                          maxLineGap=max_line_gap)


def draw_hough_lines_standard(image: np.ndarray, lines: np.ndarray,
                               color: tuple = (0, 0, 255)) -> np.ndarray:
    """표준 Hough 직선을 이미지에 그립니다."""
    result = image.copy()
    if lines is None:
        return result

    h, w = result.shape[:2]
    for line in lines:
        rho, theta = line[0]
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a * rho, b * rho
        # 이미지 대각선 길이만큼 연장
        scale = int(np.sqrt(h ** 2 + w ** 2))
        x1 = int(x0 + scale * (-b))
        y1 = int(y0 + scale * a)
        x2 = int(x0 - scale * (-b))
        y2 = int(y0 - scale * a)
        cv.line(result, (x1, y1), (x2, y2), color, 2)

    print(f"표준 Hough: {len(lines)}개 직선 검출")
    return result


def draw_hough_lines_probabilistic(image: np.ndarray, lines: np.ndarray,
                                   color: tuple = (0, 255, 0)) -> np.ndarray:
    """확률적 Hough 선분을 이미지에 그립니다."""
    result = image.copy()
    if lines is None:
        return result

    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(result, (x1, y1), (x2, y2), color, 2)

    print(f"확률적 Hough: {len(lines)}개 선분 검출")
    return result


def hough_circles(image: np.ndarray, dp: float = 1.2,
                  min_dist: int = 30, param1: int = 50,
                  param2: int = 30, min_radius: int = 10,
                  max_radius: int = 100) -> np.ndarray:
    """Hough 원 변환으로 원을 검출합니다.

    원의 방정식: (x - a)² + (y - b)² = r²
    파라미터 공간: 3D (a, b, r)

    Args:
        image: BGR 이미지
        dp: 누산기 해상도와 이미지 해상도 비율 (1 = 동일, 2 = 절반)
        min_dist: 검출된 원 중심 간 최소 거리
        param1: Canny 상한 임계값
        param2: 누산기 임계값 (낮을수록 더 많은 원 검출, 오탐 증가)
        min_radius: 최소 원 반지름 (0이면 제한 없음)
        max_radius: 최대 원 반지름 (0이면 제한 없음)

    Returns:
        원 배열 [(x, y, r), ...] 또는 None
    """
    if image.ndim == 3:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        gray = image

    # 가우시안 블러로 노이즈 제거 (원 검출 전 권장)
    blurred = cv.GaussianBlur(gray, (9, 9), 2)

    circles = cv.HoughCircles(
        blurred, cv.HOUGH_GRADIENT, dp, min_dist,
        param1=param1, param2=param2,
        minRadius=min_radius, maxRadius=max_radius
    )
    return circles


def draw_hough_circles(image: np.ndarray, circles: np.ndarray) -> np.ndarray:
    """검출된 원을 이미지에 그립니다."""
    result = image.copy()
    if circles is None:
        print("원이 검출되지 않았습니다.")
        return result

    circles_int = np.uint16(np.around(circles))
    for circle in circles_int[0]:
        x, y, r = circle
        cv.circle(result, (x, y), r, (0, 255, 0), 2)       # 원
        cv.circle(result, (x, y), 2, (0, 0, 255), 3)        # 중심점

    print(f"Hough 원: {len(circles_int[0])}개 원 검출")
    return result


def demo_with_sample_image() -> None:
    """샘플 이미지로 Hough 변환을 시연합니다."""
    # 직선과 원이 있는 샘플 이미지 생성
    sample = np.zeros((400, 500, 3), dtype=np.uint8)
    cv.line(sample, (50, 100), (450, 100), (200, 200, 200), 2)
    cv.line(sample, (50, 300), (450, 300), (200, 200, 200), 2)
    cv.line(sample, (150, 50), (150, 380), (200, 200, 200), 2)
    cv.line(sample, (50, 50), (450, 380), (180, 180, 180), 2)
    cv.circle(sample, (350, 200), 60, (220, 220, 220), 2)
    cv.circle(sample, (100, 300), 40, (200, 200, 200), 2)
    cv.circle(sample, (400, 320), 50, (210, 210, 210), 2)
    # 노이즈 추가
    noise = np.random.randint(0, 15, sample.shape, dtype=np.uint8)
    sample = cv.add(sample, noise)

    print("=== Hough 변환 시연 ===")

    # 직선 검출
    lines_std = hough_lines_standard(sample, threshold=80)
    lines_prob = hough_lines_probabilistic(sample, threshold=40,
                                           min_line_length=40, max_line_gap=10)
    circles = hough_circles(sample, dp=1.2, min_dist=50,
                             param1=50, param2=25,
                             min_radius=20, max_radius=80)

    img_std = draw_hough_lines_standard(sample, lines_std)
    img_prob = draw_hough_lines_probabilistic(sample, lines_prob)
    img_circles = draw_hough_circles(sample, circles)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("Hough 변환 결과", fontsize=14)

    display = [
        (sample, "원본"),
        (img_std, "표준 Hough 직선\n(무한 직선)"),
        (img_prob, "확률적 Hough 직선\n(선분)"),
        (img_circles, "Hough 원 검출"),
    ]

    for ax, (img, title) in zip(axes, display):
        ax.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    demo_with_sample_image()
