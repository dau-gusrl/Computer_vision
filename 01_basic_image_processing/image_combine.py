"""
이미지 결합 및 표시 (Image Combining and Display)

학습 목표:
- np.hstack(), np.vstack()을 이용한 이미지 결합
- 배열 크기 일치의 중요성 이해
- matplotlib과 OpenCV의 표시 방법 비교
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def hstack_images(images: list, target_height: int = None) -> np.ndarray:
    """이미지들을 수평으로 결합합니다.

    Args:
        images: BGR 이미지 리스트
        target_height: 통일할 높이 (None이면 첫 번째 이미지 기준)

    Returns:
        수평으로 결합된 이미지

    Note:
        np.hstack()은 배열의 높이(행 수)가 동일해야 합니다.
        크기가 다른 이미지는 먼저 리사이즈 후 결합해야 합니다.
    """
    if not images:
        raise ValueError("이미지 리스트가 비어 있습니다.")

    h = target_height if target_height else images[0].shape[0]
    resized = []
    for img in images:
        img_h, img_w = img.shape[:2]
        new_w = int(img_w * h / img_h)
        resized.append(cv.resize(img, (new_w, h)))

    return np.hstack(resized)


def vstack_images(images: list, target_width: int = None) -> np.ndarray:
    """이미지들을 수직으로 결합합니다.

    Args:
        images: BGR 이미지 리스트
        target_width: 통일할 너비 (None이면 첫 번째 이미지 기준)

    Returns:
        수직으로 결합된 이미지
    """
    if not images:
        raise ValueError("이미지 리스트가 비어 있습니다.")

    w = target_width if target_width else images[0].shape[1]
    resized = []
    for img in images:
        img_h, img_w = img.shape[:2]
        new_h = int(img_h * w / img_w)
        resized.append(cv.resize(img, (w, new_h)))

    return np.vstack(resized)


def make_grid(images: list, cols: int, target_size: tuple = (200, 200)) -> np.ndarray:
    """이미지들을 그리드 형태로 배치합니다.

    Args:
        images: BGR 이미지 리스트
        cols: 열 수
        target_size: 각 이미지 크기 (width, height)

    Returns:
        그리드 이미지
    """
    if not images:
        raise ValueError("이미지 리스트가 비어 있습니다.")

    tw, th = target_size
    rows = (len(images) + cols - 1) // cols  # 올림 나눗셈

    # 부족한 칸을 빈 이미지로 채움
    padded = list(images)
    while len(padded) < rows * cols:
        # 채널 수에 맞게 빈 이미지 생성
        channels = images[0].shape[2] if images[0].ndim == 3 else 1
        if channels > 1:
            padded.append(np.zeros((th, tw, channels), dtype=np.uint8))
        else:
            padded.append(np.zeros((th, tw), dtype=np.uint8))

    row_imgs = []
    for r in range(rows):
        row_cells = [cv.resize(padded[r * cols + c], (tw, th)) for c in range(cols)]
        row_imgs.append(np.hstack(row_cells))

    return np.vstack(row_imgs)


def add_label(image: np.ndarray, text: str, position: tuple = None,
              font_scale: float = 0.6, color: tuple = (255, 255, 255),
              thickness: int = 1) -> np.ndarray:
    """이미지에 레이블 텍스트를 추가합니다.

    Args:
        image: 입력 이미지
        text: 표시할 텍스트
        position: (x, y) 좌표 (None이면 좌하단)
        font_scale: 폰트 크기
        color: 텍스트 색상 (BGR)
        thickness: 선 두께

    Returns:
        레이블이 추가된 이미지 사본
    """
    result = image.copy()
    h, w = result.shape[:2]

    if position is None:
        position = (5, h - 10)

    # 가독성을 위한 그림자 효과
    cv.putText(result, text, (position[0] + 1, position[1] + 1),
               cv.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 1)
    cv.putText(result, text, position,
               cv.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

    return result


def show_with_matplotlib(image: np.ndarray, title: str = "", cmap: str = None) -> None:
    """matplotlib으로 이미지를 표시합니다.

    Note:
        OpenCV는 BGR 순서이므로 matplotlib 표시 전 RGB로 변환해야 합니다.
        그레이스케일 이미지는 cmap='gray'를 지정해야 올바르게 표시됩니다.
    """
    plt.figure(figsize=(8, 6))
    if image.ndim == 2:
        # 그레이스케일
        plt.imshow(image, cmap="gray")
    elif cmap:
        plt.imshow(image, cmap=cmap)
    else:
        # BGR → RGB 변환 필수
        plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def demo_image_combining() -> None:
    """이미지 결합 방법을 시연합니다."""
    # 다양한 색상의 샘플 이미지 생성
    colors_bgr = [
        (255, 0, 0, "Blue"),
        (0, 255, 0, "Green"),
        (0, 0, 255, "Red"),
        (255, 255, 0, "Cyan"),
        (0, 255, 255, "Yellow"),
        (255, 0, 255, "Magenta"),
    ]

    images = []
    for b, g, r, name in colors_bgr:
        img = np.full((150, 150, 3), (b, g, r), dtype=np.uint8)
        img = add_label(img, name)
        images.append(img)

    print("=== 이미지 결합 시연 ===\n")

    # 수평 결합
    h_combined = hstack_images(images[:3], target_height=150)
    print(f"수평 결합 (3개): shape = {h_combined.shape}")

    # 수직 결합
    v_combined = vstack_images(images[:3], target_width=150)
    print(f"수직 결합 (3개): shape = {v_combined.shape}")

    # 그리드 배치
    grid = make_grid(images, cols=3, target_size=(150, 150))
    print(f"그리드 배치 (2×3): shape = {grid.shape}")

    # 결과 표시
    plt.figure(figsize=(14, 8))

    plt.subplot(1, 3, 1)
    plt.imshow(cv.cvtColor(h_combined, cv.COLOR_BGR2RGB))
    plt.title("np.hstack() - 수평 결합")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(cv.cvtColor(v_combined, cv.COLOR_BGR2RGB))
    plt.title("np.vstack() - 수직 결합")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(cv.cvtColor(grid, cv.COLOR_BGR2RGB))
    plt.title("make_grid() - 그리드 배치")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    demo_image_combining()
