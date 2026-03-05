"""
기초 이미지 처리 - 이미지 로드, 저장, 기본 조작
(Basic Image Processing - Load, Save, Basic Operations)

학습 목표:
- cv.imread() 로 이미지 로드 및 오류 처리
- 이미지 속성(크기, 채널, 데이터 타입) 확인
- 크기 조정(resize), 회전(rotate), 뒤집기(flip)
- cv.imwrite() 로 이미지 저장
"""

import cv2 as cv
import numpy as np
import os


def load_image(path: str) -> np.ndarray:
    """이미지를 로드하고 파일 존재 여부를 확인합니다.

    Args:
        path: 이미지 파일 경로

    Returns:
        로드된 이미지 (BGR 형식)

    Raises:
        FileNotFoundError: 파일이 존재하지 않을 때
        ValueError: 이미지 로드에 실패했을 때
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {path}")

    # cv.imread()는 실패 시 None을 반환 — 반드시 확인 필요
    image = cv.imread(path)
    if image is None:
        raise ValueError(f"이미지 로드 실패 (손상된 파일일 수 있습니다): {path}")

    return image


def print_image_info(image: np.ndarray, label: str = "Image") -> None:
    """이미지의 기본 속성을 출력합니다."""
    h, w = image.shape[:2]
    channels = image.shape[2] if image.ndim == 3 else 1
    print(f"[{label}]")
    print(f"  크기(Height x Width): {h} x {w}")
    print(f"  채널 수: {channels}")
    print(f"  데이터 타입: {image.dtype}")
    print(f"  픽셀 값 범위: {image.min()} ~ {image.max()}")


def resize_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """이미지 크기를 조정합니다.

    Args:
        image: 입력 이미지
        width: 목표 너비
        height: 목표 높이

    Returns:
        크기가 조정된 이미지

    Note:
        cv.INTER_AREA   : 축소 시 권장 (앨리어싱 최소화)
        cv.INTER_LINEAR : 확대 시 기본값 (속도와 품질의 균형)
        cv.INTER_CUBIC  : 확대 시 고품질 (더 느림)
    """
    return cv.resize(image, (width, height), interpolation=cv.INTER_LINEAR)


def resize_by_scale(image: np.ndarray, scale: float) -> np.ndarray:
    """비율에 따라 이미지 크기를 조정합니다.

    Args:
        image: 입력 이미지
        scale: 배율 (예: 0.5 = 50%, 2.0 = 200%)

    Returns:
        크기가 조정된 이미지
    """
    h, w = image.shape[:2]
    new_w = int(w * scale)
    new_h = int(h * scale)
    interp = cv.INTER_AREA if scale < 1.0 else cv.INTER_LINEAR
    return cv.resize(image, (new_w, new_h), interpolation=interp)


def rotate_image(image: np.ndarray, angle: float, keep_size: bool = True) -> np.ndarray:
    """이미지를 중심을 기준으로 회전합니다.

    Args:
        image: 입력 이미지
        angle: 회전 각도 (양수 = 반시계 방향)
        keep_size: True이면 원본 크기 유지 (잘림 발생 가능)

    Returns:
        회전된 이미지
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)  # OpenCV 좌표: (x, y) = (열, 행)

    # 2×3 회전 변환 행렬 계산
    rotation_matrix = cv.getRotationMatrix2D(center, angle, scale=1.0)

    if keep_size:
        rotated = cv.warpAffine(image, rotation_matrix, (w, h))
    else:
        # 회전 후 이미지가 잘리지 않도록 캔버스 크기 확장
        cos_val = abs(rotation_matrix[0, 0])
        sin_val = abs(rotation_matrix[0, 1])
        new_w = int(h * sin_val + w * cos_val)
        new_h = int(h * cos_val + w * sin_val)
        rotation_matrix[0, 2] += (new_w - w) / 2
        rotation_matrix[1, 2] += (new_h - h) / 2
        rotated = cv.warpAffine(image, rotation_matrix, (new_w, new_h))

    return rotated


def flip_image(image: np.ndarray, direction: str = "horizontal") -> np.ndarray:
    """이미지를 뒤집습니다.

    Args:
        image: 입력 이미지
        direction: 'horizontal'(좌우), 'vertical'(상하), 'both'(상하좌우)

    Returns:
        뒤집힌 이미지
    """
    flip_codes = {"horizontal": 1, "vertical": 0, "both": -1}
    if direction not in flip_codes:
        raise ValueError(f"direction은 'horizontal', 'vertical', 'both' 중 하나여야 합니다. 입력값: {direction}")
    return cv.flip(image, flip_codes[direction])


def crop_image(image: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray:
    """이미지의 관심 영역(ROI)을 잘라냅니다.

    Args:
        image: 입력 이미지
        x: 시작 열 (OpenCV x 좌표)
        y: 시작 행 (OpenCV y 좌표)
        width: 잘라낼 너비
        height: 잘라낼 높이

    Returns:
        잘라낸 이미지

    Note:
        NumPy 슬라이싱: image[y:y+height, x:x+width]
        OpenCV 좌표 (x,y)는 NumPy 배열에서 (행=y, 열=x)에 해당
    """
    img_h, img_w = image.shape[:2]
    # 경계 초과 방지
    x_end = min(x + width, img_w)
    y_end = min(y + height, img_h)
    return image[y:y_end, x:x_end]


def save_image(image: np.ndarray, path: str) -> bool:
    """이미지를 파일로 저장합니다.

    Args:
        image: 저장할 이미지
        path: 저장 경로 (확장자에 따라 포맷 자동 결정)

    Returns:
        저장 성공 여부
    """
    success = cv.imwrite(path, image)
    if success:
        print(f"이미지 저장 완료: {path}")
    else:
        print(f"이미지 저장 실패: {path}")
    return success


def demo_with_sample_image() -> None:
    """샘플 이미지를 생성하여 기본 조작을 시연합니다."""
    # 테스트용 샘플 이미지 생성 (BGR 형식)
    sample = np.zeros((300, 400, 3), dtype=np.uint8)
    # 색상 영역 추가
    sample[:, :200] = (255, 0, 0)    # 왼쪽: 파란색 (BGR)
    sample[:, 200:] = (0, 0, 255)    # 오른쪽: 빨간색 (BGR)
    cv.rectangle(sample, (150, 100), (250, 200), (0, 255, 0), -1)  # 초록 사각형
    cv.putText(sample, "OpenCV Sample", (50, 270),
               cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    print("=== 이미지 기본 조작 데모 ===\n")
    print_image_info(sample, "원본")

    # 크기 조정
    resized = resize_image(sample, width=200, height=150)
    print_image_info(resized, "크기 조정 후 (200x150)")

    # 비율 조정
    scaled = resize_by_scale(sample, scale=0.5)
    print_image_info(scaled, "50% 축소")

    # 회전
    rotated = rotate_image(sample, angle=45, keep_size=True)
    print_image_info(rotated, "45도 회전")

    # 뒤집기
    flipped = flip_image(sample, direction="horizontal")
    print_image_info(flipped, "좌우 반전")

    # ROI 자르기
    cropped = crop_image(sample, x=100, y=50, width=200, height=150)
    print_image_info(cropped, "ROI 자르기 (100,50)부터 200x150")

    # 결과 나란히 표시
    # 동일한 높이로 맞춰서 합치기
    target_h = 300
    imgs = [sample, resized, scaled, rotated, flipped]
    resized_for_display = [
        cv.resize(img, (int(img.shape[1] * target_h / img.shape[0]), target_h))
        for img in imgs
    ]
    combined = np.hstack(resized_for_display)

    cv.imshow("기본 이미지 조작 결과", combined)
    print("\n아무 키나 누르면 종료됩니다...")
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    demo_with_sample_image()
