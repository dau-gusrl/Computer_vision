"""
경계 상자 시각화 유틸리티 (Bounding Box Visualization Utilities)

학습 목표:
- 다양한 형식의 경계 상자 표현 이해
- IoU(Intersection over Union) 계산
- NMS(Non-Maximum Suppression) 원리
- 시각적으로 명확한 탐지 결과 표시
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


# 클래스별 색상 팔레트 (재현 가능한 색상 생성)
def get_color_for_class(class_id: int) -> tuple:
    """클래스 ID에 따라 고정된 색상을 반환합니다.

    Args:
        class_id: 클래스 인덱스

    Returns:
        BGR 색상 튜플
    """
    # 색상환을 균등하게 분배
    hue = int((class_id * 37) % 180)
    color_hsv = np.uint8([[[hue, 220, 220]]])
    color_bgr = cv.cvtColor(color_hsv, cv.COLOR_HSV2BGR)[0, 0]
    return tuple(int(c) for c in color_bgr)


def draw_bounding_box(image: np.ndarray, x: int, y: int,
                      w: int, h: int, label: str = "",
                      confidence: float = None,
                      color: tuple = (0, 255, 0),
                      thickness: int = 2) -> np.ndarray:
    """단일 경계 상자를 이미지에 그립니다.

    Args:
        image: BGR 이미지
        x, y: 좌상단 좌표
        w, h: 너비, 높이
        label: 클래스 이름
        confidence: 신뢰도 점수 (0.0–1.0)
        color: BGR 색상
        thickness: 선 두께

    Returns:
        경계 상자가 그려진 이미지 사본
    """
    result = image.copy()

    # 경계 상자 사각형
    cv.rectangle(result, (x, y), (x + w, y + h), color, thickness)

    # 레이블 텍스트 구성
    if label or confidence is not None:
        text = label
        if confidence is not None:
            text = f"{label}: {confidence:.2f}" if label else f"{confidence:.2f}"

        # 텍스트 배경 계산
        (text_w, text_h), baseline = cv.getTextSize(
            text, cv.FONT_HERSHEY_SIMPLEX, 0.6, 1
        )
        label_y = max(y - 5, text_h + 5)

        # 배경 사각형 (가독성 향상)
        cv.rectangle(result,
                     (x, label_y - text_h - baseline),
                     (x + text_w, label_y + baseline),
                     color, -1)

        # 텍스트 색상: 배경 밝기에 따라 흰색/검은색 자동 선택
        brightness = 0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2]
        text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)
        cv.putText(result, text, (x, label_y),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)

    return result


def draw_detections_batch(image: np.ndarray, detections: list) -> np.ndarray:
    """여러 탐지 결과를 한 번에 이미지에 그립니다.

    Args:
        image: BGR 이미지
        detections: 탐지 결과 리스트
            각 항목: {'bbox': (x, y, w, h), 'label': str,
                      'confidence': float, 'class_id': int}

    Returns:
        탐지 결과가 표시된 이미지
    """
    result = image.copy()
    for det in detections:
        x, y, w, h = det["bbox"]
        label = det.get("label", "")
        confidence = det.get("confidence")
        class_id = det.get("class_id", 0)
        color = get_color_for_class(class_id)
        result = draw_bounding_box(result, x, y, w, h, label, confidence, color)

    return result


def compute_iou(box1: tuple, box2: tuple) -> float:
    """두 경계 상자의 IoU를 계산합니다.

    IoU (Intersection over Union):
        두 박스의 겹침 정도를 측정하는 지표 (0.0–1.0)
        객체 탐지 평가 및 NMS에서 사용

    Args:
        box1, box2: (x, y, w, h) 형식의 경계 상자

    Returns:
        IoU 값 (0.0–1.0)
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # 교집합 사각형 계산
    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1 + w1, x2 + w2)
    inter_y2 = min(y1 + h1, y2 + h2)

    # 교집합 면적
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    if intersection == 0:
        return 0.0

    # 합집합 면적
    union = w1 * h1 + w2 * h2 - intersection

    return intersection / union


def non_maximum_suppression(boxes: list, scores: list,
                             iou_threshold: float = 0.5) -> list:
    """Non-Maximum Suppression으로 중복 탐지를 제거합니다.

    알고리즘:
        1. 신뢰도 점수 기준으로 내림차순 정렬
        2. 가장 높은 점수 박스 선택 (유지)
        3. 선택된 박스와 IoU가 iou_threshold 이상인 나머지 박스 제거
        4. 남은 박스 중 가장 높은 점수 박스 선택 → 반복

    Args:
        boxes: (x, y, w, h) 리스트
        scores: 각 박스의 신뢰도 점수 리스트
        iou_threshold: 중복으로 간주할 IoU 임계값

    Returns:
        유지할 박스의 인덱스 리스트
    """
    if not boxes:
        return []

    # 신뢰도 내림차순 정렬
    indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    kept = []
    while indices:
        current = indices[0]
        kept.append(current)

        # 현재 박스와 나머지 박스의 IoU 계산
        indices = [
            i for i in indices[1:]
            if compute_iou(boxes[current], boxes[i]) < iou_threshold
        ]

    return kept


def demo_nms() -> None:
    """NMS 동작을 시연합니다."""
    # 동일 객체에 대한 중복 탐지 시뮬레이션
    sample = np.ones((300, 400, 3), dtype=np.uint8) * 240
    cv.rectangle(sample, (100, 80), (250, 220), (180, 180, 180), -1)
    cv.putText(sample, "Object", (130, 160), cv.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)

    # 중복 경계 상자 (실제 탐지 시뮬레이션)
    boxes = [
        (95, 75, 160, 150),   # 실제 박스와 가장 유사
        (100, 80, 155, 145),
        (90, 70, 165, 155),
        (105, 85, 150, 140),
        (200, 50, 80, 60),    # 다른 위치 박스
        (210, 55, 75, 55),
    ]
    scores = [0.95, 0.85, 0.80, 0.70, 0.90, 0.75]

    # NMS 전
    img_before = sample.copy()
    for i, ((x, y, w, h), score) in enumerate(zip(boxes, scores)):
        color = get_color_for_class(i)
        img_before = draw_bounding_box(img_before, x, y, w, h,
                                       confidence=score, color=color)

    # NMS 후
    kept_indices = non_maximum_suppression(boxes, scores, iou_threshold=0.5)
    img_after = sample.copy()
    for i in kept_indices:
        x, y, w, h = boxes[i]
        color = get_color_for_class(i)
        img_after = draw_bounding_box(img_after, x, y, w, h,
                                      confidence=scores[i], color=color)

    print(f"NMS 전: {len(boxes)}개 박스")
    print(f"NMS 후: {len(kept_indices)}개 박스 (유지된 인덱스: {kept_indices})")
    print(f"\nIoU 예시 (박스 0, 1): {compute_iou(boxes[0], boxes[1]):.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Non-Maximum Suppression (NMS)", fontsize=14)

    axes[0].imshow(cv.cvtColor(img_before, cv.COLOR_BGR2RGB))
    axes[0].set_title(f"NMS 전 ({len(boxes)}개 박스)")
    axes[0].axis("off")

    axes[1].imshow(cv.cvtColor(img_after, cv.COLOR_BGR2RGB))
    axes[1].set_title(f"NMS 후 ({len(kept_indices)}개 박스)")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    demo_nms()
