"""
YOLO 기반 객체 탐지 개요 (YOLO-based Object Detection Overview)

학습 목표:
- YOLO(You Only Look Once) 아키텍처 원리
- 그리드 기반 탐지 메커니즘 이해
- Anchor Box 개념
- OpenCV DNN 모듈을 이용한 사전 학습 모델 실행
- 전통적 방법과 딥러닝 방법의 비교

Note:
    이 파일은 YOLO의 개념 설명과 OpenCV DNN 기반 실행 방법을 다룹니다.
    실제 모델 파일(weights, cfg) 없이도 구조를 이해할 수 있도록 작성되었습니다.
    모델 파일은 공식 YOLO 저장소에서 다운로드하세요:
    https://github.com/AlexeyAB/darknet
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# COCO 데이터셋 80개 클래스 이름 (일부)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorbike", "aeroplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird",
    "cat", "dog", "horse", "sheep", "cow",
]


def explain_yolo_architecture() -> None:
    """YOLO 아키텍처의 핵심 개념을 설명합니다."""
    print("""
=== YOLO (You Only Look Once) 아키텍처 ===

1. 그리드 분할
   - 입력 이미지를 S×S 그리드로 분할 (예: 13×13)
   - 각 그리드 셀이 B개의 경계 상자를 예측

2. 각 그리드 셀의 예측 값
   - B × (5 + C) 개의 값 예측
     * 5 = [x, y, w, h, confidence]
     * C = 클래스 수 (COCO: 80개)
   - confidence = Pr(Object) × IoU(pred, truth)

3. Anchor Box
   - 사전 정의된 비율의 박스로 예측 안정성 향상
   - K-means 클러스터링으로 학습 데이터에서 결정

4. 손실 함수 (Loss Function)
   - 좌표 손실: (x, y, w, h) 예측 오차
   - 객체 신뢰도 손실: 객체 유무 오차
   - 분류 손실: 클래스 예측 오차

5. 버전별 발전
   YOLOv1 (2016): 기본 개념 제안, 속도 우선
   YOLOv2 (2017): Anchor Box, Batch Norm 추가
   YOLOv3 (2018): 다중 스케일 예측, Darknet-53
   YOLOv4 (2020): 다양한 최적화 기법 통합
   YOLOv5+ :      PyTorch 기반, 실용성 향상

전통적 방법 vs YOLO:
   전통적:  슬라이딩 윈도우 → 분류 → 느림
   YOLO:    전체 이미지 한 번 처리 → 빠름 (실시간 가능)
""")


def load_yolo_model(weights_path: str, config_path: str) -> cv.dnn.Net:
    """OpenCV DNN 모듈로 YOLO 모델을 로드합니다.

    Args:
        weights_path: .weights 파일 경로 (예: yolov3.weights)
        config_path: .cfg 파일 경로 (예: yolov3.cfg)

    Returns:
        로드된 네트워크 객체

    Raises:
        FileNotFoundError: 모델 파일이 없을 때

    Example:
        net = load_yolo_model("yolov3.weights", "yolov3.cfg")

    모델 다운로드:
        wget https://pjreddie.com/media/files/yolov3.weights
        wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg
    """
    import os
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"YOLO weights 파일 없음: {weights_path}\n"
            "다운로드: wget https://pjreddie.com/media/files/yolov3.weights"
        )
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"YOLO config 파일 없음: {config_path}\n"
            "다운로드: wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg"
        )

    net = cv.dnn.readNet(weights_path, config_path)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    return net


def preprocess_for_yolo(image: np.ndarray, input_size: tuple = (416, 416)) -> np.ndarray:
    """YOLO 입력 형식으로 이미지를 전처리합니다.

    Args:
        image: BGR 이미지
        input_size: 모델 입력 크기 (너비, 높이)

    Returns:
        blob 형식의 전처리된 배열

    Note:
        cv.dnn.blobFromImage 파라미터:
            scalefactor : 픽셀 값 정규화 계수 (1/255 = [0,1] 범위)
            size        : 모델 입력 크기
            mean        : 평균 차감값 (YOLO는 (0,0,0))
            swapRB      : BGR → RGB 변환 여부 (True)
            crop        : 중앙 자르기 여부 (False)
    """
    blob = cv.dnn.blobFromImage(
        image, scalefactor=1 / 255.0,
        size=input_size, mean=(0, 0, 0),
        swapRB=True, crop=False
    )
    return blob


def get_output_layer_names(net: cv.dnn.Net) -> list:
    """YOLO 출력 레이어 이름을 반환합니다.

    YOLO는 여러 스케일(YOLOv3: 3개)에서 예측하므로
    모든 출력 레이어에서 결과를 수집해야 합니다.
    """
    layer_names = net.getLayerNames()
    # getUnconnectedOutLayers()는 출력 레이어의 인덱스 반환
    unconnected = net.getUnconnectedOutLayers()
    return [layer_names[i - 1] for i in unconnected.flatten()]


def parse_yolo_output(outputs: list, image_shape: tuple,
                      conf_threshold: float = 0.5,
                      nms_threshold: float = 0.4) -> tuple:
    """YOLO 출력을 파싱하여 경계 상자를 추출합니다.

    Args:
        outputs: 네트워크 출력 리스트
        image_shape: (높이, 너비) 원본 이미지 크기
        conf_threshold: 신뢰도 임계값 (이 값 이상만 유지)
        nms_threshold: NMS IoU 임계값

    Returns:
        (boxes, confidences, class_ids) 튜플
        boxes: [(x, y, w, h), ...] (픽셀 좌표)
        confidences: [float, ...]
        class_ids: [int, ...]
    """
    h, w = image_shape[:2]
    boxes, confidences, class_ids = [], [], []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])

            if confidence > conf_threshold:
                # YOLO는 정규화된 좌표 (0–1) 반환 → 픽셀 좌표로 변환
                cx = int(detection[0] * w)
                cy = int(detection[1] * h)
                bw = int(detection[2] * w)
                bh = int(detection[3] * h)

                # 좌상단 좌표로 변환
                x = cx - bw // 2
                y = cy - bh // 2

                boxes.append([x, y, bw, bh])
                confidences.append(confidence)
                class_ids.append(class_id)

    # NMS 적용
    if boxes:
        indices = cv.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        if len(indices) > 0:
            kept = indices.flatten()
            return (
                [boxes[i] for i in kept],
                [confidences[i] for i in kept],
                [class_ids[i] for i in kept],
            )

    return [], [], []


def visualize_yolo_grid(image: np.ndarray, grid_size: int = 13) -> np.ndarray:
    """YOLO 그리드 분할을 시각화합니다."""
    result = image.copy()
    h, w = result.shape[:2]

    cell_h = h // grid_size
    cell_w = w // grid_size

    # 그리드 선 그리기
    for i in range(1, grid_size):
        cv.line(result, (i * cell_w, 0), (i * cell_w, h), (100, 100, 100), 1)
        cv.line(result, (0, i * cell_h), (w, i * cell_h), (100, 100, 100), 1)

    cv.putText(result, f"{grid_size}x{grid_size} Grid",
               (5, 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    return result


def demo_yolo_concept() -> None:
    """YOLO 개념을 시각적으로 시연합니다."""
    explain_yolo_architecture()

    # 객체가 있는 샘플 이미지 생성
    sample = np.ones((416, 416, 3), dtype=np.uint8) * 200
    # 객체 1: 사람 형태
    cv.rectangle(sample, (50, 80), (130, 320), (150, 120, 100), -1)
    cv.circle(sample, (90, 60), 30, (160, 130, 110), -1)
    # 객체 2: 차량 형태
    cv.rectangle(sample, (200, 200), (380, 320), (100, 100, 150), -1)
    cv.rectangle(sample, (220, 160), (360, 210), (100, 100, 150), -1)

    # 그리드 오버레이
    grid_img = visualize_yolo_grid(sample, grid_size=13)

    # 가상의 탐지 결과 (YOLO 출력 시뮬레이션)
    fake_detections = [
        {"bbox": (45, 50, 95, 280), "label": "person", "confidence": 0.92, "class_id": 0},
        {"bbox": (190, 155, 200, 175), "label": "car", "confidence": 0.88, "class_id": 2},
    ]

    from draw_detections import draw_detections_batch
    result = draw_detections_batch(sample, fake_detections)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("YOLO 객체 탐지 개념", fontsize=14)

    axes[0].imshow(cv.cvtColor(sample, cv.COLOR_BGR2RGB))
    axes[0].set_title("원본 이미지")
    axes[0].axis("off")

    axes[1].imshow(cv.cvtColor(grid_img, cv.COLOR_BGR2RGB))
    axes[1].set_title("YOLO 그리드 분할\n(13×13, 각 셀에서 예측)")
    axes[1].axis("off")

    axes[2].imshow(cv.cvtColor(result, cv.COLOR_BGR2RGB))
    axes[2].set_title("탐지 결과\n(경계 상자 + 클래스 + 신뢰도)")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    demo_yolo_concept()
