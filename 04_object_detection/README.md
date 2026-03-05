# 04. 객체 탐지 및 인식 (Object Detection and Recognition)

이 섹션에서는 전통적 방법과 딥러닝 기반의 객체 탐지를 학습합니다.

## 학습 목표
- Cascade classifier(Haar/LBP)의 원리와 활용
- 딥러닝 기반 탐지 모델(YOLO, SSD) 개념 이해
- 경계 상자(bounding box) 그리기 및 라벨링
- 탐지 결과의 후처리(NMS 등)

## 파일 설명

| 파일 | 내용 |
|------|------|
| `cascade_classifier.py` | Haar Cascade를 이용한 얼굴 검출 |
| `yolo_detection.py` | YOLOv3/v4 기반 객체 탐지 (개념 및 코드) |
| `draw_detections.py` | 경계 상자 시각화 유틸리티 |

## 핵심 개념

### 전통 방법 vs 딥러닝 방법

| 방법 | 속도 | 정확도 | 학습 데이터 필요 | 환경 적응성 |
|------|------|--------|-----------------|-------------|
| Haar Cascade | 빠름 | 낮음–중간 | 불필요 | 낮음 |
| YOLO | 빠름 | 높음 | 필요 | 높음 |
| Faster R-CNN | 느림 | 매우 높음 | 필요 | 높음 |

### NMS (Non-Maximum Suppression)
```
중복 탐지 제거 과정:
1. 신뢰도 점수 기준으로 정렬
2. 가장 높은 점수 박스 선택
3. IoU가 임계값 이상인 나머지 박스 제거
4. 반복
```
