# 05. 딥러닝 기반 컴퓨터 비전 (Deep Learning-based Computer Vision)

이 섹션에서는 CNN 아키텍처와 전이 학습을 활용한 컴퓨터 비전을 학습합니다.

## 학습 목표
- CNN 아키텍처(ResNet, VGG, MobileNet)의 특징 이해
- 전이 학습(Transfer Learning)을 활용한 분류 모델 구축
- 실시간 처리를 위한 경량 모델 최적화

## 파일 설명

| 파일 | 내용 |
|------|------|
| `cnn_concepts.py` | CNN 아키텍처 개요 및 특징 비교 |
| `transfer_learning.py` | 사전학습 모델을 활용한 전이 학습 |
| `realtime_inference.py` | 실시간 추론 및 FPS 측정 |

## 핵심 개념

### 주요 CNN 아키텍처 비교

| 모델 | 파라미터 수 | Top-1 정확도 | 특징 |
|------|------------|-------------|------|
| VGG16 | 138M | ~71% | 단순하고 깊은 구조 |
| ResNet50 | 25M | ~76% | 잔차 연결로 깊은 네트워크 학습 |
| MobileNetV2 | 3.4M | ~72% | 모바일/엣지 기기용 경량 모델 |

### 전이 학습 전략
```
Feature Extraction: 사전학습된 가중치를 고정하고 분류 헤드만 학습
Fine-tuning      : 상위 레이어를 포함하여 전체 또는 일부를 재학습
```

### 입력 정규화
```python
# ImageNet 평균/표준편차로 정규화
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```
