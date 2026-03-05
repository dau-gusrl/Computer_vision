# 컴퓨터 비전 강의 (Computer Vision Lecture)

OpenCV와 딥러닝을 활용한 대학교 수준의 컴퓨터 비전 강의 자료입니다.

---

## 강의 목표

- OpenCV를 활용한 기초 이미지 처리 기술 습득
- 수학적 원리와 실제 코드 구현 능력 배양
- 딥러닝 기반 컴퓨터 비전 기술 이해 및 적용
- 실무에서 활용 가능한 컴퓨터 비전 시스템 설계 능력 함양

---

## 강의 구성

| 주차 | 주제 | 디렉터리 |
|------|------|----------|
| 1–2 | 기초 이미지 처리 | [`01_basic_image_processing/`](01_basic_image_processing/) |
| 3–4 | 이미지 처리 기법 | [`02_image_processing_techniques/`](02_image_processing_techniques/) |
| 5–6 | 특징 검출 및 추출 | [`03_feature_detection/`](03_feature_detection/) |
| 7–8 | 객체 탐지 및 인식 | [`04_object_detection/`](04_object_detection/) |
| 9–10 | 딥러닝 기반 컴퓨터 비전 | [`05_deep_learning_cv/`](05_deep_learning_cv/) |

---

## 환경 설정

```bash
pip install -r requirements.txt
```

### 주요 의존성
- Python 3.8+
- opencv-python
- numpy
- matplotlib

---

## 핵심 학습 개념

1. **좌표계**: OpenCV는 `(x, y) = (열, 행)` 사용
2. **채널 순서**: BGR (RGB 아님) — OpenCV 기본값
3. **데이터 타입**: `uint8` (0–255), `float32` (0.0–1.0) 차이 이해
4. **메모리 효율**: 대형 이미지 처리 시 최적화 필요
5. **정규화**: 딥러닝 입력 시 `[0, 1]` 또는 평균/표준편차로 정규화
6. **실시간 처리**: FPS 계산과 병목 지점 파악

---

## 강의 원칙

- 단순 코드 복사가 아닌 **원리 이해** 중심
- "어떻게"보다 **"왜"** 이해
- 오류 메시지를 읽고 **스스로 해결**하는 능력 배양
- 공식 문서 참고 습관 형성
- 다양한 접근법 비교 분석

---

## 참고 자료

- [OpenCV 공식 문서](https://docs.opencv.org/)
- [OpenCV Python 튜토리얼](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [NumPy 공식 문서](https://numpy.org/doc/)
