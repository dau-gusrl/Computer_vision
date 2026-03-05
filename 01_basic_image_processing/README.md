# 01. 기초 이미지 처리 (Basic Image Processing)

이 섹션에서는 OpenCV를 활용한 기초 이미지 처리를 학습합니다.

## 학습 목표
- 이미지 파일 로드·저장·크기 조정·회전
- 픽셀과 색상 공간(RGB, BGR, Grayscale, HSV) 이해
- OpenCV 좌표계와 채널 순서 파악
- `cv.cvtColor()` 활용 및 원리 이해

## 파일 설명

| 파일 | 내용 |
|------|------|
| `image_basics.py` | 이미지 로드, 저장, 기본 조작 |
| `color_spaces.py` | 색상 공간 변환 및 비교 |
| `image_combine.py` | 이미지 결합 및 표시 |

## 핵심 개념

### OpenCV 좌표계
```
(0,0) ──── x (열 방향) ────▶
  │
  y (행 방향)
  │
  ▼
```
- OpenCV: `image[row, col]` = `image[y, x]`
- 수학적 좌표: y 축이 반전됨에 주의

### BGR vs RGB
OpenCV는 이미지를 **BGR** 순서로 읽습니다.
```python
import cv2 as cv
img = cv.imread('image.jpg')   # B, G, R 순서
# matplotlib으로 올바르게 표시하려면 RGB로 변환 필요
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
```
