# 03. 특징 검출 및 추출 (Feature Detection and Extraction)

이 섹션에서는 이미지에서 특징을 검출하고 추출하는 방법을 학습합니다.

## 학습 목표
- SIFT, ORB 특징점 검출기 이해 및 활용
- Harris 코너 검출
- Hough Transform을 이용한 직선/원 검출
- 특징점 매칭 및 응용

## 파일 설명

| 파일 | 내용 |
|------|------|
| `feature_detectors.py` | SIFT, ORB 특징점 검출 및 비교 |
| `harris_corner.py` | Harris 코너 검출 |
| `hough_transform.py` | Hough 직선·원 검출 |

## 핵심 개념

### 특징점 검출기 비교

| 방법 | 속도 | 정확도 | 특허 | 회전 불변 | 크기 불변 |
|------|------|--------|------|-----------|-----------|
| SIFT | 느림 | 높음 | 만료 | O | O |
| ORB  | 빠름 | 중간 | 없음 | O | △ |

### Hough Transform 원리
```
이미지 공간의 점 → 파라미터 공간의 곡선
교점이 많은 파라미터 = 강한 직선/원 후보
```
