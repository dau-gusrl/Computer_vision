# 02. 이미지 처리 기법 (Image Processing Techniques)

이 섹션에서는 다양한 이미지 처리 기법을 학습합니다.

## 학습 목표
- 필터링(Blur, Sobel, Canny) 원리와 구현
- 모폴로지 연산(Erosion, Dilation, Opening, Closing)
- 히스토그램 처리 및 평활화
- 임계값 처리(Binary, Adaptive thresholding)

## 파일 설명

| 파일 | 내용 |
|------|------|
| `filtering.py` | 블러, 소벨, 캐니 에지 검출 |
| `morphology.py` | 모폴로지 연산 |
| `histogram.py` | 히스토그램 분석 및 평활화 |
| `thresholding.py` | 임계값 처리 |

## 핵심 개념

### 컨볼루션 연산
```
커널(kernel)을 이미지 위에서 슬라이딩하며 가중합 계산
커널 크기 ↑ → 더 강한 효과, 더 느린 처리
```

### 모폴로지 연산 관계
```
Opening  = Erosion  → Dilation  (작은 노이즈 제거)
Closing  = Dilation → Erosion   (작은 구멍 메우기)
```
