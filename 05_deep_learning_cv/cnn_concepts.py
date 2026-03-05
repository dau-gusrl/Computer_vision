"""
CNN 아키텍처 개요 (CNN Architecture Overview)

학습 목표:
- CNN의 기본 구성 요소 이해 (Convolution, Pooling, FC)
- VGG, ResNet, MobileNet의 구조적 특징
- 잔차 연결(Residual Connection)의 필요성
- 깊이별 분리 합성곱(Depthwise Separable Convolution)

Note:
    이 파일은 OpenCV와 NumPy만으로 CNN 개념을 설명합니다.
    실제 딥러닝 모델은 PyTorch나 TensorFlow를 사용합니다.
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def explain_cnn_building_blocks() -> None:
    """CNN 기본 구성 요소를 설명합니다."""
    print("""
=== CNN 기본 구성 요소 ===

1. Convolutional Layer (합성곱 층)
   - 입력: (H, W, C_in)
   - 필터: (kH, kW, C_in, C_out)
   - 출력: (H', W', C_out)
   - 파라미터 수: kH × kW × C_in × C_out + C_out (bias)
   - 역할: 지역적 특징(에지, 패턴) 추출

2. Activation (활성화 함수)
   - ReLU: f(x) = max(0, x)  → 가장 일반적
   - 비선형성 도입으로 복잡한 패턴 학습 가능

3. Pooling Layer (풀링 층)
   - MaxPooling: 최대값 선택 → 이동 불변성
   - AvgPooling: 평균값 → 특징 압축
   - 역할: 공간 해상도 축소, 과적합 방지

4. Batch Normalization
   - 각 미니배치의 출력을 정규화
   - 학습 안정화, 더 높은 학습률 사용 가능

5. Fully Connected Layer (완전 연결 층)
   - 분류 헤드에 주로 사용
   - 모든 입력 뉴런과 출력 뉴런을 연결
""")


def explain_vgg() -> None:
    """VGG 아키텍처를 설명합니다."""
    print("""
=== VGG (Visual Geometry Group, 2014) ===

핵심 아이디어:
    3×3 작은 커널을 여러 층 쌓는 것이 큰 커널 하나보다 효과적
    3×3 × 2층 = 5×5 커널과 동일한 수용 영역, 더 적은 파라미터

구조 (VGG16):
    Input (224×224×3)
    → [Conv3-64] × 2 → MaxPool → (112×112×64)
    → [Conv3-128] × 2 → MaxPool → (56×56×128)
    → [Conv3-256] × 3 → MaxPool → (28×28×256)
    → [Conv3-512] × 3 → MaxPool → (14×14×512)
    → [Conv3-512] × 3 → MaxPool → (7×7×512)
    → FC-4096 → FC-4096 → FC-1000 → Softmax

특징:
    파라미터 수: 약 138M (매우 많음)
    단순하고 이해하기 쉬운 구조
    전이 학습 베이스라인으로 자주 사용

한계:
    FC 레이어의 막대한 파라미터 (138M 중 ~124M이 FC)
    메모리 사용량이 많아 실용적이지 않음
""")


def explain_resnet() -> None:
    """ResNet 아키텍처를 설명합니다."""
    print("""
=== ResNet (Residual Networks, 2015) ===

문제 인식:
    깊은 네트워크에서 Gradient Vanishing/Exploding 문제
    층을 추가할수록 오히려 성능이 저하되는 현상

핵심 아이디어 - 잔차 연결 (Residual Connection / Skip Connection):
    일반:   출력 = F(x)
    ResNet: 출력 = F(x) + x  (항등 사상 추가)

    → 최소한 항등 함수는 학습 보장
    → 기울기가 잔차 경로로 직접 흐름 (Gradient Highway)

Residual Block 구조:
    x → [Conv → BN → ReLU → Conv → BN] → + x → ReLU
                                          ↑
                                    Shortcut Connection

버전:
    ResNet-18/34:  기본 블록 (Conv-BN-ReLU × 2)
    ResNet-50/101: Bottleneck 블록 (1×1-3×3-1×1)

특징:
    파라미터 수: ResNet-50 약 25M (VGG16의 1/5)
    ImageNet Top-1: ~76% (ResNet-50)
    매우 깊은 네트워크 학습 가능 (150층 이상)
""")


def explain_mobilenet() -> None:
    """MobileNet 아키텍처를 설명합니다."""
    print("""
=== MobileNet (2017) ===

목표: 모바일/임베디드 기기에서 실시간 추론

핵심 아이디어 - 깊이별 분리 합성곱 (Depthwise Separable Convolution):

    일반 합성곱:
        입력 (H×W×C_in) × 필터 (k×k×C_in×C_out)
        계산량: H × W × k² × C_in × C_out

    깊이별 분리 합성곱 (= Depthwise + Pointwise):
        1. Depthwise: 각 채널에 독립적으로 k×k 합성곱
           계산량: H × W × k² × C_in
        2. Pointwise: 1×1 합성곱으로 채널 변환
           계산량: H × W × C_in × C_out

        총 계산량 감소: 약 1/k² + 1/C_out ≈ 1/9 (k=3)

MobileNetV2 (2018) 추가 개선:
    - Inverted Residuals: 채널 확장 후 압축
    - Linear Bottleneck: 마지막 활성화 없음

특징:
    파라미터 수: ~3.4M (VGG16의 1/40)
    ImageNet Top-1: ~72%
    실시간 모바일 추론 가능
""")


def simulate_convolution(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """합성곱 연산을 시각화하기 위해 OpenCV로 적용합니다.

    Args:
        image: 그레이스케일 이미지
        kernel: 합성곱 커널

    Returns:
        합성곱 결과
    """
    # float64로 변환 후 합성곱 적용
    return cv.filter2D(image.astype(np.float64), -1, kernel)


def visualize_feature_maps(image: np.ndarray) -> None:
    """CNN의 다양한 필터 응답(특징 맵)을 시각화합니다."""
    if image.ndim == 3:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY).astype(np.float64)
    else:
        gray = image.astype(np.float64)

    # 다양한 CNN 스타일 필터
    filters = {
        "가로 에지 (Sobel-Y)": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                         dtype=np.float64),
        "세로 에지 (Sobel-X)": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                         dtype=np.float64),
        "가우시안 블러": cv.getGaussianKernel(5, 1) @ cv.getGaussianKernel(5, 1).T,
        "라플라시안 (2차 미분)": np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                                           dtype=np.float64),
        "샤프닝": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
                            dtype=np.float64),
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("CNN 필터 응답 (특징 맵) 시각화", fontsize=14)

    # 원본
    axes[0, 0].imshow(gray, cmap="gray")
    axes[0, 0].set_title("원본 이미지")
    axes[0, 0].axis("off")

    for ax, (name, kernel) in zip(axes.flat[1:], filters.items()):
        feature_map = simulate_convolution(gray, kernel)
        feature_map_abs = np.abs(feature_map)
        # 정규화하여 표시
        normalized = cv.normalize(feature_map_abs, None, 0, 255,
                                  cv.NORM_MINMAX).astype(np.uint8)
        ax.imshow(normalized, cmap="gray")
        ax.set_title(f"필터: {name}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def compare_architectures() -> None:
    """주요 CNN 아키텍처를 비교합니다."""
    architectures = {
        "VGG16": {"params_M": 138, "top1_acc": 71.3, "year": 2014, "layers": 16},
        "ResNet-50": {"params_M": 25.6, "top1_acc": 76.1, "year": 2015, "layers": 50},
        "ResNet-101": {"params_M": 44.5, "top1_acc": 77.4, "year": 2015, "layers": 101},
        "MobileNetV1": {"params_M": 4.2, "top1_acc": 70.6, "year": 2017, "layers": 28},
        "MobileNetV2": {"params_M": 3.4, "top1_acc": 72.0, "year": 2018, "layers": 53},
    }

    names = list(architectures.keys())
    params = [v["params_M"] for v in architectures.values()]
    accuracy = [v["top1_acc"] for v in architectures.values()]
    years = [v["year"] for v in architectures.values()]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("CNN 아키텍처 비교 (ImageNet)", fontsize=14)

    # 파라미터 수
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))
    bars = axes[0].barh(names, params, color=colors)
    axes[0].set_xlabel("파라미터 수 (백만)")
    axes[0].set_title("모델 크기 비교")
    for bar, val in zip(bars, params):
        axes[0].text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                     f"{val}M", va="center")

    # 정확도 vs 파라미터 산점도
    scatter = axes[1].scatter(params, accuracy,
                               c=[v["year"] for v in architectures.values()],
                               cmap="plasma", s=200, zorder=5)
    for name, p, a in zip(names, params, accuracy):
        axes[1].annotate(name, (p, a), textcoords="offset points",
                         xytext=(5, 5), fontsize=9)
    axes[1].set_xlabel("파라미터 수 (백만)")
    axes[1].set_ylabel("Top-1 정확도 (%)")
    axes[1].set_title("정확도 vs 모델 크기\n(색상: 발표 연도)")
    plt.colorbar(scatter, ax=axes[1], label="발표 연도")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def demo_cnn_concepts() -> None:
    """CNN 개념 시연."""
    explain_cnn_building_blocks()
    explain_vgg()
    explain_resnet()
    explain_mobilenet()

    # 특징 맵 시각화
    sample = np.zeros((200, 300, 3), dtype=np.uint8)
    cv.rectangle(sample, (30, 30), (130, 130), (200, 200, 200), 2)
    cv.circle(sample, (220, 100), 60, (180, 180, 180), 2)
    cv.putText(sample, "CNN", (80, 180), cv.FONT_HERSHEY_SIMPLEX, 1.2, (220, 220, 220), 2)

    visualize_feature_maps(sample)
    compare_architectures()


if __name__ == "__main__":
    demo_cnn_concepts()
