"""
전이 학습 (Transfer Learning)

학습 목표:
- 전이 학습의 개념과 필요성
- Feature Extraction vs Fine-tuning 전략 차이
- OpenCV DNN을 이용한 사전 학습 모델 추론
- 데이터 전처리와 정규화

Note:
    이 파일은 전이 학습의 개념을 OpenCV DNN 모듈로 시연합니다.
    PyTorch/TensorFlow 기반 전이 학습은 별도 프레임워크가 필요합니다.
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# ImageNet 정규화 파라미터 (PyTorch torchvision 기준)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def explain_transfer_learning() -> None:
    """전이 학습의 개념과 전략을 설명합니다."""
    print("""
=== 전이 학습 (Transfer Learning) ===

개념:
    대규모 데이터셋(예: ImageNet 1.2M장)에서 학습된 모델의 가중치를
    새로운 작업(소규모 데이터)에 재활용하는 기법

왜 필요한가?
    - 딥러닝은 대량의 레이블된 데이터 필요
    - 처음부터 학습(from scratch)은 막대한 계산 비용
    - 전이 학습: 하위 층의 일반적 특징(에지, 텍스처)은 재사용 가능

전략 1 - Feature Extraction (특징 추출):
    사전 학습된 가중치를 완전히 고정 (freeze)
    새로운 분류 헤드(FC 레이어)만 학습

    언제 사용?
        - 새 데이터셋이 소규모일 때
        - 새 데이터셋과 원본 데이터셋이 유사할 때

전략 2 - Fine-tuning (미세 조정):
    사전 학습된 일부 또는 전체 레이어를 재학습
    매우 낮은 학습률 사용 (예: 1e-5)

    언제 사용?
        - 충분한 새 데이터가 있을 때
        - 새 데이터셋과 원본이 다를 때 (도메인 이동)

    일반적 전략:
        초기: FC 레이어만 학습 (더 빠른 수렴)
        이후: 상위 Convolution 레이어도 학습 (Fine-tuning)

레이어별 학습된 특징 (CNN 계층 구조):
    하위 층: 저수준 특징 (에지, 색상, 질감) - 범용적
    중위 층: 패턴, 부분 모양 - 반범용적
    상위 층: 고수준 특징 (객체 부위) - 작업 특화적
""")


def preprocess_for_imagenet(image: np.ndarray,
                             input_size: tuple = (224, 224)) -> np.ndarray:
    """ImageNet 사전 학습 모델 입력을 위한 전처리를 수행합니다.

    Args:
        image: BGR 이미지
        input_size: 모델 입력 크기 (너비, 높이)

    Returns:
        전처리된 배열 (shape: [1, C, H, W], float32)

    처리 순서:
        1. 크기 조정
        2. BGR → RGB 변환
        3. [0, 255] → [0.0, 1.0] 정규화
        4. ImageNet 평균 차감
        5. ImageNet 표준편차 나눔
        6. HWC → CHW 형식 변환
        7. 배치 차원 추가
    """
    # 1. 크기 조정
    resized = cv.resize(image, input_size)

    # 2. BGR → RGB
    rgb = cv.cvtColor(resized, cv.COLOR_BGR2RGB)

    # 3. float32로 변환 후 [0, 1] 정규화
    normalized = rgb.astype(np.float32) / 255.0

    # 4-5. ImageNet 평균/표준편차 정규화
    normalized = (normalized - IMAGENET_MEAN) / IMAGENET_STD

    # 6. HWC → CHW (Height, Width, Channel → Channel, Height, Width)
    chw = normalized.transpose(2, 0, 1)

    # 7. 배치 차원 추가: CHW → NCHW
    return np.expand_dims(chw, axis=0)


def reverse_imagenet_normalization(tensor: np.ndarray) -> np.ndarray:
    """ImageNet 정규화를 역변환합니다 (시각화용).

    Args:
        tensor: 정규화된 배열 (CHW 또는 HWC)

    Returns:
        [0, 255] uint8 이미지
    """
    if tensor.ndim == 3 and tensor.shape[0] in (1, 3):
        # CHW → HWC
        arr = tensor.transpose(1, 2, 0)
    else:
        arr = tensor.copy()

    arr = arr * IMAGENET_STD + IMAGENET_MEAN
    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    return arr


def load_pretrained_model_onnx(model_path: str) -> cv.dnn.Net:
    """ONNX 형식의 사전 학습 모델을 로드합니다.

    Args:
        model_path: .onnx 파일 경로

    Returns:
        OpenCV DNN 네트워크 객체

    Raises:
        FileNotFoundError: 모델 파일이 없을 때

    모델 변환 예시 (PyTorch → ONNX):
        import torch
        model = torchvision.models.resnet50(pretrained=True)
        model.eval()
        dummy = torch.randn(1, 3, 224, 224)
        torch.onnx.export(model, dummy, "resnet50.onnx",
                          input_names=["input"], output_names=["output"])
    """
    import os
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"ONNX 모델 파일 없음: {model_path}\n"
            "모델 변환 방법: PyTorch → torch.onnx.export()"
        )

    net = cv.dnn.readNetFromONNX(model_path)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    return net


def run_inference(net: cv.dnn.Net, blob: np.ndarray) -> np.ndarray:
    """전처리된 입력으로 추론을 실행합니다.

    Args:
        net: OpenCV DNN 네트워크
        blob: blobFromImage() 또는 직접 전처리된 배열

    Returns:
        네트워크 출력 배열
    """
    net.setInput(blob)
    return net.forward()


def visualize_preprocessing_steps(image: np.ndarray) -> None:
    """전처리 단계를 단계별로 시각화합니다."""
    # 1. 원본
    original_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    # 2. 크기 조정
    resized = cv.resize(image, (224, 224))
    resized_rgb = cv.cvtColor(resized, cv.COLOR_BGR2RGB)

    # 3. [0,1] 정규화
    normalized_01 = resized_rgb.astype(np.float32) / 255.0

    # 4. ImageNet 정규화
    imagenet_normalized = (normalized_01 - IMAGENET_MEAN) / IMAGENET_STD

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("ImageNet 전처리 단계별 시각화", fontsize=14)

    axes[0, 0].imshow(original_rgb)
    axes[0, 0].set_title(f"원본\n{image.shape[:2]} BGR")

    axes[0, 1].imshow(resized_rgb)
    axes[0, 1].set_title("크기 조정\n224×224 RGB")

    axes[0, 2].imshow(normalized_01)
    axes[0, 2].set_title(f"[0,1] 정규화\n값 범위: 0.0 ~ 1.0")

    # 채널별 정규화 값 시각화
    for ch, (name, mean_val, std_val) in enumerate(zip(
        ["R", "G", "B"],
        IMAGENET_MEAN,
        IMAGENET_STD,
    )):
        channel = imagenet_normalized[:, :, ch]
        axes[1, ch].imshow(channel, cmap="RdBu_r",
                           vmin=-3, vmax=3)
        axes[1, ch].set_title(
            f"ImageNet 정규화 - {name} 채널\n"
            f"mean={mean_val:.3f}, std={std_val:.3f}\n"
            f"값 범위: {channel.min():.2f} ~ {channel.max():.2f}"
        )

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def explain_fine_tuning_strategy() -> None:
    """Fine-tuning 전략을 시각화합니다."""
    print("""
=== Fine-tuning 레이어 전략 ===

단계 1 (Feature Extraction):
    [사전학습 Conv 레이어 - 모두 동결] ──┐
    [새 FC 레이어 - 학습] ◀─────────────┘
    학습률: 1e-3 ~ 1e-4
    에포크: 10~20

단계 2 (Fine-tuning, 선택적):
    [사전학습 하위 층 - 동결] ──────────┐
    [사전학습 상위 층 - 미세 조정] ─────┤ 매우 낮은 학습률
    [새 FC 레이어 - 계속 학습] ◀────────┘
    학습률: 1e-5 ~ 1e-6
    에포크: 추가 10~20

주의사항:
    - 사전학습 레이어를 처음부터 높은 학습률로 훈련하면 특징 망가짐
    - Batch Normalization 레이어는 동결 상태 유지 권장
    - 데이터 증강(Data Augmentation) 적극 활용
""")


def demo_transfer_learning() -> None:
    """전이 학습 개념을 시연합니다."""
    explain_transfer_learning()
    explain_fine_tuning_strategy()

    # 전처리 단계 시각화
    sample = np.zeros((300, 400, 3), dtype=np.uint8)
    cv.rectangle(sample, (50, 50), (200, 200), (200, 150, 100), -1)
    cv.circle(sample, (300, 150), 80, (100, 180, 200), -1)
    cv.putText(sample, "Transfer", (80, 270), cv.FONT_HERSHEY_SIMPLEX, 1, (230, 230, 230), 2)
    cv.putText(sample, "Learning", (90, 300), cv.FONT_HERSHEY_SIMPLEX, 1, (230, 230, 230), 2)

    visualize_preprocessing_steps(sample)

    # blob 형식 전처리 결과 확인
    blob = preprocess_for_imagenet(sample)
    print(f"\n전처리 결과:")
    print(f"  원본 shape: {sample.shape}")
    print(f"  blob shape: {blob.shape}  (N, C, H, W)")
    print(f"  blob dtype: {blob.dtype}")
    print(f"  값 범위: {blob.min():.3f} ~ {blob.max():.3f}")


if __name__ == "__main__":
    demo_transfer_learning()
