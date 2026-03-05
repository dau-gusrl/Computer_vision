"""
Cascade Classifier를 이용한 객체 검출 (Object Detection with Cascade Classifier)

학습 목표:
- Haar Cascade의 원리 (Viola-Jones 알고리즘)
- Integral Image의 효율적 계산 방법
- Adaboost와 Cascade 구조 이해
- 얼굴/눈 검출 파라미터 조정
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def load_cascade(cascade_type: str = "face") -> cv.CascadeClassifier:
    """사전 훈련된 Cascade 분류기를 로드합니다.

    Args:
        cascade_type: 'face'(얼굴), 'eye'(눈), 'fullbody'(전신)

    Returns:
        CascadeClassifier 객체

    Note:
        OpenCV 설치 시 사전 훈련된 XML 파일 포함
        경로: cv.data.haarcascades + '파일명'

    Raises:
        ValueError: 지원하지 않는 cascade_type인 경우
    """
    cascade_files = {
        "face": "haarcascade_frontalface_default.xml",
        "eye": "haarcascade_eye.xml",
        "fullbody": "haarcascade_fullbody.xml",
        "profile_face": "haarcascade_profileface.xml",
    }

    if cascade_type not in cascade_files:
        raise ValueError(
            f"cascade_type은 {list(cascade_files.keys())} 중 하나여야 합니다."
        )

    cascade_path = cv.data.haarcascades + cascade_files[cascade_type]
    cascade = cv.CascadeClassifier(cascade_path)

    if cascade.empty():
        raise RuntimeError(f"Cascade 파일 로드 실패: {cascade_path}")

    return cascade


def detect_faces(image: np.ndarray, scale_factor: float = 1.1,
                 min_neighbors: int = 5,
                 min_size: tuple = (30, 30)) -> list:
    """얼굴을 검출합니다.

    Args:
        image: BGR 이미지
        scale_factor: 스케일 피라미드 축소 비율 (1.05–1.4 권장)
            - 작을수록 정확하지만 느림
        min_neighbors: 최소 이웃 사각형 수 (높을수록 오탐 감소, 검출 감소)
        min_size: 검출할 최소 얼굴 크기 (픽셀)

    Returns:
        검출된 얼굴 사각형 리스트 [(x, y, w, h), ...]

    Note:
        Viola-Jones 알고리즘 3가지 핵심:
        1. Integral Image: 사각형 합 O(1) 계산
        2. Adaboost: 가장 판별력 있는 Haar 특징 선택
        3. Cascade: 빠른 탈락으로 처리 속도 향상
    """
    face_cascade = load_cascade("face")
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) if image.ndim == 3 else image

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size,
    )
    return faces if len(faces) > 0 else []


def detect_eyes_in_faces(image: np.ndarray, faces: list) -> dict:
    """검출된 얼굴 영역 내에서 눈을 검출합니다.

    Args:
        image: BGR 이미지
        faces: 얼굴 사각형 리스트 [(x, y, w, h), ...]

    Returns:
        얼굴 인덱스 → 눈 사각형 리스트 딕셔너리
    """
    eye_cascade = load_cascade("eye")
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) if image.ndim == 3 else image

    eyes_per_face = {}
    for i, (x, y, w, h) in enumerate(faces):
        # 얼굴 ROI 내에서만 눈 검출 (속도 향상 + 오탐 감소)
        face_roi = gray[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=3)
        eyes_per_face[i] = eyes if len(eyes) > 0 else []

    return eyes_per_face


def draw_detections(image: np.ndarray, faces: list,
                    eyes_per_face: dict = None) -> np.ndarray:
    """검출된 얼굴과 눈을 이미지에 그립니다.

    Args:
        image: BGR 이미지
        faces: 얼굴 사각형 리스트
        eyes_per_face: 얼굴 인덱스 → 눈 리스트 딕셔너리 (선택)

    Returns:
        검출 결과가 표시된 이미지
    """
    result = image.copy()

    for i, (x, y, w, h) in enumerate(faces):
        # 얼굴 사각형: 파란색
        cv.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv.putText(result, f"Face {i+1}", (x, y - 8),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # 눈 사각형: 초록색 (얼굴 ROI 기준 좌표 → 전체 이미지 좌표로 변환)
        if eyes_per_face and i in eyes_per_face:
            for (ex, ey, ew, eh) in eyes_per_face[i]:
                cv.rectangle(result,
                              (x + ex, y + ey),
                              (x + ex + ew, y + ey + eh),
                              (0, 255, 0), 2)

    print(f"검출된 얼굴: {len(faces)}개")
    return result


def explain_viola_jones() -> None:
    """Viola-Jones 알고리즘의 핵심 개념을 출력합니다."""
    print("""
=== Viola-Jones 알고리즘 (Haar Cascade) ===

1. Haar 특징 (Haar-like Features)
   - 인접한 직사각형 영역의 픽셀 합 차이
   - 종류: 엣지, 선, 대각선 특징
   - 24×24 검출 윈도우에서 약 180,000개 특징 존재

2. Integral Image (적분 이미지)
   - ii(x,y) = Σ i(x',y') (x'≤x, y'≤y)
   - 임의의 직사각형 합을 O(1) 시간에 계산
   - 속도 향상의 핵심

3. Adaboost 학습
   - 수천 개 Haar 특징 중 판별력 높은 소수 선택
   - 약한 분류기들의 가중 합으로 강한 분류기 생성

4. Cascade 구조
   - 초기 단계에서 배경을 빠르게 탈락 (reject)
   - 양성으로 판단된 경우만 다음 단계로 진행
   - 처리 속도 대폭 향상 (실시간 가능)

장점: 빠른 속도, 경량
단점: 측면 얼굴, 가려진 얼굴, 조명 변화에 취약
""")


def demo_with_sample_image() -> None:
    """샘플 이미지로 Cascade 검출을 시연합니다.

    Note:
        실제 얼굴 이미지 없이 개념을 시연하기 위해
        Haar cascade를 로드하고 동작 원리를 설명합니다.
    """
    explain_viola_jones()

    # 얼굴과 유사한 타원/도형으로 구성된 합성 이미지 (실제 얼굴 아님)
    sample = np.ones((400, 400, 3), dtype=np.uint8) * 200
    # 얼굴 형태 (타원)
    cv.ellipse(sample, (200, 200), (120, 150), 0, 0, 360, (180, 160, 140), -1)
    # 눈 영역
    cv.ellipse(sample, (160, 160), (25, 15), 0, 0, 360, (80, 60, 50), -1)
    cv.ellipse(sample, (240, 160), (25, 15), 0, 0, 360, (80, 60, 50), -1)
    # 코
    cv.ellipse(sample, (200, 200), (10, 15), 0, 0, 360, (160, 130, 110), -1)
    # 입
    cv.ellipse(sample, (200, 240), (35, 15), 0, 180, 360, (130, 90, 80), -1)

    print("샘플 이미지 생성 완료")
    print("(참고: Haar Cascade는 실제 얼굴 이미지에서 효과적입니다)")
    print()

    # 실제 얼굴 검출 시도 (합성 이미지이므로 검출 안 될 수 있음)
    try:
        faces = detect_faces(sample, scale_factor=1.1, min_neighbors=3)
        result = draw_detections(sample, faces)
    except Exception as e:
        print(f"검출 오류: {e}")
        result = sample

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv.cvtColor(sample, cv.COLOR_BGR2RGB))
    plt.title("샘플 이미지\n(합성 얼굴 형태)")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(cv.cvtColor(result, cv.COLOR_BGR2RGB))
    plt.title("Haar Cascade 검출 결과\n(실제 얼굴 이미지에서 효과적)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    demo_with_sample_image()
