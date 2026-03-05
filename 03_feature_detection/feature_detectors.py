"""
특징점 검출 및 설명자 (Feature Detectors and Descriptors)

학습 목표:
- SIFT 특징점 검출 원리와 구현
- ORB 특징점 검출 (오픈소스 대안)
- 특징점 매칭 방법 (BFMatcher, FLANN)
- 각 방법의 장단점 비교
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def detect_and_describe_sift(image: np.ndarray) -> tuple:
    """SIFT 특징점을 검출하고 설명자를 계산합니다.

    SIFT (Scale-Invariant Feature Transform) 원리:
        1. 스케일 공간(Scale Space) 구성: DoG(Difference of Gaussians)
        2. 극값(Extrema) 검출: 스케일과 위치에서의 극값 찾기
        3. 키포인트 정제: 에지/저대비 포인트 제거
        4. 방향 할당: 기울기 방향 히스토그램으로 방향 결정
        5. 설명자 생성: 128차원 벡터

    특징:
        - 크기(Scale) 불변
        - 회전(Rotation) 불변
        - 조명 변화에 강건
        - 계산 비용이 높음 (ORB 대비)

    Args:
        image: BGR 이미지

    Returns:
        (keypoints, descriptors) 튜플
    """
    sift = cv.SIFT_create()
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) if image.ndim == 3 else image
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors


def detect_and_describe_orb(image: np.ndarray, n_features: int = 500) -> tuple:
    """ORB 특징점을 검출하고 설명자를 계산합니다.

    ORB (Oriented FAST and Rotated BRIEF) 원리:
        1. FAST 코너 검출로 키포인트 찾기
        2. Harris 코너 점수로 상위 N개 선택
        3. 픽셀 강도 패치의 모멘트로 방향 추정
        4. rBRIEF로 256비트(32바이트) 이진 설명자 생성

    특징:
        - SIFT보다 훨씬 빠름 (실시간 처리 적합)
        - 특허 없음 (자유롭게 사용 가능)
        - 크기(Scale) 불변성은 SIFT보다 약함
        - 이진 설명자: Hamming 거리로 매칭

    Args:
        image: BGR 이미지
        n_features: 검출할 최대 특징점 수

    Returns:
        (keypoints, descriptors) 튜플
    """
    orb = cv.ORB_create(nfeatures=n_features)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) if image.ndim == 3 else image
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return keypoints, descriptors


def match_features_bf(desc1: np.ndarray, desc2: np.ndarray,
                      method: str = "orb") -> list:
    """BFMatcher(Brute-Force)로 특징점을 매칭합니다.

    Args:
        desc1: 첫 번째 이미지의 설명자
        desc2: 두 번째 이미지의 설명자
        method: 'sift' (L2 노름) 또는 'orb' (Hamming 거리)

    Returns:
        정렬된 매칭 리스트

    Note:
        SIFT/SURF → 실수 설명자 → L2 노름(유클리드 거리) 사용
        ORB/BRIEF  → 이진 설명자 → Hamming 거리 사용
    """
    norm = cv.NORM_HAMMING if method == "orb" else cv.NORM_L2
    bf = cv.BFMatcher(norm, crossCheck=True)
    matches = bf.match(desc1, desc2)
    # 거리 기준 오름차순 정렬
    return sorted(matches, key=lambda x: x.distance)


def match_features_knn(desc1: np.ndarray, desc2: np.ndarray,
                       k: int = 2, ratio: float = 0.75) -> list:
    """KNN 매칭 + Lowe's ratio test로 정확한 매칭을 선택합니다.

    Lowe's Ratio Test:
        각 특징점에서 가장 가까운 k개의 매칭 후보 선택
        1등과 2등 매칭의 거리 비율이 ratio 미만인 경우만 좋은 매칭으로 인정
        → 모호한 매칭 제거

    Args:
        desc1: 첫 번째 이미지의 설명자
        desc2: 두 번째 이미지의 설명자
        k: KNN에서 선택할 이웃 수
        ratio: Lowe's ratio test 임계값 (일반적으로 0.7–0.8)

    Returns:
        검증된 매칭 리스트
    """
    bf = cv.BFMatcher()
    knn_matches = bf.knnMatch(desc1, desc2, k=k)

    # Lowe's ratio test
    good_matches = [m for m, n in knn_matches if m.distance < ratio * n.distance]
    return good_matches


def visualize_keypoints(image: np.ndarray, keypoints: list, title: str = "") -> np.ndarray:
    """특징점을 이미지에 그립니다.

    Returns:
        특징점이 표시된 이미지
    """
    result = cv.drawKeypoints(
        image, keypoints, None,
        flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    return result


def visualize_matches(img1: np.ndarray, kp1: list,
                      img2: np.ndarray, kp2: list,
                      matches: list, max_matches: int = 30) -> np.ndarray:
    """매칭 결과를 시각화합니다.

    Args:
        img1, img2: 매칭할 이미지 쌍
        kp1, kp2: 각 이미지의 키포인트
        matches: 매칭 리스트
        max_matches: 표시할 최대 매칭 수

    Returns:
        매칭이 시각화된 이미지
    """
    return cv.drawMatches(
        img1, kp1, img2, kp2,
        matches[:max_matches], None,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )


def compare_sift_orb(image: np.ndarray) -> None:
    """SIFT와 ORB 특징점 검출 결과를 비교합니다."""
    import time

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) if image.ndim == 3 else image

    # SIFT 시간 측정
    start = time.time()
    kp_sift, desc_sift = detect_and_describe_sift(image)
    sift_time = time.time() - start

    # ORB 시간 측정
    start = time.time()
    kp_orb, desc_orb = detect_and_describe_orb(image)
    orb_time = time.time() - start

    print("=== SIFT vs ORB 비교 ===")
    print(f"SIFT: {len(kp_sift)}개 특징점, 처리 시간: {sift_time:.3f}초")
    print(f"ORB : {len(kp_orb)}개 특징점, 처리 시간: {orb_time:.3f}초")
    print(f"속도 비율: ORB가 SIFT보다 약 {sift_time/orb_time:.1f}배 빠름")

    img_sift = visualize_keypoints(image.copy(), kp_sift)
    img_orb = visualize_keypoints(image.copy(), kp_orb)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("SIFT vs ORB 특징점 비교", fontsize=14)

    axes[0].imshow(cv.cvtColor(img_sift, cv.COLOR_BGR2RGB))
    axes[0].set_title(f"SIFT ({len(kp_sift)}개, {sift_time:.3f}초)")
    axes[0].axis("off")

    axes[1].imshow(cv.cvtColor(img_orb, cv.COLOR_BGR2RGB))
    axes[1].set_title(f"ORB ({len(kp_orb)}개, {orb_time:.3f}초)")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


def demo_feature_matching() -> None:
    """두 이미지 간 특징점 매칭을 시연합니다."""
    # 원본 및 변환된 샘플 이미지 생성
    img1 = np.zeros((300, 400, 3), dtype=np.uint8)
    cv.rectangle(img1, (50, 50), (200, 200), (200, 150, 100), -1)
    cv.circle(img1, (300, 150), 70, (100, 200, 150), -1)
    cv.putText(img1, "Feature", (60, 270), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # 회전 + 노이즈를 추가한 두 번째 이미지
    M = cv.getRotationMatrix2D((200, 150), 15, 0.9)
    img2 = cv.warpAffine(img1, M, (400, 300))
    noise = np.random.randint(0, 20, img2.shape, dtype=np.uint8)
    img2 = cv.add(img2, noise)

    print("=== 특징점 매칭 시연 ===")
    compare_sift_orb(img1)

    # ORB 매칭
    kp1, desc1 = detect_and_describe_orb(img1)
    kp2, desc2 = detect_and_describe_orb(img2)

    if desc1 is not None and desc2 is not None:
        matches = match_features_bf(desc1, desc2, method="orb")
        match_img = visualize_matches(img1, kp1, img2, kp2, matches, max_matches=20)

        plt.figure(figsize=(14, 5))
        plt.imshow(cv.cvtColor(match_img, cv.COLOR_BGR2RGB))
        plt.title(f"ORB BFMatcher 매칭 결과 (상위 20개, 전체 {len(matches)}개)")
        plt.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    demo_feature_matching()
