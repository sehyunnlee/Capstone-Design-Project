# Capstone Design Project

차량 객체 검지 모델의 하이퍼파라미터를 추천하기 위한 캡스톤 디자인 프로젝트입니다.

본 모듈은 샘플 이미지 5장의 시각적 특성을 분석하여 다음 항목을 추천합니다.

- Learning Rate
- Backbone Learning Rate
- Input Image Size

## 알고리즘 개요

추천 알고리즘은 입력 이미지에서 다음 3가지 특징을 추출합니다.

1. 공간주파수(Spatial Frequency)
2. 밝기(Brightness)
3. 채도(Saturation)

각 샘플 이미지는 다음 3가지 기준 환경과 비교됩니다.

- DAY_CLEAR_NEW
- NIGHT
- WEATHER

각 이미지별 환경 유사도를 계산한 뒤, 5장의 유사도를 평균하여 최종 환경 가중치를 생성합니다.

이후 실제 실험을 통해 측정한 각 환경별 mAP@0.5:0.95 결과를 이용하여 모든 Learning Rate × Image Size 조합의 예상 성능을 계산합니다.

현재 평가 대상 조합은 다음과 같습니다.

- Learning Rate 배수: 0.5x, 1.0x, 1.5x, 2.0x
- Image Size: 640, 800, 960, 1024

따라서 총 16개의 Learning Rate × Image Size 조합을 비교하여 가장 높은 예상 성능을 가진 조합을 추천합니다.

## 프로젝트 구조

```text
Capstone-Design-Project/
├── README.md
├── recommender.py
└── algorithm_data/
    ├── final_lr_image_grid.csv
    └── sf_brightness_saturation_prototypes.json
```

## 필요 라이브러리

```bash
pip install opencv-python numpy pandas
```

## 웹 UI 실행 방법 (Streamlit)

1. 패키지 설치
```bash
pip install -r requirements.txt

## 사용 방법

```python
from recommender import recommend

image_paths = [
    "sample/1.jpg",
    "sample/2.jpg",
    "sample/3.png",
    "sample/4.jpeg",
    "sample/5.png",
]

result = recommend(
    image_paths,
    top_k=3
)
```

샘플 이미지는 정확히 5장이 필요합니다.

지원 이미지 형식:

- `.jpg`
- `.jpeg`
- `.png`

## 주요 출력값

최종 추천 결과는 다음과 같이 확인할 수 있습니다.

```python
best = result["recommended"]

print(best["lr_multiplier"])
print(best["main_lr"])
print(best["backbone_lr"])
print(best["image_size"])
print(best["expected_map50_95"])
```

예시:

```python
{
    "lr_multiplier": 1.0,
    "main_lr": 0.0005,
    "backbone_lr": 0.000025,
    "image_size": 960,
    "expected_map50_95": 0.7573
}
```

## 환경 유사도

```python
result["environment_similarity"]
```

예시:

```python
{
    "DAY_CLEAR_NEW": 0.49,
    "NIGHT": 0.21,
    "WEATHER": 0.30
}
```

각 값은 입력 샘플 이미지가 기존 실험 환경과 얼마나 유사한지를 나타냅니다.

## 대표 환경

```python
result["dominant_environment"]
```

가능한 출력값:

```text
DAY_CLEAR_NEW
NIGHT
WEATHER
MIXED
```

특정 환경의 유사도가 설정된 기준 이상으로 높지 않을 경우 `MIXED`로 표시됩니다.

현재 기준은 최대 환경 유사도 45%입니다.

## Recommendation Margin

```python
result["recommendation_margin"]
```

가능한 출력값:

```text
HIGH
MEDIUM
LOW
```

Recommendation Margin은 1위 추천 조합과 2위 추천 조합의 예상 mAP 차이를 기반으로 계산합니다.

현재 기준:

```text
HIGH   : 1.0%p 이상
MEDIUM : 0.5%p 이상
LOW    : 0.5%p 미만
```

이 값은 통계적 신뢰도를 의미하지 않으며, 추천 조합 간의 상대적인 성능 차이를 나타내는 휴리스틱 지표입니다.

## Top-K 추천

```python
result["top_k"]
```

`top_k=3`으로 설정하면 예상 성능이 높은 상위 3개의 Learning Rate × Image Size 조합이 반환됩니다.

## 알고리즘 데이터

### `final_lr_image_grid.csv`

다음 실험 결과를 저장한 파일입니다.

- 3개 환경
- 4개 Learning Rate
- 4개 Image Size

총 실험 조합:

```text
3 × 4 × 4 = 48
```

추천 알고리즘에서는 각 조합의 validation mAP@0.5:0.95 값을 사용합니다.

### `sf_brightness_saturation_prototypes.json`

다음 3개 환경의 대표 이미지 특징값을 저장한 파일입니다.

- DAY_CLEAR_NEW
- NIGHT
- WEATHER

저장되는 특징:

- 공간주파수(Spatial Frequency)
- 밝기(Brightness)
- 채도(Saturation)

입력 이미지의 특징값과 각 환경의 대표값 사이의 거리를 계산하여 환경별 유사도를 산출합니다.

## 추천 알고리즘 흐름

```text
샘플 이미지 5장
        ↓
특징 추출
(SF + Brightness + Saturation)
        ↓
이미지별 환경 유사도 계산
        ↓
5장 유사도 평균
        ↓
환경별 가중치 생성
        ↓
48개 실험 결과 참조
        ↓
16개 LR × Image Size 조합의
가중 예상 mAP 계산
        ↓
성능 순위 정렬
        ↓
최적 하이퍼파라미터 추천
```

## 주요 계산 방식

각 환경에 대해 입력 이미지와의 특징 거리를 계산합니다.

사용 특징:

```text
Spatial Frequency
Brightness
Saturation
```

각 특징은 전체 데이터 분포의 표준편차를 이용하여 정규화한 뒤, 유클리드 거리를 계산합니다.

이후 거리값을 기반으로 환경 유사도 가중치를 생성합니다.

최종적으로 각 하이퍼파라미터 조합의 예상 성능은 다음과 같이 계산됩니다.

```text
예상 성능
=
DAY_CLEAR_NEW 유사도 × DAY_CLEAR_NEW mAP
+
NIGHT 유사도 × NIGHT mAP
+
WEATHER 유사도 × WEATHER mAP
```

16개의 모든 Learning Rate × Image Size 조합에 대해 이 값을 계산한 후 가장 높은 조합을 최종 추천합니다.

## 참고 사항

- 본 모듈은 객체 검지 모델을 직접 재학습하지 않습니다.
- 추천 결과는 기존 실험에서 측정한 validation 성능을 기반으로 계산됩니다.
- `expected_map50_95`는 세 기준 환경의 실제 측정 결과를 가중 결합한 예상값입니다.
- 추천 조합 간 성능 차이가 매우 작을 경우 강한 우위를 의미하지 않습니다.
- Recommendation Margin은 통계적 유의성을 의미하지 않습니다.
- 현재 알고리즘은 본 캡스톤 프로젝트에서 사용한 데이터셋 및 실험 조건을 기준으로 설계되었습니다.
- UI와 추천 알고리즘은 분리되어 있으며, 외부 프로그램에서는 `recommend()` 함수만 호출하면 사용할 수 있습니다.
