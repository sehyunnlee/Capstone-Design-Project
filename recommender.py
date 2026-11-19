# ============================================================
# recommender.py
#
# Vehicle Detector Hyperparameter Recommender
#
# Features:
#   1. spatial_frequency    (공간 주파수)
#   2. brightness           (밝기)
#   3. saturation           (채도)
#
# Input:
#   image_paths : list of image file paths
#
# Output:
#   dict
#     - environment similarity
#     - dominant environment
#     - recommended LR
#     - recommended image size
#     - expected mAP50-95
#     - Top-K combinations
#
# Required files:
#   algorithm_data/
#     ├── final_lr_image_grid.csv
#     └── sf_brightness_saturation_prototypes.json
# ============================================================

from pathlib import Path
import math
import json

import cv2
import numpy as np
import pandas as pd


# ============================================================
# 1. 내부 파일 경로
# ============================================================

BASE_DIR = Path(
    __file__
).resolve().parent


DATA_DIR = (
    BASE_DIR
    / "algorithm_data"
)


PERFORMANCE_CSV = (
    DATA_DIR
    / "final_lr_image_grid.csv"
)


PROTOTYPE_PATH = (
    DATA_DIR
    / "sf_brightness_saturation_prototypes.json"
)


# ============================================================
# 2. 지원 이미지 형식
# ============================================================

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}


# ============================================================
# 3. LR 설정
# ============================================================

LR_CONFIG = {

    0.5: {
        "main_lr":
            0.00025,

        "backbone_lr":
            0.0000125,
    },

    1.0: {
        "main_lr":
            0.0005,

        "backbone_lr":
            0.000025,
    },

    1.5: {
        "main_lr":
            0.00075,

        "backbone_lr":
            0.0000375,
    },

    2.0: {
        "main_lr":
            0.001,

        "backbone_lr":
            0.00005,
    },
}


LR_VALUES = [
    0.5,
    1.0,
    1.5,
    2.0,
]


IMAGE_SIZES = [
    640,
    800,
    960,
    1024,
]


EXPECTED_ENVIRONMENTS = [
    "DAY_CLEAR_NEW",
    "NIGHT",
    "WEATHER",
]


# ============================================================
# 4. 데이터 로딩
# ============================================================

def _load_algorithm_data():

    if not PERFORMANCE_CSV.exists():

        raise FileNotFoundError(
            f"Performance CSV not found:\n"
            f"{PERFORMANCE_CSV}"
        )


    if not PROTOTYPE_PATH.exists():

        raise FileNotFoundError(
            f"Prototype JSON not found:\n"
            f"{PROTOTYPE_PATH}"
        )


    # --------------------------------------------------------
    # Prototype load
    # --------------------------------------------------------

    prototype_data = json.loads(
        PROTOTYPE_PATH.read_text(encoding="utf-8")
    )


    analysis_size = int(
        prototype_data["analysis_size"]
    )


    global_sf_std = float(
        prototype_data["global_feature_scale"]["spatial_frequency_std"]
    )


    global_brightness_std = float(
        prototype_data["global_feature_scale"]["brightness_std"]
    )


    global_saturation_std = float(
        prototype_data["global_feature_scale"]["saturation_std"]
    )


    prototypes = {}


    for env, values in (
        prototype_data["prototypes"].items()
    ):

        prototypes[env] = {

            "spatial_frequency":
                float(
                    values["spatial_frequency"]["mean"]
                ),

            "brightness":
                float(
                    values["brightness"]["mean"]
                ),

            "saturation":
                float(
                    values["saturation"]["mean"]
                ),
        }


    if (
        set(prototypes.keys()) != set(EXPECTED_ENVIRONMENTS)
    ):

        raise ValueError(
            "Prototype environment names do not match "
            "DAY_CLEAR_NEW / NIGHT / WEATHER."
        )


    # --------------------------------------------------------
    # Performance grid load
    # --------------------------------------------------------

    performance_df = pd.read_csv(
        PERFORMANCE_CSV
    )


    required_columns = [
        "dataset",
        "lr_multiplier",
        "image_size",
        "mAP50_95",
        "status",
    ]


    for column in required_columns:

        if column not in performance_df.columns:

            raise ValueError(
                f"Required CSV column missing: "
                f"{column}"
            )


    performance_df = (
        performance_df[
            performance_df["status"] == "COMPLETE"
        ].copy()
    )


    expected_count = (
        len(EXPECTED_ENVIRONMENTS)
        *
        len(LR_VALUES)
        *
        len(IMAGE_SIZES)
    )


    if len(performance_df) != expected_count:
        raise ValueError(
            "Performance grid is incomplete.\n"
            f"Expected: {expected_count}\n"
            f"Found   : {len(performance_df)}"
        )


    if not (performance_df["mAP50_95"]
        .notna()
        .all()
    ):

        raise ValueError(
            "mAP50_95 contains missing values."
        )


    return {
        "analysis_size":
            analysis_size,

        "global_sf_std":
            global_sf_std,

        "global_brightness_std":
            global_brightness_std,

        "global_saturation_std":
            global_saturation_std,

        "prototypes":
            prototypes,

        "performance_df":
            performance_df,
    }


# ============================================================
# 5. 모듈 import 시 1회 로드
# ============================================================

_ALGORITHM_DATA = (
    _load_algorithm_data()
)


ANALYSIS_SIZE = (
    _ALGORITHM_DATA["analysis_size"]
)


GLOBAL_SF_STD = (
    _ALGORITHM_DATA["global_sf_std"]
)


GLOBAL_BRIGHTNESS_STD = (
    _ALGORITHM_DATA["global_brightness_std"]
)


GLOBAL_SATURATION_STD = (
    _ALGORITHM_DATA["global_saturation_std"]
)


PROTOTYPES = (
    _ALGORITHM_DATA["prototypes"]
)


PERFORMANCE_DF = (
    _ALGORITHM_DATA["performance_df"]
)


# ============================================================
# 6. 입력 이미지 검증
# ============================================================

def _validate_image_paths(
    image_paths,
):

    if image_paths is None:

        raise ValueError("image_paths is required.")


    paths = [
        Path(path)
        for path in image_paths
    ]


    if not (1 <= len(paths) <= 5):
        raise ValueError(
            f"1 to 5 sample images are required. "
            f"Received: {len(paths)}"
        )


    for path in paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Image file not found:\n"
                f"{path}"
            )


        if (
            path.suffix.lower()
            not in SUPPORTED_EXTENSIONS
        ):
            raise ValueError(
                f"Unsupported image format: "
                f"{path.suffix}\n"
                f"Supported: jpg, jpeg, png"
            )


    return paths


# ============================================================
# 7. 이미지 특징 추출
# ============================================================

def _calculate_features(image_path, ):

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR, )


    if image is None:
        raise ValueError(
            f"Failed to read image:\n"
            f"{image_path}"
        )


    # --------------------------------------------------------
    # 동일 크기로 정규화
    # --------------------------------------------------------

    image = cv2.resize(
        image,
        (ANALYSIS_SIZE, ANALYSIS_SIZE, ),
        interpolation=cv2.INTER_AREA,
    )


    # --------------------------------------------------------
    # Brightness
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )


    gray_float = (
        gray.astype(np.float32) / 255.0
    )


    brightness = float(
        np.mean(gray_float)
    )


    # --------------------------------------------------------
    # Saturation
    # --------------------------------------------------------

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV, )


    saturation = float(
        np.mean(
            hsv[:, :, 1].astype(np.float32) / 255.0
        )
    )


    # --------------------------------------------------------
    # Spatial Frequency
    # --------------------------------------------------------

    row_diff = (
        gray_float[:, 1:] - gray_float[:, :-1]
    )


    col_diff = (
        gray_float[1:, :] - gray_float[:-1, :]
    )


    row_frequency = math.sqrt(
        float(
            np.mean(row_diff ** 2)
        )
    )


    column_frequency = math.sqrt(
        float(
            np.mean(col_diff ** 2)
        )
    )


    spatial_frequency = math.sqrt(
        row_frequency ** 2
        +
        column_frequency ** 2
    )


    return {
        "spatial_frequency":
            spatial_frequency,

        "brightness":
            brightness,

        "saturation":
            saturation,
    }


# ============================================================
# 8. 단일 이미지 환경 유사도
# ============================================================

def _calculate_environment_weights(
    spatial_frequency,
    brightness,
    saturation,
):

    distances = {}


    for env, prototype in (
        PROTOTYPES.items()
    ):

        sf_distance = (
            spatial_frequency
            -
            prototype["spatial_frequency"]
        ) / GLOBAL_SF_STD


        brightness_distance = (
            brightness
            -
            prototype["brightness"]
        ) / GLOBAL_BRIGHTNESS_STD


        saturation_distance = (
            saturation
            -
            prototype["saturation"]
        ) / GLOBAL_SATURATION_STD


        distance = math.sqrt(
            sf_distance ** 2
            +
            brightness_distance ** 2
            +
            saturation_distance ** 2
        )


        distances[env] = distance


    # --------------------------------------------------------
    # Softmax(-distance)
    # --------------------------------------------------------

    similarities = {

        env:
            math.exp(-distance)

        for env, distance
        in distances.items()
    }


    total = sum(similarities.values())


    weights = {

        env:
            value / total

        for env, value
        in similarities.items()
    }


    return (distances, weights, )


# ============================================================
# 9. LR × Image Size 조합 점수 계산
# ============================================================

def _score_hyperparameter_grid(
    environment_weights,
):

    results = []


    for lr in LR_VALUES:

        for image_size in IMAGE_SIZES:

            weighted_map = 0.0

            environment_maps = {}


            for env in (
                EXPECTED_ENVIRONMENTS
            ):

                selected = (
                    PERFORMANCE_DF[
                        (PERFORMANCE_DF["dataset"] == env)
                        &
                        (PERFORMANCE_DF["lr_multiplier"] == lr)
                        &
                        (PERFORMANCE_DF["image_size"] == image_size)
                    ]
                )


                if len(selected) != 1:
                    raise ValueError(
                        "Performance grid lookup failed: "
                        f"{env}, "
                        f"LR={lr}, "
                        f"IMG={image_size}"
                    )


                map_value = float(
                    selected.iloc[0]["mAP50_95"]
                )


                environment_maps[env] = map_value


                weighted_map += (environment_weights[env] * map_value)


            results.append(
                {

                    "lr_multiplier":
                        lr,

                    "main_lr":
                        LR_CONFIG[lr]["main_lr"],

                    "backbone_lr":
                        LR_CONFIG[lr]["backbone_lr"],

                    "image_size":
                        image_size,

                    "expected_map50_95":
                        weighted_map,

                    "environment_maps":
                        environment_maps,
                }
            )


    results.sort(
        key=lambda item:
            item["expected_map50_95"],
        reverse=True,
    )


    return results


# ============================================================
# 10. 외부 공개 함수
# ============================================================

def recommend(
    image_paths,
    top_k=3,
):

    """
    Recommend learning rate and image size
    from exactly 5 sample images.

    Parameters
    ----------
    image_paths : list[str | Path]
        Exactly 5 jpg/jpeg/png image paths.

    top_k : int
        Number of ranked hyperparameter combinations
        returned in result["top_k"].

    Returns
    -------
    dict
        Recommendation result.
    """


    # --------------------------------------------------------
    # 입력 검증
    # --------------------------------------------------------

    image_paths = (
        _validate_image_paths(image_paths)
    )


    if top_k < 1:

        raise ValueError("top_k must be >= 1.")


    top_k = min(
        int(top_k),
        len(LR_VALUES) * len(IMAGE_SIZES),
    )


    # --------------------------------------------------------
    # 개별 이미지 분석
    # --------------------------------------------------------

    individual_results = []


    final_weights = {

        "DAY_CLEAR_NEW":
            0.0,

        "NIGHT":
            0.0,

        "WEATHER":
            0.0,
    }


    sf_values = []
    brightness_values = []
    saturation_values = []


    for image_path in (image_paths):

        features = (
            _calculate_features(image_path)
        )


        sf = features["spatial_frequency"]

        brightness = features["brightness"]

        saturation = features["saturation"]


        sf_values.append(sf)

        brightness_values.append(brightness)

        saturation_values.append(saturation)


        (distances, weights, ) = (
            _calculate_environment_weights(
                sf,
                brightness,
                saturation,
            )
        )


        nearest_environment = min(
            distances,
            key = distances.get,
        )


        for env in (
            EXPECTED_ENVIRONMENTS
        ):

            final_weights[env] += (weights[env])


        individual_results.append(
            {

                "image":
                    image_path.name,

                "spatial_frequency":
                    float(sf),

                "brightness":
                    float(brightness),

                "saturation":
                    float(saturation),

                "environment_distance":
                    {
                        env:
                            float(value)

                        for env, value
                        in distances.items()
                    },

                "environment_similarity":
                    {
                        env:
                            float(value)

                        for env, value
                        in weights.items()
                    },

                "nearest_environment":
                    nearest_environment,
            }
        )


    # --------------------------------------------------------
    # 이미지별 similarity 평균
    # --------------------------------------------------------

    num_images = len(image_paths)


    final_weights = {

        env:
            value / num_images

        for env, value
        in final_weights.items()
    }


    # floating point 보정
    total_weight = sum(
        final_weights.values()
    )


    final_weights = {

        env:
            value / total_weight

        for env, value
        in final_weights.items()
    }


    # --------------------------------------------------------
    # 참고용 sample 평균 특징
    # --------------------------------------------------------

    sample_sf_mean = float(
        np.mean(sf_values)
    )


    sample_sf_std = float(
        np.std(sf_values)
    )


    sample_brightness_mean = float(
        np.mean(brightness_values)
    )


    sample_brightness_std = float(
        np.std(brightness_values)
    )


    sample_saturation_mean = float(
        np.mean(saturation_values)
    )


    sample_saturation_std = float(
        np.std(saturation_values)
    )


    # --------------------------------------------------------
    # Environment label
    # --------------------------------------------------------

    max_env = max(
        final_weights,
        key=final_weights.get,
    )


    max_weight = (
        final_weights[max_env]
    )


    if max_weight < 0.45:

        dominant_environment = ("MIXED")

    else:

        dominant_environment = (max_env)


    # --------------------------------------------------------
    # Hyperparameter ranking
    # --------------------------------------------------------

    ranking = (
        _score_hyperparameter_grid(final_weights)
    )


    best = ranking[0]


    # --------------------------------------------------------
    # Recommendation margin
    # --------------------------------------------------------

    score_margin = (
        best["expected_map50_95"]
        -
        ranking[1]["expected_map50_95"]
    )


    if score_margin >= 0.01:

        recommendation_margin = ("HIGH")

    elif score_margin >= 0.005:

        recommendation_margin = ("MEDIUM")

    else:

        recommendation_margin = ("LOW")


    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    result = {

        "num_sample_images":
            num_images,

        "sample_spatial_frequency":
            sample_sf_mean,

        "sample_spatial_frequency_std":
            sample_sf_std,

        "sample_brightness":
            sample_brightness_mean,

        "sample_brightness_std":
            sample_brightness_std,

        "sample_saturation":
            sample_saturation_mean,

        "sample_saturation_std":
            sample_saturation_std,

        "individual_results":
            individual_results,

        "prototype_features":
            PROTOTYPES,

        "environment_similarity":
            {
                env:
                    float(weight)

                for env, weight
                in final_weights.items()
            },

        "dominant_environment":
            dominant_environment,

        "recommended": {

            "lr_multiplier":
                best["lr_multiplier"],

            "main_lr":
                best["main_lr"],

            "backbone_lr":
                best["backbone_lr"],

            "image_size":
                best["image_size"],

            "expected_map50_95":
                best["expected_map50_95"],
        },

        "score_margin_top1_top2":
            score_margin,

        "recommendation_margin":
            recommendation_margin,

        "top_k":
            ranking[:top_k],
    }


    return result
