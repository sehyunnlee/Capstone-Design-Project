# ============================================================
# recommender.py
#
# Vehicle Detector Hyperparameter Recommender
# (Hybrid: Gemini Vision API + Visual Statistics Fallback)
# ============================================================

from pathlib import Path
import math
import json
import os

import cv2
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


# ============================================================
# 1. 내부 파일 경로 및 설정
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "algorithm_data"
PERFORMANCE_CSV = DATA_DIR / "final_lr_image_grid.csv"
PROTOTYPE_PATH = DATA_DIR / "sf_brightness_saturation_prototypes.json"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

LR_CONFIG = {
    0.5: {"main_lr": 0.00025, "backbone_lr": 0.0000125},
    1.0: {"main_lr": 0.0005,  "backbone_lr": 0.000025},
    1.5: {"main_lr": 0.00075, "backbone_lr": 0.0000375},
    2.0: {"main_lr": 0.001,   "backbone_lr": 0.00005},
}

LR_VALUES = [0.5, 1.0, 1.5, 2.0]
IMAGE_SIZES = [640, 800, 960, 1024]
EXPECTED_ENVIRONMENTS = ["DAY_CLEAR_NEW", "NIGHT", "WEATHER"]


# ============================================================
# 2. Gemini Structured Output 스키마 정의
# ============================================================

class EnvironmentWeightsSchema(BaseModel):
    DAY_CLEAR_NEW: float = Field(description="주간-맑음 환경 확률 (0.0 ~ 1.0)")
    NIGHT: float = Field(description="야간-맑음 환경 확률 (0.0 ~ 1.0)")
    WEATHER: float = Field(description="악천후(비, 눈, 안개, 젖은 노면 반사 등) 환경 확률 (0.0 ~ 1.0)")
    reason: str = Field(description="환경 판정 근거 한 줄 요약")


# ============================================================
# 3. 기준 데이터 로딩
# ============================================================

def _load_algorithm_data():
    if not PERFORMANCE_CSV.exists():
        raise FileNotFoundError(f"Performance CSV not found:\n{PERFORMANCE_CSV}")
    if not PROTOTYPE_PATH.exists():
        raise FileNotFoundError(f"Prototype JSON not found:\n{PROTOTYPE_PATH}")

    prototype_data = json.loads(PROTOTYPE_PATH.read_text(encoding="utf-8"))

    analysis_size = int(prototype_data["analysis_size"])
    global_sf_std = float(prototype_data["global_feature_scale"]["spatial_frequency_std"])
    global_brightness_std = float(prototype_data["global_feature_scale"]["brightness_std"])
    global_saturation_std = float(prototype_data["global_feature_scale"]["saturation_std"])

    prototypes = {}
    for env, values in prototype_data["prototypes"].items():
        prototypes[env] = {
            "spatial_frequency": float(values["spatial_frequency"]["mean"]),
            "brightness": float(values["brightness"]["mean"]),
            "saturation": float(values["saturation"]["mean"]),
        }

    if set(prototypes.keys()) != set(EXPECTED_ENVIRONMENTS):
        raise ValueError("Prototype environment names do not match DAY_CLEAR_NEW / NIGHT / WEATHER.")

    performance_df = pd.read_csv(PERFORMANCE_CSV)
    required_columns = ["dataset", "lr_multiplier", "image_size", "mAP50_95", "status"]
    for column in required_columns:
        if column not in performance_df.columns:
            raise ValueError(f"Required CSV column missing: {column}")

    performance_df = performance_df[performance_df["status"] == "COMPLETE"].copy()
    expected_count = len(EXPECTED_ENVIRONMENTS) * len(LR_VALUES) * len(IMAGE_SIZES)

    if len(performance_df) != expected_count:
        raise ValueError(f"Performance grid is incomplete. Expected: {expected_count}, Found: {len(performance_df)}")
    if not performance_df["mAP50_95"].notna().all():
        raise ValueError("mAP50_95 contains missing values.")

    return {
        "analysis_size": analysis_size,
        "global_sf_std": global_sf_std,
        "global_brightness_std": global_brightness_std,
        "global_saturation_std": global_saturation_std,
        "prototypes": prototypes,
        "performance_df": performance_df,
    }


_ALGORITHM_DATA = _load_algorithm_data()
ANALYSIS_SIZE = _ALGORITHM_DATA["analysis_size"]
GLOBAL_SF_STD = _ALGORITHM_DATA["global_sf_std"]
GLOBAL_BRIGHTNESS_STD = _ALGORITHM_DATA["global_brightness_std"]
GLOBAL_SATURATION_STD = _ALGORITHM_DATA["global_saturation_std"]
PROTOTYPES = _ALGORITHM_DATA["prototypes"]
PERFORMANCE_DF = _ALGORITHM_DATA["performance_df"]


# ============================================================
# 4. 입력 이미지 검증 (1~5장 지원)
# ============================================================

def _validate_image_paths(image_paths):
    if image_paths is None:
        raise ValueError("image_paths is required.")

    paths = [Path(path) for path in image_paths]
    if not (1 <= len(paths) <= 5):
        raise ValueError(f"1 to 5 sample images are required. Received: {len(paths)}")

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Image file not found:\n{path}")
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported image format: {path.suffix} (Supported: jpg, jpeg, png)")

    return paths


# ============================================================
# 5. 물리적 시각 특징 추출 (통계 및 상세표 출력용)
# ============================================================

def _calculate_features(image_path: Path):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image:\n{image_path}")

    image = cv2.resize(image, (ANALYSIS_SIZE, ANALYSIS_SIZE), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_float = gray.astype(np.float32) / 255.0
    brightness = float(np.mean(gray_float))

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = float(np.mean(hsv[:, :, 1].astype(np.float32) / 255.0))

    row_diff = gray_float[:, 1:] - gray_float[:, :-1]
    col_diff = gray_float[1:, :] - gray_float[:-1, :]
    row_frequency = math.sqrt(float(np.mean(row_diff ** 2)))
    column_frequency = math.sqrt(float(np.mean(col_diff ** 2)))
    spatial_frequency = math.sqrt(row_frequency ** 2 + column_frequency ** 2)

    return {
        "spatial_frequency": spatial_frequency,
        "brightness": brightness,
        "saturation": saturation,
    }


# ============================================================
# 6-A. 통계 기반 환경 유사도 계산 (Fallback용)
# ============================================================

def _calculate_weights_statistical(spatial_frequency, brightness, saturation):
    distances = {}
    for env, prototype in PROTOTYPES.items():
        sf_distance = (spatial_frequency - prototype["spatial_frequency"]) / GLOBAL_SF_STD
        brightness_distance = (brightness - prototype["brightness"]) / GLOBAL_BRIGHTNESS_STD
        saturation_distance = (saturation - prototype["saturation"]) / GLOBAL_SATURATION_STD
        distances[env] = math.sqrt(sf_distance ** 2 + brightness_distance ** 2 + saturation_distance ** 2)

    similarities = {env: math.exp(-dist) for env, dist in distances.items()}
    total = sum(similarities.values())
    weights = {env: val / total for env, val in similarities.items()}
    return distances, weights


# ============================================================
# 6-B. Gemini Vision API 기반 환경 유사도 계산 (기본 엔진)
# ============================================================

def _calculate_weights_llm(image_path: Path):
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not (GENAI_AVAILABLE and api_key):
        return None

    try:
        client = genai.Client(api_key=api_key)
        image_bytes = image_path.read_bytes()
        mime_type = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"

        prompt = (
            "당신은 도로 관제 CCTV 영상 환경 분류 전문가입니다. "
            "첨부된 도로 이미지를 분석하여 다음 3가지 환경에 해당하는 확률을 산출하세요.\n"
            "1. DAY_CLEAR_NEW: 주간이며 날씨가 맑고 노면이 건조한 환경\n"
            "2. NIGHT: 야간이나 새벽이지만 비·눈·안개 없이 시야가 맑고 노면이 건조한 환경\n"
            "3. WEATHER: 비, 눈, 안개, 흐림, 적설, 또는 빗물로 인해 젖은 노면 빛 반사가 있는 악천후 환경\n"
            "세 확률(DAY_CLEAR_NEW, NIGHT, WEATHER)의 합은 반드시 1.0이 되어야 합니다."
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                prompt,
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=EnvironmentWeightsSchema,
                temperature=0.1,
            ),
        )

        parsed = json.loads(response.text)
        weights = {
            "DAY_CLEAR_NEW": max(0.0, float(parsed["DAY_CLEAR_NEW"])),
            "NIGHT": max(0.0, float(parsed["NIGHT"])),
            "WEATHER": max(0.0, float(parsed["WEATHER"])),
        }

        total = sum(weights.values())
        if total <= 0:
            return None

        weights = {env: val / total for env, val in weights.items()}
        # 확률의 역변환 개념으로 참고용 가상 거리 생성
        distances = {env: round(-math.log(max(val, 1e-6)), 4) for env, val in weights.items()}
        return distances, weights, parsed.get("reason", "")

    except Exception as e:
        print(f"[Warning] Gemini Vision API 호출 실패로 통계 모델로 전환합니다 ({image_path.name}): {e}")
        return None


# ============================================================
# 7. LR × Image Size 조합 점수 계산
# ============================================================

def _score_hyperparameter_grid(environment_weights):
    results = []
    for lr in LR_VALUES:
        for image_size in IMAGE_SIZES:
            weighted_map = 0.0
            environment_maps = {}

            for env in EXPECTED_ENVIRONMENTS:
                selected = PERFORMANCE_DF[
                    (PERFORMANCE_DF["dataset"] == env)
                    & (PERFORMANCE_DF["lr_multiplier"] == lr)
                    & (PERFORMANCE_DF["image_size"] == image_size)
                ]
                if len(selected) != 1:
                    raise ValueError(f"Performance grid lookup failed: {env}, LR={lr}, IMG={image_size}")

                map_value = float(selected.iloc[0]["mAP50_95"])
                environment_maps[env] = map_value
                weighted_map += environment_weights[env] * map_value

            results.append({
                "lr_multiplier": lr,
                "main_lr": LR_CONFIG[lr]["main_lr"],
                "backbone_lr": LR_CONFIG[lr]["backbone_lr"],
                "image_size": image_size,
                "expected_map50_95": weighted_map,
                "environment_maps": environment_maps,
            })

    results.sort(key=lambda item: item["expected_map50_95"], reverse=True)
    return results


# ============================================================
# 8. 외부 공개 함수
# ============================================================

def recommend(image_paths, top_k=3):
    image_paths = _validate_image_paths(image_paths)

    if top_k < 1:
        raise ValueError("top_k must be >= 1.")
    top_k = min(int(top_k), len(LR_VALUES) * len(IMAGE_SIZES))

    individual_results = []
    final_weights = {"DAY_CLEAR_NEW": 0.0, "NIGHT": 0.0, "WEATHER": 0.0}
    sf_values, brightness_values, saturation_values = [], [], []
    used_engine = "statistical"

    for image_path in image_paths:
        features = _calculate_features(image_path)
        sf = features["spatial_frequency"]
        brightness = features["brightness"]
        saturation = features["saturation"]

        sf_values.append(sf)
        brightness_values.append(brightness)
        saturation_values.append(saturation)

        # 1순위: Gemini Vision API 시도 / 실패 또는 키 미설정 시 통계 모델 자동 전환
        llm_result = _calculate_weights_llm(image_path)
        if llm_result is not None:
            distances, weights, reason = llm_result
            used_engine = "gemini_vision"
        else:
            distances, weights = _calculate_weights_statistical(sf, brightness, saturation)
            reason = "시각 통계(SF·밝기·채도) 기준 거리 산출"

        nearest_environment = min(distances, key=distances.get)

        for env in EXPECTED_ENVIRONMENTS:
            final_weights[env] += weights[env]

        individual_results.append({
            "image": image_path.name,
            "spatial_frequency": float(sf),
            "brightness": float(brightness),
            "saturation": float(saturation),
            "environment_distance": {env: float(val) for env, val in distances.items()},
            "environment_similarity": {env: float(val) for env, val in weights.items()},
            "nearest_environment": nearest_environment,
            "diagnosis_reason": reason,
        })

    num_images = len(image_paths)
    final_weights = {env: val / num_images for env, val in final_weights.items()}
    total_weight = sum(final_weights.values())
    final_weights = {env: val / total_weight for env, val in final_weights.items()}

    max_env = max(final_weights, key=final_weights.get)
    dominant_environment = "MIXED" if final_weights[max_env] < 0.45 else max_env

    ranking = _score_hyperparameter_grid(final_weights)
    best = ranking[0]

    score_margin = best["expected_map50_95"] - ranking[1]["expected_map50_95"]
    if score_margin >= 0.01:
        recommendation_margin = "HIGH"
    elif score_margin >= 0.005:
        recommendation_margin = "MEDIUM"
    else:
        recommendation_margin = "LOW"

    return {
        "num_sample_images": num_images,
        "diagnosis_engine": used_engine,
        "sample_spatial_frequency": float(np.mean(sf_values)),
        "sample_spatial_frequency_std": float(np.std(sf_values)),
        "sample_brightness": float(np.mean(brightness_values)),
        "sample_brightness_std": float(np.std(brightness_values)),
        "sample_saturation": float(np.mean(saturation_values)),
        "sample_saturation_std": float(np.std(saturation_values)),
        "individual_results": individual_results,
        "prototype_features": PROTOTYPES,
        "environment_similarity": {env: float(w) for env, w in final_weights.items()},
        "dominant_environment": dominant_environment,
        "recommended": {
            "lr_multiplier": best["lr_multiplier"],
            "main_lr": best["main_lr"],
            "backbone_lr": best["backbone_lr"],
            "image_size": best["image_size"],
            "expected_map50_95": best["expected_map50_95"],
        },
        "score_margin_top1_top2": score_margin,
        "recommendation_margin": recommendation_margin,
        "top_k": ranking[:top_k],
    }
