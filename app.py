import tempfile
from pathlib import Path
import pandas as pd
import streamlit as st
from PIL import Image

from recommender import recommend

# ============================================================
# 1. 페이지 기본 설정 및 스타일
# ============================================================
st.set_page_config(
    page_title="차량 객체 검지 하이퍼파라미터 추천 시스템",
    page_icon="🚗",
    layout="wide",
)

st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .highlight-blue {
        color: #2563eb;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 환경명 한글 라벨 매핑
ENV_LABELS = {
    "DAY_CLEAR_NEW": "주간-맑음 (DAY_CLEAR_NEW)",
    "NIGHT": "야간 (NIGHT)",
    "WEATHER": "악천후 (WEATHER)",
    "MIXED": "복합 환경 (MIXED)",
}

MARGIN_DESC = {
    "HIGH": "높음 (1위-2위 격차 ≥ 1.0%p)",
    "MEDIUM": "보통 (1위-2위 격차 ≥ 0.5%p)",
    "LOW": "낮음 (1위-2위 격차 < 0.5%p)",
}

# ============================================================
# 2. 헤더 및 사이드바 안내
# ============================================================
st.title("🚗 차량 객체 검지 최적 하이퍼파라미터 추천 시스템")
st.markdown(
    "도로 샘플 이미지 **5장**을 업로드하면 **공간주파수(SF) · 밝기(Brightness) · 채도(Saturation)**를 분석하여 "
    "최적의 학습률(Learning Rate)과 입력 해상도 조합을 추천합니다."
)

with st.sidebar:
    st.header("⚙️ 알고리즘 설정")
    top_k = st.slider("추천 순위 표시 개수 (Top-K)", min_value=1, max_value=16, value=3)
    st.divider()
    st.subheader("📌 평가 대상 조합")
    st.markdown(
        """
        - **LR 배수:** `0.5x`, `1.0x`, `1.5x`, `2.0x`
        - **해상도:** `640`, `800`, `960`, `1024`
        - **기준 환경:** 주간, 야간, 악천후
        - **총 평가 조합:** 16개 (4 × 4)
        """
    )

# ============================================================
# 3. 이미지 업로드 및 미리보기 섹션
# ============================================================
st.subheader("1. 샘플 도로 이미지 업로드 (정확히 5장)")
uploaded_files = st.file_uploader(
    "지원 형식: JPG, JPEG, PNG (5장을 한 번에 선택해 주세요)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

if uploaded_files:
    num_files = len(uploaded_files)
    if num_files != 5:
        st.warning(f"⚠️ 현재 **{num_files}장**이 선택되었습니다. 정확한 분석을 위해 **정확히 5장**의 이미지를 업로드해 주세요.")
    else:
        st.success("✅ 5장의 샘플 이미지가 준비되었습니다.")

    # 썸네일 미리보기 (최대 5열)
    cols = st.columns(5)
    for idx, file in enumerate(uploaded_files[:5]):
        with cols[idx]:
            img = Image.open(file)
            st.image(img, caption=f"사진 {idx + 1}: {file.name}", use_container_width=True)

# ============================================================
# 4. 분석 실행 및 결과 리포트 출력
# ============================================================
if uploaded_files and len(uploaded_files) == 5:
    if st.button("⚡ 최적 하이퍼파라미터 계산하기", type="primary", use_container_width=True):
        with st.spinner("이미지 특징 추출 및 16개 파라미터 조합 시뮬레이션 중..."):
            try:
                # 업로드된 파일을 임시 경로에 저장하여 recommender.recommend()에 전달
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_paths = []
                    for idx, uploaded_file in enumerate(uploaded_files):
                        ext = Path(uploaded_file.name).suffix.lower()
                        temp_file_path = Path(temp_dir) / f"sample_{idx + 1}_{uploaded_file.name}"
                        if not temp_file_path.suffix:
                            temp_file_path = temp_file_path.with_suffix(ext)
                        temp_file_path.write_bytes(uploaded_file.getvalue())
                        temp_paths.append(temp_file_path)

                    result = recommend(temp_paths, top_k=top_k)

                st.divider()
                st.subheader("2. 분석 요약 및 최적 파라미터 추천 (1순위)")

                best = result["recommended"]
                dom_env = result["dominant_environment"]
                margin = result["recommendation_margin"]

                # 핵심 요약 지표 카드 (4열)
                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                with m_col1:
                    st.metric(
                        label="판정 대표 환경",
                        value=ENV_LABELS.get(dom_env, dom_env).split(" ")[0],
                        delta=dom_env,
                        delta_color="off",
                    )
                with m_col2:
                    st.metric(
                        label="추천 입력 해상도",
                        value=f"{best['image_size']} × {best['image_size']} px",
                    )
                with m_col3:
                    st.metric(
                        label="최적 학습률 (LR 배수)",
                        value=f"{best['lr_multiplier']}x",
                        delta=f"Main: {best['main_lr']} / Backbone: {best['backbone_lr']}",
                        delta_color="off",
                    )
                with m_col4:
                    st.metric(
                        label="예상 mAP@0.5:0.95",
                        value=f"{best['expected_map50_95'] * 100:.2f}%",
                        delta=f"추천 격차: {margin}",
                        delta_color="normal" if margin == "HIGH" else "off",
                    )

                st.info(
                    f"💡 **추천 근거:** 5장 평균 환경 유사도를 반영한 결과, **{best['image_size']}px** 해상도와 "
                    f"**{best['lr_multiplier']}x** 학습률 조합이 가장 높은 예상 성능(**{best['expected_map50_95']*100:.2f}%**)을 보입니다. "
                    f"(1위-2위 성능 격차: **{MARGIN_DESC.get(margin, margin)}**)"
                )

                # --------------------------------------------------------
                # 환경 유사도 분포 및 Top-K 추천 테이블
                # --------------------------------------------------------
                col_left, col_right = st.columns([1, 1.8])

                with col_left:
                    st.subheader("3. 환경별 평균 유사도")
                    sim_data = result["environment_similarity"]
                    sim_df = pd.DataFrame(
                        {
                            "기준 환경": [ENV_LABELS.get(k, k) for k in sim_data.keys()],
                            "유사도 가중치 (%)": [round(v * 100, 2) for v in sim_data.values()],
                        }
                    )
                    st.bar_chart(sim_df.set_index("기준 환경"), color="#2563eb")
                    st.dataframe(sim_df, hide_index=True, use_container_width=True)

                with col_right:
                    st.subheader(f"4. Top-{top_k} 추천 하이퍼파라미터 조합 비교")
                    top_rows = []
                    for rank, item in enumerate(result["top_k"], start=1):
                        top_rows.append(
                            {
                                "순위": f"{rank}위",
                                "LR 배수": f"{item['lr_multiplier']}x",
                                "Main LR": f"{item['main_lr']:.5f}",
                                "Backbone LR": f"{item['backbone_lr']:.7f}",
                                "입력 해상도": f"{item['image_size']} × {item['image_size']}",
                                "예상 mAP50-95": f"{item['expected_map50_95'] * 100:.2f}%",
                            }
                        )
                    st.dataframe(pd.DataFrame(top_rows), hide_index=True, use_container_width=True)

                # --------------------------------------------------------
                # 개별 이미지 특징 추출 상세 결과 표
                # --------------------------------------------------------
                st.subheader("5. 샘플 이미지별 시각적 특징 및 환경 유사도 상세")
                indiv_rows = []
                for idx, item in enumerate(result["individual_results"], start=1):
                    orig_name = uploaded_files[idx - 1].name
                    sims = item["environment_similarity"]
                    indiv_rows.append(
                        {
                            "항목": f"사진 {idx}",
                            "파일명": orig_name,
                            "최근접 환경": item["nearest_environment"],
                            "공간주파수 (SF)": round(item["spatial_frequency"], 4),
                            "밝기 (Brightness)": round(item["brightness"], 4),
                            "채도 (Saturation)": round(item["saturation"], 4),
                            "주간 유사도": f"{sims['DAY_CLEAR_NEW']*100:.1f}%",
                            "야간 유사도": f"{sims['NIGHT']*100:.1f}%",
                            "악천후 유사도": f"{sims['WEATHER']*100:.1f}%",
                        }
                    )
                st.dataframe(pd.DataFrame(indiv_rows), hide_index=True, use_container_width=True)

            except Exception as e:
                st.error(f"분석 중 오류가 발생했습니다: {e}")
