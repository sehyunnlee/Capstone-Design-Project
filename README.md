# 🚗 차량 객체 검지기(RT-DETRv4-L) 하이퍼파라미터 추천 시스템

배포 환경의 도로 CCTV 샘플 이미지(1~5장)를 **멀티모달 비전 AI(Gemini Vision API)와 시각 통계 하이브리드 엔진**으로 진단하여, 해당 도로 환경에 최적화된 차량 객체 검지기(`RT-DETRv4-L`)의 **입력 해상도(`Image Size`)**와 **학습률 배수(`LR Multiplier`)**를 자동으로 추천해 주는 웹 기반 의사결정 지원 시스템입니다.

---

## ✨ 주요 기능

1. **하이브리드 도로 환경 진단 엔진 (AI + Statistical Fallback)**
   - **1순위 — Gemini Vision AI 단일 배치 멀티모달 진단**: 입력된 1~5장의 샘플 이미지를 1회의 배치 API 호출로 동시 분석하여 주간·맑음(`DAY_CLEAR_NEW`), 야간(`NIGHT`), 악천후(`WEATHER` — 빗길 반사, 폭설, 안개 등)에 대한 환경별 확률 분포와 **이미지별 AI 판정 근거**를 도출합니다.
   - **2순위 — 시각 통계 자동 전환(Fallback) 엔진**: API 키가 미설정되었거나 네트워크/서버 일시 혼잡 시, 공간주파수(Spatial Frequency), 평균 밝기(Brightness), 채도(Saturation)의 정규화 유클리드 거리 기반 통계 모델로 자동 전환되어 중단 없이 결과를 제공합니다.
2. **가중 기대 성능(`Expected mAP50-95`) 기반 하이퍼파라미터 탐색**
   - 산출된 환경별 가중치와 사전 구축된 성능 그리드(`4가지 LR × 4가지 해상도 = 16개 조합`)를 결합해 최적의 `Image Size`(640~1024px) 및 `LR Multiplier`(0.5x~2.0x) 순위를 산출합니다.
3. **직관적인 웹 대시보드 제공**
   - 분석에 사용된 엔진 배지(`GEMINI VISION AI` / `STATISTICAL FALLBACK`), Top-5 추천 후보 비교표, 개별 샘플 이미지의 AI 환경 판정 근거 및 물리 통계 수치를 한눈에 확인할 수 있습니다.

---

## 📁 프로젝트 구조

```text
├── app.py                     # FastAPI 웹 서버 및 추천 API 엔드포인트 (/api/recommend)
├── recommender.py             # Gemini Vision 배치 분석 + 시각 통계 Fallback 핵심 추천 엔진
├── requirements.txt           # Python 의존성 패키지 목록
├── .env.example               # 환경 변수(GEMINI_API_KEY) 설정 템플릿
├── .gitignore                 # 보안 키(.env) 및 임시 파일 제외 설정
├── algorithm_data/
│   ├── final_lr_image_grid.csv                      # 환경·LR·해상도별 기준 성능(mAP50-95) 데이터
│   └── sf_brightness_saturation_prototypes.json     # 환경별 시각 특징 프로토타입 기준값
└── static/
    └── index.html             # 샘플 이미지 업로드 및 AI 진단 결과 대시보드 UI
```

---

## 🚀 로컬 실행 방법

### 1. 저장소 클론 및 가상환경 설정
```bash
git clone <저장소_URL>
cd <프로젝트_폴더명>

python -m venv venv
# Windows (PowerShell)
.\venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 2. 필수 패키지 설치
```bash
pip install -r requirements.txt
```

### 3. API 키 설정 (선택 사항)
- **AI 정밀 진단 모드 사용 시**: `.env.example` 파일을 복사하여 `.env` 파일을 생성한 뒤 발급받은 Gemini API 키를 입력합니다.
  ```env
  GEMINI_API_KEY=your_gemini_api_key_here
  ```
- **무설정(Offline) 통계 모드 사용 시**: `.env` 파일을 만들지 않고 바로 실행해도 시스템이 자동으로 **시각 통계(SF·밝기·채도) Fallback 엔진**으로 동작합니다.

### 4. 웹 서버 실행 및 접속
```bash
python app.py
```
서버 실행 후 브라우저에서 **`http://127.0.0.1:8000`**으로 접속합니다.

---

## 🛠️ 기술 스택

- **Backend**: Python, FastAPI, Uvicorn, Pydantic
- **AI & Vision**: Google GenAI SDK (`google-genai`), OpenCV (`opencv-python-headless`), NumPy, Pandas
- **Frontend**: HTML5, CSS3, Vanilla JavaScript (Fetch API)