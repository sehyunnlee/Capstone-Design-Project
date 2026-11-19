import tempfile
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from recommender import recommend

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="차량 객체 검지 최적 하이퍼파라미터 추천 시스템")

# 정적 파일(HTML/CSS/JS) 경로 마운트
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def read_index():
    """메인 웹 페이지 반환"""
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/recommend")
async def get_recommendation(
    files: List[UploadFile] = File(...),
    top_k: int = 3,
):
    """샘플 이미지 1~5장을 받아 최적 하이퍼파라미터 추천 결과 반환"""
    if not (1 <= len(files) <= 5):
        raise HTTPException(
            status_code=400,
            detail=f"1장 이상 5장 이하의 이미지만 업로드할 수 있습니다. (현재 업로드: {len(files)}장)",
        )

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_paths = []
            original_names = []

            for idx, uploaded_file in enumerate(files):
                orig_name = uploaded_file.filename or f"image_{idx + 1}.jpg"
                ext = Path(orig_name).suffix.lower()
                if ext not in {".jpg", ".jpeg", ".png"}:
                    raise HTTPException(
                        status_code=400,
                        detail=f"지원하지 않는 파일 형식입니다: {orig_name} (jpg, jpeg, png만 지원)",
                    )

                temp_file_path = Path(temp_dir) / f"sample_{idx + 1}{ext}"
                content = await uploaded_file.read()
                temp_file_path.write_bytes(content)

                temp_paths.append(temp_file_path)
                original_names.append(orig_name)

            # recommender.py 핵심 엔진 호출
            result = recommend(temp_paths, top_k=top_k)

            # 임시 파일명 대신 원본 파일명 매핑
            for idx, item in enumerate(result["individual_results"]):
                item["image"] = original_names[idx]

            return result

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"분석 중 서버 오류 발생: {str(e)}")


import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
