from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
from app.ai_logic import analyzer

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(
    title="Basil Health Analyzer API",
    description="바질 식물 상태 분석 및 엽면적(PLA) 계산 서비스",
    version="1.2.1",
)

# CORS 설정 (프론트엔드 접근 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 배포 시 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    return {
        "status": "healthy",
        "message": "서버가 정상적으로 작동 중입니다.",
    }


@app.post("/analyze")
async def analyze_plant(file: UploadFile = File(...)):
    """
    식물 이미지 분석 엔드포인트

    - **file**: 분석할 식물 이미지 (JPEG, PNG 등)

    Returns:
        - status: 'success' 또는 'error'
        - data: 분석 결과 (status='success'인 경우)
          - diagnosis: 'healthy' 또는 'unhealthy'
          - confidence: 분류 신뢰도 (%)
          - pla_mm2: 엽면적 (mm²)
          - pla_cm2: 엽면적 (cm²)
          - green_pixels: 검출된 초록색 픽셀 수
          - message: 결과 메시지
        - message: 에러 메시지 (status='error'인 경우)
    """
    try:
        # 1. 파일 읽기
        logger.info(f"📥 파일 수신: {file.filename}")
        image_data = await file.read()

        if not image_data:
            raise HTTPException(status_code=400, detail="이미지 데이터가 비어있습니다.")

        # 2. AI 로직 실행
        logger.info("🔄 분석 시작...")
        result = analyzer.process(image_data)

        # 3. 결과 반환
        if result["status"] == "success":
            logger.info("✅ 분석 완료")
            return result
        else:
            logger.warning(f"⚠️ 분석 실패: {result['message']}")
            return result

    except Exception as e:
        logger.error(f"❌ 요청 처리 중 오류: {e}", exc_info=True)
        return {
            "status": "error",
            "message": f"요청 처리 중 오류가 발생했습니다: {str(e)}",
        }


@app.get("/")
async def root():
    """API 정보"""
    return {
        "name": "Basil Health Analyzer API",
        "version": "1.0.0",
        "endpoints": {
            "health": "GET /health",
            "analyze": "POST /analyze",
            "docs": "/docs",
        },
    }


if __name__ == "__main__":
    import uvicorn

    # 로컬 실행 (개발 모드)
    # uvicorn main:app --reload --host 127.0.0.1 --port 8000
    uvicorn.run(app, host="127.0.0.1", port=8000)
