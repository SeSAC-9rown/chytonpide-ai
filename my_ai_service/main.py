import uvicorn
import os
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.ai_logic import analyzer  # 위에서 만든 로직 import

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# FastAPI 앱 초기화
app = FastAPI(
    title="Basil AI Server (Hybrid)",
    description="Local YOLO + Azure Remote SAM Architecture",
    version="1.2.1"
)

# CORS 설정 (모든 도메인 허용 - 개발용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Basil AI Server is Running 🚀"}

@app.get("/health")
def health_check():
    """서버 및 모델 상태 확인"""
    # 간단한 상태 체크 로직
    if analyzer.det_model is None:
         raise HTTPException(status_code=503, detail="AI Model not loaded")
    return {"status": "healthy", "azure_connected": bool(os.getenv("AZURE_API_KEY"))}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """이미지 업로드 및 분석 요청"""
    try:
        # 파일 유효성 검사
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다.")

        logger.info(f"📥 요청 수신: {file.filename}")
        
        # 파일 읽기
        image_bytes = await file.read()
        
        # AI 분석 실행
        result = analyzer.process(image_bytes)
        
        if result["status"] == "error":
            logger.warning(f"분석 실패: {result['message']}")
            # 비즈니스 로직상 200 OK를 주되 에러 메시지를 담을지, 
            # 500 에러를 줄지는 선택사항입니다. 여기선 200 반환.
            return result
            
        logger.info("✅ 분석 성공 및 응답 전송")
        return result

    except Exception as e:
        logger.error(f"❌ 서버 내부 에러: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # 포트 설정: Azure가 지정한 포트(WEBSITES_PORT) 혹은 8000
    port = int(os.environ.get("WEBSITES_PORT", 8000))
    
    logger.info(f"🚀 서버 시작 (Port: {port})")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)