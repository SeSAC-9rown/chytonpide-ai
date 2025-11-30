# Stage 1: 기본 이미지
FROM python:3.10.19-slim

WORKDIR /app

# 시스템 패키지 설치 (OpenCV 의존성)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python 패키지 설치
COPY requirements.txt .
# CPU 버전 PyTorch 설치 (CUDA 라이브러리 제외로 빌드 시간 단축)
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# 애플리케이션 코드 복사 (weights는 app 폴더 안에 있음)
COPY app/ /app/app/

# 포트 노출
EXPOSE 8000

# 서버 실행 (uvicorn)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]