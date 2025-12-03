# 🌱 Chytonpide AI - Basil Health Analyzer API v1.2.0

**바질 식물의 건강 상태를 분석하고 엽면적(PLA)을 계산하는 AI 서비스**

**핵심 기술**: YOLO11 (객체 탐지 + 분류) + **FastSAM** (세그멘테이션)

---

## 📁 프로젝트 구조

```
chytonpide-ai/
│
├── 🚀 my_ai_service/                      # ⭐ FastAPI AI 서비스 (메인)
│   ├── app/
│   │   ├── main.py                        # FastAPI v1.0.0
│   │   │   ├── GET /                      # API 정보
│   │   │   ├── GET /health                # 헬스 체크
│   │   │   └── POST /analyze              # 식물 분석
│   │   ├── ai_logic.py                    # ⭐ BasilAnalyzer 클래스 (핵심)
│   │   │   ├── __init__()                 # YOLO11 + FastSAM 모델 로딩
│   │   │   ├── _separate_overlapping_leaves()  # Watershed 알고리즘
│   │   │   ├── _count_leaves()            # FastSAM + Watershed로 잎 개수
│   │   │   ├── _calculate_pla()           # 엽면적(PLA) 계산
│   │   │   └── process()                  # 전체 처리 파이프라인
│   │   └── config.py                      # 설정 상수
│   │
│   ├── weights/
│   │   ├── det_best.pt                    # YOLO11 탐지 (Scale + Basil)
│   │   ├── cls_best.pt                    # YOLO 분류 (Healthy/Unhealthy)
│   │   └── FastSAM-s.pt                   # FastSAM 모델
│   │
│   ├── .dockerignore
│   ├── Dockerfile                         # Docker 이미지
│   ├── requirements.txt                   # 의존성: FastAPI, OpenCV, Ultralytics FastSAM 등
│   ├── ARCHITECTURE.md
│   ├── SYSTEM_OVERVIEW.md
│   └── README.md
│
├── 📚 train_cls.py, train_det.py          # YOLO 학습 스크립트
├── predict_cls.py, predict_det.py         # 예측 스크립트
├── calculate_pla.py                       # PLA 독립 계산
│
├── ⚙️ configs/                            # YOLO 설정
├── 📊 runs/                               # 학습 결과
├── 🔬 segmentation_code/                  # FastSAM 실험 코드
└── CHANGELOG.md
```

---

## 🔄 AI 처리 흐름 (ai_logic.py)

### BasilAnalyzer.process() 전체 파이프라인

<img width="352" height="1217" alt="image" src="https://github.com/user-attachments/assets/6338eb51-0577-4dc8-878a-84400661d715" />

---

## 📊 각 단계 상세

### 1️⃣ YOLO11 탐지 (det_best.pt)
- **목표**: Scale 마커(16mm) + 바질 식물 검출
- **입력**: 원본 이미지
- **출력**: 2개 클래스 (ID 0=Basil, ID 1=Scale)
- **신뢰도**: 0.15 (낮춤, 작은 스티커도 감지)
- **소요시간**: ~800ms

### 2️⃣ FastSAM 세그멘테이션 (FastSAM-x.pt)
```python
# ai_logic.py 71-88번 줄
results = self.sam_model(basil_crop_bgr)
masks = results[0].masks.data.cpu().numpy()  # 여러 마스크 반환
```
- **목표**: 바질 이미지를 여러 영역으로 분할
- **입력**: 바질 크롭 (YOLO 탐지 결과)
- **출력**: 여러 마스크 (각각이 하나의 객체)
- **소요시간**: ~5000ms

### 3️⃣ Watershed로 겹친 잎 분리 (_count_leaves)
```python
# FastSAM 마스크 → Watershed로 분리
markers = self._separate_overlapping_leaves(mask_uint8)
```
1. 각 FastSAM 마스크에 대해:
   - HSV 초록색 필터링 (H: 35-85)
   - 초록색 비율 50% 이상 확인

2. Watershed 알고리즘 적용:
   - Distance Transform
   - 확실한 전경/배경 구분
   - Watershed 수행

3. 분리된 각 영역 검증:
   - 크기 100px 이상
   - 초록색 비율 40% 이상
   - 잎으로 카운트

**결과**: 잎 개수, 각 잎의 면적
- **소요시간**: ~1000ms
### 4️⃣ PLA 계산 (_calculate_pla)
```python
# 초록색 픽셀 기반 면적 계산
green_mask = cv2.inRange(hsv, lower_green, upper_green)
green_pixel_count = cv2.countNonZero(green_mask)
area_mm2 = green_pixel_count * (mm_per_pixel ** 2)
```
- **입력**: 바질 크롭 이미지, mm_per_pixel
- **처리**: HSV 필터 + 모폴로지 연산 (노이즈 제거)
- **출력**: pla_mm2, pla_cm2, green_pixels
- **소요시간**: ~100ms

###  YOLO11 분류 (cls_best.pt)
- **목표**: 바질의 건강여부파악
- **입력**: 512사이즈 이미지
- **출력**: 2개 클래스 (Healthy,Unhealthy)
- **신뢰도**: 0.95
- **소요시간**: ~400ms
---

## ⏱️ 성능

| 단계 | 소요시간 |
|------|---------|
| 이미지 전처리 | ~100ms |
| YOLO11 탐지 | ~5000ms |
| FastSAM 세그멘테이션 | ~500ms |
| Watershed + 잎 개수 | ~200ms |
| PLA 계산 | ~100ms |
| YOLO 분류 | ~400ms |
| **총합** | **~6.2초** |

> 📌 첫 요청: ~3000ms (모델 로딩)

최대 9.2초 소요됨(local)

(실제 네트워크 환경에 따라 차 발생함)

B2 cpu 환경에서 사용할시 더 늦어질수 있음 최소 15초 소요
---

## 🚀 설치 & 실행

### 로컬 개발
```bash
cd my_ai_service
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Docker
```bash
docker build -t basil-analyzer:v1.2.0 .
docker run -p 8000:8000 basil-analyzer:v1.2.0
```

---

## 📡 API

### POST /analyze
```bash
curl -X POST "http://localhost:8000/analyze" \
  -F "file=@plant_image.jpg"
```

**응답:**
```json
{
  "status": "success",
  "data": {
    "diagnosis": "healthy",
    "confidence": "95.50%",
    "pla_mm2": 2500.45,
    "pla_cm2": 25.00,
    "leaf_count": 12
  }
}
```

**응답 필드 설명:**
- `diagnosis`: 식물 상태 (healthy/unhealthy)
- `confidence`: 분류 신뢰도 (%)
- `pla_mm2`: 엽면적 (제곱밀리미터)
- `pla_cm2`: 엽면적 (제곱센티미터)
- `leaf_count`: 잎 개수
- `growth_stage`: 성장 단계 (Sprout/Middle/Adult) -> backend 서버에서 잎개수를 바탕으로 계산

---

**최종 수정**: 2025-12-2
