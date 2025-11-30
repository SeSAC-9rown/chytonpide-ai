import cv2
import numpy as np
import io
import logging
import os
import torch
import warnings
from PIL import Image, ImageOps
from ultralytics import YOLO, SAM

# ==========================================
# [설정] 로그 및 경고 설정
# ==========================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# ==========================================
# [설정] 모델 및 파일 경로
# ==========================================
# 1. YOLO 모델 (PLA 계산 및 건강 체크용)
DET_MODEL_PATH = r"runs\detect\det_exp1\weights\best.pt"
CLS_MODEL_PATH = r"runs\classify\test1\weights\best.pt"

# 2. SAM 모델 (가볍고 빠른 Base 모델 사용)
SAM_MODEL_PATH = "sam2.1_b.pt"

# 3. 테스트할 이미지 경로
TEST_IMAGE_PATH = r"C:\Users\sega0\Desktop\chytonpide-ai\predict_image\test6.jpg"

# ★★★ [핵심] 사용자가 직접 입력하는 잎의 좌표 (Original Image 기준) ★★★
# 그림판(Paint) 등에서 마우스를 올려 확인한 [X, Y] 좌표를 입력하세요.
# 예시: 잎이 3개라면 -> [[550, 430], [600, 480], [520, 500]]
MANUAL_LEAF_POINTS = [
    [550, 430], 
    [620, 480],
    [480, 510],
    [530, 390] 
    # ... 잎 개수만큼 추가하세요
]

# 4. 기타 설정
SCALE_REAL_DIAMETER_MM = 16.0
GREEN_HSV_LOWER = [35, 40, 40]
GREEN_HSV_UPPER = [85, 255, 255]


class ManualBasilAnalyzer:
    """사용자가 입력한 좌표를 기반으로 SAM 2가 정밀 분석하는 클래스"""

    def __init__(self):
        logger.info("🤖 AI 모델 로딩 시작...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"👉 사용 장치: {self.device}")

        try:
            # 1. YOLO 모델 로딩
            self.det_model = YOLO(DET_MODEL_PATH)
            self.cls_model = YOLO(CLS_MODEL_PATH)
            
            # 2. SAM 2 모델 로딩
            self.sam_model = SAM(SAM_MODEL_PATH)
            logger.info(f"✅ 모델 로딩 완료 (SAM: {SAM_MODEL_PATH})")

        except Exception as e:
            logger.error(f"❌ 모델 로딩 실패: {e}")
            raise

    def _determine_stage(self, leaf_count):
        """잎 개수에 따른 단계 판별"""
        if leaf_count <= 2:
            return "🌱 떡잎 단계", "떡잎만 존재하거나, 본엽이 나오기 직전입니다."
        elif 3 <= leaf_count <= 4:
            return "🌿 본엽 2매", "본엽이 1쌍(2장) 전개된 상태입니다."
        elif 5 <= leaf_count <= 8:
            return "🌿 본엽 4매 ~ 8매", "본엽이 2쌍에서 4쌍까지 활발히 자라는 중입니다."
        elif 9 <= leaf_count <= 10:
            return "🌿 본엽 8매 ~ 10매", "본엽 성장이 거의 완료되어 가며, 곧 분지가 예상됩니다."
        else:
            return "🌳 분지 발생", "잎이 10매 이상이며, 곁가지(분지)가 발달하는 단계입니다."

    def _analyze_with_manual_points(self, origin_img_pil):
        """
        사용자가 지정한 좌표(MANUAL_LEAF_POINTS)를 SAM에 입력하여 마스크 생성
        """
        try:
            if not MANUAL_LEAF_POINTS:
                logger.warning("⚠️ 입력된 좌표가 없습니다. MANUAL_LEAF_POINTS를 설정해주세요.")
                return None

            logger.info(f"🔍 사용자 좌표 {len(MANUAL_LEAF_POINTS)}개에 대해 SAM 분석 시작...")
            
            collected_masks = []
            
            # 각 점마다 SAM에게 물어봅니다.
            # (한 번에 다 보내면 하나의 객체로 인식할 수 있어, 루프를 돕니다)
            for i, point in enumerate(MANUAL_LEAF_POINTS):
                # points=[[x, y]], labels=[1] (1은 전경/Foreground 의미)
                results = self.sam_model(origin_img_pil, points=[[point]], labels=[1], verbose=False)
                
                if results and results[0].masks:
                    # 마스크 데이터 추출 (가장 높은 신뢰도)
                    mask_data = results[0].masks.data.cpu().numpy()[0] # (H, W)
                    collected_masks.append(mask_data)
                    logger.info(f"   👉 Point {point}: 마스크 생성 성공")
                else:
                    logger.warning(f"   ⚠️ Point {point}: SAM이 객체를 찾지 못했습니다.")

            # 결과 정리
            leaf_count = len(collected_masks)
            stage_name, message = self._determine_stage(leaf_count)
            
            # 시각화용 마스크 합치기
            w, h = origin_img_pil.size
            combined_mask = np.zeros((h, w), dtype=np.uint8)
            
            for mask in collected_masks:
                # 마스크 크기가 원본과 맞는지 확인
                if mask.shape != (h, w):
                     mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                combined_mask = np.maximum(combined_mask, (mask > 0).astype(np.uint8) * 255)

            return {
                "leaf_count": leaf_count,
                "stage": stage_name,
                "message": message,
                "mask": combined_mask
            }

        except Exception as e:
            logger.error(f"❌ SAM 수동 분석 중 오류: {e}")
            return None

    def _calculate_pla(self, basil_crop_bgr, mm_per_pixel):
        """기존 PLA 계산"""
        try:
            basil_hsv = cv2.cvtColor(basil_crop_bgr, cv2.COLOR_BGR2HSV)
            lower = np.array(GREEN_HSV_LOWER, dtype=np.uint8)
            upper = np.array(GREEN_HSV_UPPER, dtype=np.uint8)
            mask = cv2.inRange(basil_hsv, lower, upper)
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            pixels = cv2.countNonZero(mask)
            area = pixels * (mm_per_pixel ** 2)
            return {"pla_mm2": round(area, 2), "green_pixels": pixels}
        except:
            return None

    def process_file(self, image_path):
        if not os.path.exists(image_path):
            logger.error("파일 없음")
            return
        with open(image_path, "rb") as f:
            return self.process(f.read())

    def process(self, image_bytes):
        try:
            # 1. 이미지 로드
            origin_img_pil = Image.open(io.BytesIO(image_bytes))
            origin_img_pil = ImageOps.exif_transpose(origin_img_pil).convert("RGB")
            origin_img_bgr = cv2.cvtColor(np.array(origin_img_pil), cv2.COLOR_RGB2BGR)

            # 2. YOLO 실행 (Crop 이미지와 mm_per_pixel을 얻기 위함)
            # 수동 분석이므로 YOLO가 실패해도 진행할 수는 있지만, PLA를 위해 실행
            results = self.det_model(origin_img_pil, conf=0.15, verbose=False)
            
            mm_per_pixel = 0.1 # 기본값
            basil_crop_pil = None
            basil_crop_bgr = None
            
            if len(results) > 0:
                boxes = results[0].boxes
                cls_ids = boxes.cls.cpu().numpy().astype(int)
                for i, cls_id in enumerate(cls_ids):
                    x1, y1, x2, y2 = map(int, boxes[i].xyxy[0])
                    
                    if cls_id == 1: # Scale Marker
                        d = max(x2 - x1, y2 - y1)
                        mm_per_pixel = SCALE_REAL_DIAMETER_MM / d
                        logger.info(f"📏 Scale: 1px = {mm_per_pixel:.4f}mm")
                    
                    elif cls_id == 0: # Basil
                        basil_crop_bgr = origin_img_bgr[y1:y2, x1:x2]
                        basil_crop_pil = Image.fromarray(cv2.cvtColor(basil_crop_bgr, cv2.COLOR_BGR2RGB))

            # 3. [핵심] 사용자가 찍은 좌표로 SAM 분석 실행
            # (YOLO Crop 이미지가 아니라 '원본 이미지'를 넣습니다)
            growth_info = self._analyze_with_manual_points(origin_img_pil)

            # 4. 기타 분석 (PLA 등) - 바질을 못 찾았으면 원본 전체로 계산 시도
            if basil_crop_bgr is None:
                basil_crop_bgr = origin_img_bgr
                basil_crop_pil = origin_img_pil

            pla_result = self._calculate_pla(basil_crop_bgr, mm_per_pixel)
            
            cls_res = self.cls_model(basil_crop_pil, verbose=False)[0]
            health = cls_res.names[cls_res.probs.top1]
            conf = float(cls_res.probs.top1conf) * 100

            # 5. 결과 시각화
            self._save_visualization(origin_img_bgr, growth_info)

            return {
                "status": "success",
                "data": {
                    "health": {"status": health, "confidence": f"{conf:.2f}%"},
                    "pla": pla_result,
                    "growth": growth_info
                }
            }

        except Exception as e:
            logger.error(f"오류: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    def _save_visualization(self, origin_img, growth_info):
        """결과 저장 (원본 이미지 위에 표시)"""
        try:
            if growth_info and growth_info['mask'] is not None:
                mask = growth_info['mask']
                
                # 초록색 마스크
                color_mask = np.zeros_like(origin_img)
                color_mask[mask > 0] = [0, 255, 0]
                
                # 오버레이
                overlay = cv2.addWeighted(origin_img, 0.7, color_mask, 0.3, 0)
                
                # 사용자가 찍은 점 표시 (빨간 점)
                for pt in MANUAL_LEAF_POINTS:
                    cv2.circle(overlay, (pt[0], pt[1]), 5, (0, 0, 255), -1)

                # 텍스트
                txt = f"{growth_info['stage']} (Count: {growth_info['leaf_count']})"
                cv2.putText(overlay, txt, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                cv2.imwrite("result_manual_sam.jpg", overlay)
                logger.info("💾 결과 저장 완료: result_manual_sam.jpg")
        except Exception as e:
            logger.warning(f"시각화 저장 실패: {e}")

if __name__ == "__main__":
    analyzer = ManualBasilAnalyzer()
    print(f"\n🚀 분석 시작: {TEST_IMAGE_PATH}")
    analyzer.process_file(TEST_IMAGE_PATH)