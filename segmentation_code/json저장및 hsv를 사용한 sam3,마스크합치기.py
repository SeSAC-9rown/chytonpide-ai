import cv2
import numpy as np
import io
import logging
import os
import torch
import warnings
import json
import time
from PIL import Image, ImageOps
from ultralytics import YOLO

# ★ Hugging Face Transformers (SAM 3)
# 실행 전: pip install git+https://github.com/huggingface/transformers.git
try:
    from transformers import Sam3Processor, Sam3Model
except ImportError:
    print("❌ [오류] transformers 라이브러리 업데이트가 필요합니다.")
    print("👉 터미널에 다음을 입력하세요: pip install git+https://github.com/huggingface/transformers.git")
    raise

# ==========================================
# [설정] 로그 및 경고 설정
# ==========================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# ==========================================
# [설정] 모델 및 파일 경로
# ==========================================
# 1. YOLO 모델 (전체 식물 탐지용)
DET_MODEL_PATH = r"runs\detect\det_exp1\weights\best.pt"
CLS_MODEL_PATH = r"runs\classify\test1\weights\best.pt"

# 2. SAM 3 설정 (Transformers)
# 자동으로 facebook/sam3 모델을 다운로드합니다.
SAM3_MODEL_ID = "facebook/sam3"
SAM_TEXT_PROMPT = "leaf"  # ★ 텍스트로 잎 찾기

# 3. 테스트할 이미지 경로
TEST_IMAGE_PATH = r"C:\Users\sega0\Desktop\chytonpide-ai\predict_image\test6.jpg"

# 4. 기타 설정
SCALE_REAL_DIAMETER_MM = 16.0
GREEN_HSV_LOWER = [35, 40, 40]
GREEN_HSV_UPPER = [85, 255, 255]


class Sam3TransformersAnalyzer:
    """
    [SAM 3 Transformers 분석기]
    - YOLO: 바질 위치(Crop) 찾기
    - SAM 3: 'basil leaf' 텍스트 프롬프트로 잎 정밀 분할
    """

    def __init__(self):
        logger.info("🤖 AI 모델 로딩 시작...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"👉 사용 장치: {self.device}")

        try:
            # 1. YOLO 모델 로딩
            self.det_model = YOLO(DET_MODEL_PATH)
            self.cls_model = YOLO(CLS_MODEL_PATH)
            logger.info("✅ YOLO 모델 로딩 완료")

            # 2. SAM 3 모델 로딩 (Transformers)
            logger.info(f"⏳ SAM 3 모델({SAM3_MODEL_ID}) 다운로드 및 로딩 중...")
            self.processor = Sam3Processor.from_pretrained(SAM3_MODEL_ID)
            self.sam_model = Sam3Model.from_pretrained(SAM3_MODEL_ID).to(self.device)
            logger.info(f"✅ SAM 3 모델 로딩 완료")

        except Exception as e:
            logger.error(f"❌ 초기화 실패: {e}")
            raise

    def _determine_stage(self, leaf_count):
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

    def _run_sam3_text(self, basil_crop_pil):
        """
        [SAM 3 텍스트 프롬프트 로직]
        Transformers 라이브러리를 사용하여 텍스트로 세그멘테이션 수행
        """
        try:
            # 이미지 전처리 및 텍스트 프롬프트 입력
            inputs = self.processor(
                images=basil_crop_pil, 
                text=SAM_TEXT_PROMPT, 
                return_tensors="pt"
            ).to(self.device)

            # 추론 실행
            with torch.no_grad():
                outputs = self.sam_model(**inputs)

            # 후처리 (Instance Segmentation)
            # threshold: 확신도, mask_threshold: 마스크 이진화 임계값
            results = self.processor.post_process_instance_segmentation(
                outputs,
                threshold=0.4,       # 감도 조절 (낮을수록 많이 찾음)
                mask_threshold=0.5,
                target_sizes=inputs.get("original_sizes").tolist()
            )[0]

            # 결과 추출
            masks = results['masks']  # (N, H, W) Tensor
            scores = results['scores'] # (N,) Tensor
            
            # 텐서를 넘파이로 변환
            masks_np = masks.cpu().numpy().astype(np.uint8)
            leaf_count = len(masks_np)
            
            logger.info(f"🔍 SAM 3가 '{SAM_TEXT_PROMPT}'로 {leaf_count}개의 잎을 찾았습니다.")

            # 시각화를 위해 마스크 합치기
            w, h = basil_crop_pil.size
            combined_mask = np.zeros((h, w), dtype=np.uint8)

            for mask in masks_np:
                # 각 마스크(0/1)를 255(흰색)로 변환하여 합침
                combined_mask = np.maximum(combined_mask, mask * 255)

            stage_name, message = self._determine_stage(leaf_count)

            return {
                "leaf_count": leaf_count,
                "stage": stage_name,
                "message": message,
                "mask": combined_mask,
                "raw_detected": leaf_count
            }

        except Exception as e:
            logger.error(f"❌ SAM 3 분석 오류: {e}")
            import traceback
            traceback.print_exc()
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
            return
        with open(image_path, "rb") as f:
            return self.process(f.read())

    def process(self, image_bytes):
        total_start_time = time.time()
        
        try:
            # 1. 이미지 로드
            origin_img_pil = Image.open(io.BytesIO(image_bytes))
            origin_img_pil = ImageOps.exif_transpose(origin_img_pil).convert("RGB")
            origin_img_bgr = cv2.cvtColor(np.array(origin_img_pil), cv2.COLOR_RGB2BGR)

            # 2. YOLO 탐지 (Crop & Scale)
            yolo_start = time.time()
            results = self.det_model(origin_img_pil, conf=0.15, verbose=False)
            
            mm_per_pixel = 0.1
            basil_crop_pil = None
            basil_crop_bgr = None
            
            if len(results) > 0:
                boxes = results[0].boxes
                cls_ids = boxes.cls.cpu().numpy().astype(int)
                for i, cls_id in enumerate(cls_ids):
                    x1, y1, x2, y2 = map(int, boxes[i].xyxy[0])
                    if cls_id == 1: # Scale
                        d = max(x2 - x1, y2 - y1)
                        mm_per_pixel = SCALE_REAL_DIAMETER_MM / d
                    elif cls_id == 0: # Basil
                        basil_crop_bgr = origin_img_bgr[y1:y2, x1:x2]
                        basil_crop_pil = Image.fromarray(cv2.cvtColor(basil_crop_bgr, cv2.COLOR_BGR2RGB))
                        logger.info(f"🌿 바질 발견! 크기: {basil_crop_pil.size}")

            if basil_crop_pil is None:
                return {"status": "error", "message": "바질을 찾을 수 없습니다."}

            # 3. SAM 3 텍스트 프롬프트 분석
            growth_info = self._run_sam3_text(basil_crop_pil)

            # 4. PLA 및 건강 분석
            pla_result = self._calculate_pla(basil_crop_bgr, mm_per_pixel)
            
            cls_res = self.cls_model(basil_crop_pil, verbose=False)[0]
            health = cls_res.names[cls_res.probs.top1]
            conf = float(cls_res.probs.top1conf) * 100

            # 5. 시각화 저장
            self._save_visualization(basil_crop_bgr, growth_info)

            # JSON용 데이터
            growth_data_json = growth_info.copy() if growth_info else None
            if growth_data_json and 'mask' in growth_data_json:
                del growth_data_json['mask']

            total_duration = time.time() - total_start_time
            logger.info(f"⚡ 실행 완료: {total_duration:.2f}초")

            return {
                "status": "success",
                "data": {
                    "health": {"status": health, "confidence": f"{conf:.2f}%"},
                    "pla": pla_result,
                    "growth": growth_data_json,
                    "execution_time": round(total_duration, 2)
                }
            }

        except Exception as e:
            logger.error(f"오류: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    def _save_visualization(self, crop_img, growth_info):
        try:
            if growth_info and growth_info.get('mask') is not None:
                mask = growth_info['mask']
                if mask.shape[:2] != crop_img.shape[:2]:
                    mask = cv2.resize(mask, (crop_img.shape[1], crop_img.shape[0]), interpolation=cv2.INTER_NEAREST)
                
                color_mask = np.zeros_like(crop_img)
                color_mask[mask > 0] = [0, 255, 0]
                overlay = cv2.addWeighted(crop_img, 0.7, color_mask, 0.3, 0)
                
                txt = f"{growth_info['stage']} (Leaves: {growth_info['leaf_count']})"
                cv2.putText(overlay, txt, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                cv2.imwrite("result_sam3_transformers.jpg", overlay)
                logger.info("💾 결과 저장: result_sam3_transformers.jpg")
        except:
            pass

if __name__ == "__main__":
    analyzer = Sam3TransformersAnalyzer()
    print(f"\n🚀 SAM 3 (Transformers) 분석 시작: {TEST_IMAGE_PATH}")
    
    final_result = analyzer.process_file(TEST_IMAGE_PATH)
    
    if final_result:
        json_filename = "result.json"
        with open(json_filename, "w", encoding="utf-8") as f:
            json.dump(final_result, f, ensure_ascii=False, indent=4)
        print("\n📊 [분석 결과]")
        print(json.dumps(final_result, ensure_ascii=False, indent=4))
    else:
        print("❌ 분석 실패") 