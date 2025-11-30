import cv2
import numpy as np
import io
import logging
import os
import torch
import warnings
from PIL import Image, ImageOps
from ultralytics import YOLO
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

# ==========================================
# [설정] 로그 및 경고 설정
# ==========================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# ==========================================
# [설정] 모델 및 파일 경로 (★★ 수정 필요 ★★)
# ==========================================
# 1. YOLO 모델 경로
DET_MODEL_PATH = r"runs\detect\det_exp1\weights\best.pt"        # 탐지 모델 (없으면 자동 다운로드됨, 실제 경로 입력 권장)
CLS_MODEL_PATH = r"runs\classify\test1\weights\best.pt"    # 분류 모델 (실제 경로 입력 권장)

# 2. Segformer 모델 경로 및 가중치
SEG_MODEL_NAME = "nvidia/mit-b0"
SEG_WEIGHT_PATH = r"runs\seg\PA\PA_MIOU.pth"  # ★ 사용자의 학습된 가중치 경로

# 3. 테스트할 이미지 경로
TEST_IMAGE_PATH = r"C:\Users\sega0\Desktop\chytonpide-ai\predict_image\test4.jpg"

# 4. 기타 설정
SCALE_REAL_DIAMETER_MM = 16.0  # 스티커 실제 지름
GREEN_HSV_LOWER = [35, 40, 40]
GREEN_HSV_UPPER = [85, 255, 255]
NUM_SEG_CLASSES = 2  # 0: 배경, 1: 잎


class IntegratedBasilAnalyzer:
    """YOLO와 Segformer가 통합된 바질 분석기"""

    def __init__(self):
        logger.info("🤖 AI 모델 로딩 시작...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"👉 사용 장치: {self.device}")

        try:
            # 1. YOLO 모델 로딩
            self.det_model = YOLO(DET_MODEL_PATH)
            self.cls_model = YOLO(CLS_MODEL_PATH)
            logger.info("✅ YOLO 모델 로딩 완료")

            # 2. Segformer 모델 로딩
            self.seg_processor = SegformerImageProcessor.from_pretrained(SEG_MODEL_NAME)
            
            id2label = {0: "background", 1: "leaf"}
            label2id = {"background": 0, "leaf": 1}
            
            self.seg_model = SegformerForSemanticSegmentation.from_pretrained(
                SEG_MODEL_NAME,
                num_labels=NUM_SEG_CLASSES,
                id2label=id2label,
                label2id=label2id,
                ignore_mismatched_sizes=True
            )

            # 가중치 로드
            if os.path.exists(SEG_WEIGHT_PATH):
                checkpoint = torch.load(SEG_WEIGHT_PATH, map_location=self.device)
                state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
                self.seg_model.load_state_dict(state_dict, strict=False)
                self.seg_model.to(self.device)
                self.seg_model.eval()
                logger.info("✅ Segformer 모델 및 가중치 로딩 완료")
            else:
                logger.warning(f"⚠️ Segformer 가중치 파일 없음: {SEG_WEIGHT_PATH}. 기본 가중치로 실행됩니다.")
                self.seg_model.to(self.device)

        except Exception as e:
            logger.error(f"❌ 모델 로딩 실패: {e}")
            raise

    def _determine_stage(self, leaf_count):
        """잎 개수에 따른 생육 단계 판별"""
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

    def _analyze_growth_stage(self, pil_image):
        """Segformer를 이용한 잎 개수 및 성장 단계 분석"""
        try:
            # 리사이징 (모델 입력 크기에 맞춤)
            target_size = (640, 640)
            img_resized = pil_image.resize(target_size)
            
            inputs = self.seg_processor(images=img_resized, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.seg_model(**inputs)
            
            logits = outputs.logits
            # 원래 크기로 복원하지 않고 리사이즈된 상태에서 마스크 생성 (연산 효율성)
            upsampled_logits = torch.nn.functional.interpolate(
                logits,
                size=target_size[::-1],
                mode="bilinear",
                align_corners=False,
            )
            
            # 마스크 추출 (0: 배경, 1: 잎)
            pred_mask = upsampled_logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
            
            # 잎(1) 부분만 추출하여 개수 세기
            leaf_mask = (pred_mask == 1).astype(np.uint8) * 255
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(leaf_mask, connectivity=8)
            leaf_count = num_labels - 1  # 배경 제외

            stage_name, message = self._determine_stage(leaf_count)
            
            return {
                "leaf_count": leaf_count,
                "stage": stage_name,
                "message": message,
                "mask": leaf_mask  # 시각화를 위해 마스크 반환
            }

        except Exception as e:
            logger.error(f"❌ 성장 단계 분석 중 오류: {e}")
            return None

    def _calculate_pla(self, basil_crop_bgr, mm_per_pixel):
        """기존 PLA(엽면적) 계산 로직"""
        try:
            basil_hsv = cv2.cvtColor(basil_crop_bgr, cv2.COLOR_BGR2HSV)
            lower_green = np.array(GREEN_HSV_LOWER, dtype=np.uint8)
            upper_green = np.array(GREEN_HSV_UPPER, dtype=np.uint8)
            green_mask = cv2.inRange(basil_hsv, lower_green, upper_green)
            
            kernel = np.ones((3, 3), np.uint8)
            green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
            green_pixel_count = cv2.countNonZero(green_mask)

            area_mm2 = green_pixel_count * (mm_per_pixel ** 2)
            return {
                "pla_mm2": round(area_mm2, 2),
                "green_pixels": int(green_pixel_count),
            }
        except Exception as e:
            logger.error(f"❌ PLA 계산 오류: {e}")
            return None

    def process_file(self, image_path):
        """로컬 파일 처리용 함수"""
        if not os.path.exists(image_path):
            logger.error(f"파일을 찾을 수 없습니다: {image_path}")
            return
        
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        
        return self.process(image_bytes)

    def process(self, image_bytes):
        """전체 분석 프로세스 실행"""
        try:
            # 1. 이미지 준비
            origin_img_pil = Image.open(io.BytesIO(image_bytes))
            origin_img_pil = ImageOps.exif_transpose(origin_img_pil).convert("RGB")
            origin_img_bgr = cv2.cvtColor(np.array(origin_img_pil), cv2.COLOR_RGB2BGR)

            # 2. YOLO 탐지 (Crop & Scale)
            results = self.det_model(origin_img_pil, conf=0.15, verbose=False)
            
            mm_per_pixel = 0.0
            basil_crop_pil = None
            basil_crop_bgr = None
            
            # --- 결과 파싱 ---
            found_ids = results[0].boxes.cls.cpu().numpy().astype(int) if len(results) > 0 else []
            boxes = results[0].boxes

            # A. Scale 마커 찾기 (ID: 1)
            for i, cls_id in enumerate(found_ids):
                if cls_id == 1:
                    box = boxes[i]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    diameter = max(x2 - x1, y2 - y1)
                    mm_per_pixel = SCALE_REAL_DIAMETER_MM / diameter
                    logger.info(f"📏 Scale 감지됨: 1px = {mm_per_pixel:.4f}mm")
                    break
            
            # B. 바질 찾기 (ID: 0)
            for i, cls_id in enumerate(found_ids):
                if cls_id == 0:
                    box = boxes[i]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    basil_crop_bgr = origin_img_bgr[y1:y2, x1:x2]
                    basil_crop_pil = Image.fromarray(cv2.cvtColor(basil_crop_bgr, cv2.COLOR_BGR2RGB))
                    logger.info(f"🌿 바질 감지됨: 크기 {basil_crop_pil.size}")
                    break

            if basil_crop_pil is None:
                return {"status": "error", "message": "바질을 찾을 수 없습니다."}
            
            if mm_per_pixel == 0:
                logger.warning("⚠️ 스케일 마커를 찾지 못했습니다. 임의 값(1.0)을 사용합니다.")
                mm_per_pixel = 0.1 # 임의 값

            # 3. PLA 계산
            pla_result = self._calculate_pla(basil_crop_bgr, mm_per_pixel)

            # 4. 건강 상태 분류 (YOLO-CLS)
            cls_results = self.cls_model(basil_crop_pil, verbose=False)[0]
            health_status = cls_results.names[cls_results.probs.top1]
            health_conf = float(cls_results.probs.top1conf) * 100

            # 5. [NEW] 성장 단계 분석 (Segformer)
            # 바질 부분만 잘린 이미지(basil_crop_pil)를 넣어서 분석 정확도 향상
            logger.info("🔍 성장 단계 정밀 분석 중 (Segformer)...")
            growth_info = self._analyze_growth_stage(basil_crop_pil)

            # 6. 결과 종합 및 시각화 저장
            result_data = {
                "health": {"status": health_status, "confidence": f"{health_conf:.2f}%"},
                "pla": pla_result,
                "growth": growth_info
            }
            
            self._save_visualization(origin_img_bgr, basil_crop_bgr, growth_info)

            return {"status": "success", "data": result_data}

        except Exception as e:
            logger.error(f"처리 중 치명적 오류: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    def _save_visualization(self, original_img, crop_img, growth_info):
        """결과 이미지 저장 (로컬 실험용)"""
        try:
            save_path = "result_combined.jpg"
            
            # 마스크 시각화 (Segformer 결과가 있다면)
            if growth_info and growth_info['mask'] is not None:
                mask = growth_info['mask']
                # 마스크를 crop 이미지 크기로 리사이징
                mask_resized = cv2.resize(mask, (crop_img.shape[1], crop_img.shape[0]), interpolation=cv2.INTER_NEAREST)
                
                # 초록색 오버레이
                color_mask = np.zeros_like(crop_img)
                color_mask[mask_resized == 255] = [0, 255, 0]
                overlay_crop = cv2.addWeighted(crop_img, 0.7, color_mask, 0.3, 0)
                
                # 텍스트 추가
                txt = f"{growth_info['stage']} (Leaf: {growth_info['leaf_count']})"
                cv2.putText(overlay_crop, txt, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                cv2.imwrite("result_crop_seg.jpg", overlay_crop)
                logger.info(f"💾 분할 결과 저장됨: result_crop_seg.jpg")

            logger.info("✅ 분석 완료")
        except Exception as e:
            logger.warning(f"이미지 저장 중 오류: {e}")

# ==========================================
# 실행 부 (Main)
# ==========================================
if __name__ == "__main__":
    # 인스턴스 생성
    analyzer = IntegratedBasilAnalyzer()
    
    # 로컬 파일로 테스트
    print("\n" + "="*50)
    print(f"🚀 분석 시작: {TEST_IMAGE_PATH}")
    print("="*50)
    
    result = analyzer.process_file(TEST_IMAGE_PATH)
    
    import json
    if result and result['status'] == 'success':
        data = result['data']
        print("\n📊 [최종 분석 결과]")
        print(f"1. 건강 상태 : {data['health']['status']} ({data['health']['confidence']})")
        print(f"2. 엽면적(PLA): {data['pla']['pla_mm2']} mm²")
        
        if data['growth']:
            print(f"3. 잎 개수   : {data['growth']['leaf_count']} 장")
            print(f"4. 성장 단계 : {data['growth']['stage']}")
            print(f"5. 상세 코멘트: {data['growth']['message']}")
        else:
            print("3. 성장 단계 : 분석 실패")
    else:
        print("❌ 분석 실패:", result)