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
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# ==========================================
# [설정] 모델 및 파일 경로
# ==========================================
# 1. YOLO 모델 경로
DET_MODEL_PATH = r"C:\Users\sega0\Desktop\grwon\git\chytonpide-ai\runs\detect\det_exp1\weights\best.pt"
CLS_MODEL_PATH = r"C:\Users\sega0\Desktop\grwon\git\chytonpide-ai\runs\classify\test1\weights\best.pt"

# 2. SAM 모델 경로 (성능 향상을 위해 Large 모델 사용 권장)
# sam2.1_l.pt (Large)가 Base보다 겹친 잎 분리에 훨씬 강력합니다.
SAM_MODEL_PATH = "sam2.1_t.pt" 

# 3. 테스트할 이미지 경로
TEST_IMAGE_PATH = r"C:\Users\sega0\Desktop\grwon\git\chytonpide-ai\segmentation_code\results\1_original_crop.jpg"

# 4. 기타 설정 
SCALE_REAL_DIAMETER_MM = 16.0
GREEN_HSV_LOWER = [35, 40, 40]
GREEN_HSV_UPPER = [85, 255, 255]


class IntegratedBasilAnalyzer:
    """YOLO와 SAM 2가 통합된 바질 분석기"""

    def __init__(self):
        logger.info("🤖 AI 모델 로딩 시작...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"👉 사용 장치: {self.device}")

        try:
            self.det_model = YOLO(DET_MODEL_PATH)
            self.cls_model = YOLO(CLS_MODEL_PATH)
            logger.info("✅ YOLO 모델 로딩 완료")

            # SAM 2 모델 로딩
            self.sam_model = SAM(SAM_MODEL_PATH)
            logger.info(f"✅ SAM 2 모델 로딩 완료 ({SAM_MODEL_PATH})")

        except Exception as e:
            logger.error(f"❌ 모델 로딩 실패: {e}")
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

    def _calculate_iou(self, mask1, mask2):
        """두 마스크 간의 IoU(교집합/합집합) 계산"""
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        if union == 0: return 0
        return intersection / union

    def _analyze_growth_with_sam(self, pil_image):
        """
        [개선된 로직]
        1. 이미지 업스케일링 (작은 잎 감지력 향상)
        2. 그리드 포인트 프롬프트 (점 찍어서 찾기)
        3. IoU 기반 중복 제거 (NMS)
        """
        try:
            # 1. 이미지 업스케일링 (1024px로 리사이징하여 인식률 높임)
            w, h = pil_image.size
            scale_factor = 1024 / max(w, h)
            new_w, new_h = int(w * scale_factor), int(h * scale_factor)
            img_resized = pil_image.resize((new_w, new_h), Image.BILINEAR)
            
            # 2. 그리드 포인트 생성 (6x6 = 36개의 점을 찍어 물어봄)
            # 잎이 겹쳐있을 때, 각 위치마다 "여기에 뭐가 있어?"라고 물어보는 방식
            n_points = 6  # 4 → 6으로 증가 (16개 → 36개)
            x = np.linspace(new_w * 0.15, new_w * 0.85, n_points)  # 범위도 넓힘
            y = np.linspace(new_h * 0.15, new_h * 0.85, n_points)
            xv, yv = np.meshgrid(x, y)
            points = np.column_stack((xv.ravel(), yv.ravel()))

            collected_masks = []
            
            # 각 포인트에 대해 SAM 추론 실행
            # (한 번에 배치로 넣을 수도 있지만, Ultralytics wrapper 특성상 루프가 안정적일 수 있음)
            logger.info(f"🔍 그리드 탐색 시작 ({len(points)}개 포인트)...")
            logger.info(f"📐 리사이징된 이미지 크기: {new_w}x{new_h}")

            for idx, pt in enumerate(points):
                # 점 하나를 프롬프트로 전달 (labels=[1]은 전경/Foreground 의미)
                # SAM에게 "이 점(pt)에 해당하는 객체를 따줘"라고 요청
                try:
                    logger.debug(f"   포인트 {idx}: {pt}")
                    results = self.sam_model(img_resized, points=[[pt]], labels=[1], verbose=False)

                    if results and results[0].masks:
                        # 마스크 추출 (가장 신뢰도 높은 것 하나)
                        mask_tensor = results[0].masks.data.cpu().numpy()
                        logger.debug(f"   ✓ 마스크 발견 - 차원: {mask_tensor.shape}")

                        # 마스크 차원 확인 및 처리
                        if mask_tensor.ndim == 3:  # (N, H, W) 형태
                            mask_data = mask_tensor[0]
                        elif mask_tensor.ndim == 2:  # (H, W) 형태
                            mask_data = mask_tensor
                        else:
                            logger.warning(f"⚠️ 예상치 못한 마스크 차원: {mask_tensor.shape}")
                            continue

                        collected_masks.append(mask_data)
                    else:
                        logger.debug(f"   ✗ 마스크 못찾음")
                except Exception as e:
                    logger.debug(f"포인트 {pt} 처리 중 오류: {e}")
                    continue

            logger.info(f"📊 총 {len(collected_masks)}개의 마스크 수집됨")

            # 3. 중복 마스크 제거 (NMS와 유사한 로직)
            unique_masks = []
            img_area = new_h * new_w
            
            for mask in collected_masks:
                mask_binary = (mask > 0).astype(np.uint8)
                mask_area = mask_binary.sum()
                
                # A. 크기 필터링 (너무 작거나 큰 것은 노이즈/배경)
                if mask_area < (img_area * 0.005) or mask_area > (img_area * 0.8):
                    continue
                
                # B. 중복 검사 (기존 찾은 잎들과 IoU 비교)
                is_duplicate = False
                for existing_mask in unique_masks:
                    iou = self._calculate_iou(mask_binary, existing_mask)
                    if iou > 0.6: # 60% 이상 겹치면 같은 잎으로 간주
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    unique_masks.append(mask_binary)

            # 4. 결과 정리
            valid_leaf_count = len(unique_masks)

            # 시각화용 합치기
            combined_mask = np.zeros((new_h, new_w), dtype=np.uint8)
            colored_masks = np.zeros((new_h, new_w, 3), dtype=np.uint8)

            # 각 마스크에 다른 색상 부여
            for i, mask in enumerate(unique_masks):
                # 회색톤 마스크 합치기
                combined_mask = np.maximum(combined_mask, mask * 255)

                # 색상 마스크 생성 (각 잎마다 다른 색)
                color = (
                    int(100 + (i * 50) % 155),
                    int(100 + (i * 100) % 155),
                    int(100 + (i * 150) % 155)
                )
                colored_masks[mask > 0] = color

            # 원본 크기로 마스크 복원 (시각화 저장을 위해)
            combined_mask_orig = cv2.resize(combined_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            colored_masks_orig = cv2.resize(colored_masks, (w, h), interpolation=cv2.INTER_NEAREST)

            stage_name, message = self._determine_stage(valid_leaf_count)

            return {
                "leaf_count": valid_leaf_count,
                "stage": stage_name,
                "message": message,
                "mask": combined_mask_orig,
                "colored_mask": colored_masks_orig,
                "raw_detected_count": len(collected_masks),
                "unique_mask_count": valid_leaf_count
            }

        except Exception as e:
            logger.error(f"❌ SAM 성장 단계 분석 중 오류: {e}")
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
            origin_img_pil = Image.open(io.BytesIO(image_bytes))
            origin_img_pil = ImageOps.exif_transpose(origin_img_pil).convert("RGB")
            origin_img_bgr = cv2.cvtColor(np.array(origin_img_pil), cv2.COLOR_RGB2BGR)

            # 2. YOLO 탐지 (Crop & Scale)
            results = self.det_model(origin_img_pil, conf=0.15, verbose=False)
            
            mm_per_pixel = 0.0
            basil_crop_pil = None
            basil_crop_bgr = None
            
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
                mm_per_pixel = 0.1

            # 3. PLA 계산
            pla_result = self._calculate_pla(basil_crop_bgr, mm_per_pixel)

            # 4. 건강 상태 분류
            cls_results = self.cls_model(basil_crop_pil, verbose=False)[0]
            health_status = cls_results.names[cls_results.probs.top1]
            health_conf = float(cls_results.probs.top1conf) * 100

            # 5. 성장 단계 분석 (그리드 프롬프트 + SAM 2 Large)
            logger.info("🔍 성장 단계 정밀 분석 중 (SAM 2 + Grid Prompt)...")
            growth_info = self._analyze_growth_with_sam(basil_crop_pil)

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
        try:
            # 스크립트 파일명 기반으로 저장 디렉토리 생성
            script_path = os.path.abspath(__file__)
            script_dir = os.path.dirname(script_path)
            script_name = os.path.splitext(os.path.basename(script_path))[0]  # 확장자 제외한 파일명

            output_dir = os.path.join(script_dir, script_name)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # 저장 경로들
            result_crop_path = os.path.join(output_dir, f"{script_name}_분할결과.jpg")
            result_original_path = os.path.join(output_dir, f"{script_name}_원본.jpg")
            result_mask_path = os.path.join(output_dir, f"{script_name}_마스크.jpg")
            result_colored_mask_path = os.path.join(output_dir, f"{script_name}_색상마스크.jpg")

            # 원본 이미지 저장
            cv2.imwrite(result_original_path, original_img)
            logger.info(f"💾 원본 이미지 저장됨: {result_original_path}")

            if growth_info and growth_info['mask'] is not None:
                mask = growth_info['mask']
                if mask.shape[:2] != crop_img.shape[:2]:
                    mask = cv2.resize(mask, (crop_img.shape[1], crop_img.shape[0]), interpolation=cv2.INTER_NEAREST)

                # 1. 마스크 이미지 저장 (흑백)
                cv2.imwrite(result_mask_path, mask)
                logger.info(f"💾 마스크 이미지 저장됨: {result_mask_path}")

                # 2. 색상 마스크 저장 (각 잎마다 다른 색)
                if "colored_mask" in growth_info:
                    colored_mask = growth_info['colored_mask']
                    if colored_mask.shape[:2] != crop_img.shape[:2]:
                        colored_mask = cv2.resize(colored_mask, (crop_img.shape[1], crop_img.shape[0]), interpolation=cv2.INTER_NEAREST)
                    cv2.imwrite(result_colored_mask_path, colored_mask)
                    logger.info(f"💾 색상 마스크 저장됨: {result_colored_mask_path}")

                # 3. 초록색 오버레이 분할 결과
                color_mask = np.zeros_like(crop_img)
                color_mask[mask > 0] = [0, 255, 0]

                overlay_crop = cv2.addWeighted(crop_img, 0.7, color_mask, 0.3, 0)

                txt = f"{growth_info['stage']} (Leaves: {growth_info['leaf_count']})"
                cv2.putText(overlay_crop, txt, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                success = cv2.imwrite(result_crop_path, overlay_crop)
                if success:
                    logger.info(f"💾 분할 결과 저장됨: {result_crop_path}")
                else:
                    logger.error(f"❌ 분할 이미지 저장 실패: {result_crop_path}")
            else:
                logger.warning("⚠️ Growth info가 없어 분할 이미지를 저장할 수 없습니다")

            logger.info("✅ 전체 분석 완료")
        except Exception as e:
            logger.error(f"❌ 이미지 저장 중 오류: {e}", exc_info=True)

if __name__ == "__main__":
    analyzer = IntegratedBasilAnalyzer()
    
    print("\n" + "="*50)
    print(f"🚀 분석 시작: {TEST_IMAGE_PATH}")
    print("="*50)
    
    result = analyzer.process_file(TEST_IMAGE_PATH)
    
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