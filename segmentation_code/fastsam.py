import cv2
import numpy as np
import logging
from PIL import Image, ImageOps
from ultralytics import YOLO, FastSAM
import os

# [수정 1] GPU가 있어도 강제로 CPU만 사용하게 설정 (가장 확실한 방법)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# 설정
# ==========================================
DET_MODEL_PATH = r"runs\detect\det_exp1\weights\best.pt"
CLS_MODEL_PATH = r"runs\classify\test1\weights\best.pt"
SCALE_REAL_DIAMETER_MM = 16
GREEN_HSV_LOWER = [35, 40, 40]
GREEN_HSV_UPPER = [85, 255, 255]
SAVE_DIR = "results"
DEVICE = "cpu"  # [수정 2] 디바이스 변수 설정

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)


class BasilAnalyzer:
    def __init__(self):
        logger.info(f"🤖 AI 모델 로딩 시작... (Device: {DEVICE})")

        self.det_model = YOLO(DET_MODEL_PATH)
        self.cls_model = YOLO(CLS_MODEL_PATH)
        
        logger.info("🌿 SAM2 (FastSAM) 모델 로딩 중...")
        self.sam_model = FastSAM("FastSAM-s.pt")
        
        logger.info("✅ 모델 로딩 완료!")

    def _separate_overlapping_leaves(self, mask):
        """겹친 잎 분리 (Watershed)"""
        # 거리 변환
        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        
        # 로컬 최대값 찾기 (각 잎의 중심)
        _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        
        # 확실한 배경 영역
        kernel = np.ones((3, 3), np.uint8)
        sure_bg = cv2.dilate(mask, kernel, iterations=3)
        
        # 불확실한 영역
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # 마커 생성
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        # Watershed 적용
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(mask_bgr, markers)
        
        return markers

    def _count_leaves(self, basil_crop_bgr, mm_per_pixel):
        """SAM2 + Watershed를 이용한 잎 개수 세기"""
        try:
            logger.info("🔍 SAM2 세그멘테이션 진행 중...")
            # [수정 3] FastSAM 실행 시 device='cpu' 추가
            results = self.sam_model(basil_crop_bgr, device=DEVICE)

            basil_hsv = cv2.cvtColor(basil_crop_bgr, cv2.COLOR_BGR2HSV)
            lower_green = np.array(GREEN_HSV_LOWER, dtype=np.uint8)
            upper_green = np.array(GREEN_HSV_UPPER, dtype=np.uint8)
            green_mask = cv2.inRange(basil_hsv, lower_green, upper_green)

            leaf_count = 0
            leaf_areas = []
            min_leaf_pixels = 100

            # 시각화용 이미지들
            vis_all_masks = basil_crop_bgr.copy()
            vis_selected = basil_crop_bgr.copy()
            vis_watershed = basil_crop_bgr.copy()
            vis_green_mask = cv2.cvtColor(green_mask, cv2.COLOR_GRAY2BGR)

            def get_color(idx):
                np.random.seed(idx * 10)
                return tuple(map(int, np.random.randint(0, 255, 3)))

            if results[0].masks is not None:
                masks = results[0].masks.data.cpu().numpy()
                logger.info(f"📊 SAM이 찾은 총 마스크 개수: {len(masks)}개")
                
                for i, mask in enumerate(masks):
                    mask_uint8 = (mask * 255).astype(np.uint8)
                    
                    if mask_uint8.shape != green_mask.shape:
                        mask_uint8 = cv2.resize(
                            mask_uint8, 
                            (green_mask.shape[1], green_mask.shape[0]),
                            interpolation=cv2.INTER_NEAREST
                        )
                    
                    mask_pixels = np.sum(mask_uint8 > 127)
                    if mask_pixels < min_leaf_pixels:
                        continue

                    overlap = np.sum((mask_uint8 > 127) & (green_mask > 0))
                    overlap_ratio = (overlap / mask_pixels) if mask_pixels > 0 else 0

                    # 모든 마스크 시각화
                    color_all = get_color(i)
                    vis_all_masks[mask_uint8 > 127] = color_all

                    # 잎으로 판단된 마스크 (초록색 비율 50% 이상)
                    if overlap_ratio > 0.5:
                        # Watershed로 겹친 잎 분리 시도
                        markers = self._separate_overlapping_leaves(mask_uint8)
                        
                        # 분리된 각 영역 처리 (0=경계, 1=배경, 2+=객체)
                        unique_labels = np.unique(markers)
                        separated_count = 0
                        
                        for label in unique_labels:
                            if label <= 1:  # 배경, 경계 스킵
                                continue
                            
                            # 해당 라벨의 마스크
                            label_mask = (markers == label).astype(np.uint8) * 255
                            label_pixels = np.sum(label_mask > 0)
                            
                            # 너무 작으면 스킵
                            if label_pixels < min_leaf_pixels:
                                continue
                            
                            # 초록 영역과 겹치는 부분
                            label_overlap = np.sum((label_mask > 0) & (green_mask > 0))
                            label_ratio = label_overlap / label_pixels if label_pixels > 0 else 0
                            
                            if label_ratio > 0.4:  # 40% 이상 초록이면 잎으로 카운트
                                leaf_count += 1
                                separated_count += 1
                                
                                leaf_area_mm2 = label_overlap * (mm_per_pixel ** 2)
                                leaf_areas.append({
                                    "leaf_id": leaf_count,
                                    "area_mm2": round(leaf_area_mm2, 2),
                                    "area_cm2": round(leaf_area_mm2 / 100, 2),
                                    "pixels": int(label_overlap),
                                    "overlap_ratio": round(label_ratio * 100, 1)
                                })
                                
                                # 시각화
                                color_leaf = get_color(leaf_count + 100)
                                vis_selected[label_mask > 0] = color_leaf
                                vis_watershed[label_mask > 0] = color_leaf
                                
                                # 잎 번호 표시
                                contours, _ = cv2.findContours(label_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                if contours:
                                    M = cv2.moments(contours[0])
                                    if M["m00"] > 0:
                                        cx = int(M["m10"] / M["m00"])
                                        cy = int(M["m01"] / M["m00"])
                                        cv2.putText(vis_selected, str(leaf_count), (cx, cy),
                                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                                        cv2.putText(vis_watershed, str(leaf_count), (cx, cy),
                                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        logger.info(f"  ✅ 마스크 #{i} → Watershed로 {separated_count}개 잎 분리됨")
                    else:
                        logger.info(f"  ❌ 마스크 #{i} 제외 (초록비율: {overlap_ratio*100:.1f}%)")

            # 결과 이미지 저장
            cv2.imwrite(f"{SAVE_DIR}/1_original_crop.jpg", basil_crop_bgr)
            cv2.imwrite(f"{SAVE_DIR}/2_green_mask.jpg", vis_green_mask)
            cv2.imwrite(f"{SAVE_DIR}/3_all_sam_masks.jpg", vis_all_masks)
            cv2.imwrite(f"{SAVE_DIR}/4_watershed_result.jpg", vis_watershed)
            cv2.imwrite(f"{SAVE_DIR}/5_selected_leaves.jpg", vis_selected)

            # 최종 오버레이
            overlay = cv2.addWeighted(basil_crop_bgr, 0.5, vis_selected, 0.5, 0)
            cv2.putText(overlay, f"Leaf Count: {leaf_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imwrite(f"{SAVE_DIR}/6_final_overlay.jpg", overlay)

            logger.info(f"🌿 잎 개수: {leaf_count}개")
            logger.info(f"💾 결과 이미지 저장됨: {os.path.abspath(SAVE_DIR)}")

            return {
                "leaf_count": leaf_count,
                "leaf_details": leaf_areas,
                "average_leaf_area_mm2": round(
                    sum(l["area_mm2"] for l in leaf_areas) / leaf_count, 2
                ) if leaf_count > 0 else 0
            }

        except Exception as e:
            logger.error(f"❌ 잎 개수 세기 중 오류: {e}")
            import traceback
            traceback.print_exc()
            return {"leaf_count": 0, "leaf_details": [], "average_leaf_area_mm2": 0}

    def _calculate_pla(self, basil_crop_bgr, mm_per_pixel):
        """PLA 계산"""
        try:
            basil_hsv = cv2.cvtColor(basil_crop_bgr, cv2.COLOR_BGR2HSV)
            lower_green = np.array(GREEN_HSV_LOWER, dtype=np.uint8)
            upper_green = np.array(GREEN_HSV_UPPER, dtype=np.uint8)
            green_mask = cv2.inRange(basil_hsv, lower_green, upper_green)

            kernel = np.ones((3, 3), np.uint8)
            green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)

            green_pixel_count = cv2.countNonZero(green_mask)
            area_mm2 = green_pixel_count * (mm_per_pixel ** 2)

            logger.info(f"[PLA] 면적: {area_mm2:.2f} mm²")

            return {
                "pla_mm2": round(area_mm2, 2),
                "pla_cm2": round(area_mm2 / 100, 2),
                "green_pixels": int(green_pixel_count),
            }
        except Exception as e:
            logger.error(f"❌ PLA 계산 중 오류: {e}")
            return None

    def process(self, image_path):
        """이미지 분석"""
        try:
            origin_img_pil = Image.open(image_path)
            origin_img_pil = ImageOps.exif_transpose(origin_img_pil)
            origin_img_pil = origin_img_pil.convert("RGB")
            origin_img_bgr = cv2.cvtColor(np.array(origin_img_pil), cv2.COLOR_RGB2BGR)
            
            logger.info(f"📸 이미지 로드됨: {image_path}")

            # [수정 4] Detection 실행 시 device='cpu' 추가
            results = self.det_model(origin_img_pil, conf=0.15, device=DEVICE)
            found_ids = results[0].boxes.cls.cpu().numpy().astype(int) if len(results[0].boxes) > 0 else []
            logger.info(f"👉 탐지된 ID: {list(set(found_ids))}")

            mm_per_pixel = 0
            if 1 not in found_ids:
                return {"status": "error", "message": "Scale 마커 없음"}

            for box in results[0].boxes:
                if int(box.cls[0]) == 1:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    diameter_pixel = max(x2 - x1, y2 - y1)
                    mm_per_pixel = SCALE_REAL_DIAMETER_MM / diameter_pixel
                    logger.info(f"[Scale] 1px = {mm_per_pixel:.4f}mm")
                    break

            if 0 not in found_ids:
                return {"status": "error", "message": "바질 없음"}

            basil_crop_bgr = None
            for box in results[0].boxes:
                if int(box.cls[0]) == 0:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    basil_crop_bgr = origin_img_bgr[y1:y2, x1:x2]
                    break

            pla_result = self._calculate_pla(basil_crop_bgr, mm_per_pixel)
            leaf_result = self._count_leaves(basil_crop_bgr, mm_per_pixel)

            basil_crop_pil = Image.fromarray(cv2.cvtColor(basil_crop_bgr, cv2.COLOR_BGR2RGB))
            
            # [수정 5] Classification 실행 시 device='cpu' 추가
            cls_results = self.cls_model(basil_crop_pil, device=DEVICE)[0]
            class_name = cls_results.names[cls_results.probs.top1]
            confidence = float(cls_results.probs.top1conf) * 100

            return {
                "status": "success",
                "data": {
                    "diagnosis": class_name,
                    "confidence": f"{confidence:.2f}%",
                    "pla_mm2": pla_result["pla_mm2"],
                    "pla_cm2": pla_result["pla_cm2"],
                    "leaf_count": leaf_result["leaf_count"],
                    "average_leaf_area_mm2": leaf_result["average_leaf_area_mm2"],
                    "leaf_details": leaf_result["leaf_details"],
                }
            }

        except Exception as e:
            logger.error(f"❌ 오류: {e}")
            import traceback
            traceback.print_exc()
            return {"status": "error", "message": str(e)}


# ==========================================
# 테스트 실행
# ==========================================
if __name__ == "__main__":
    TEST_IMAGE = r"predict_image\test8.jpg"
    
    analyzer = BasilAnalyzer()
    result = analyzer.process(TEST_IMAGE)
    
    print("\n" + "="*50)
    print("📊 분석 결과")
    print("="*50)
    
    if result["status"] == "success":
        data = result["data"]
        print(f"🏥 진단: {data['diagnosis']} ({data['confidence']})")
        print(f"📐 총 엽면적: {data['pla_cm2']} cm²")
        print(f"🌿 잎 개수: {data['leaf_count']}개")
        print(f"📏 평균 잎 면적: {data['average_leaf_area_mm2']} mm²")
        print("\n개별 잎 정보:")
        for leaf in data["leaf_details"]:
            print(f"  - 잎 #{leaf['leaf_id']}: {leaf['area_mm2']} mm² (초록비율: {leaf['overlap_ratio']}%)")
        
        print(f"\n💾 시각화 결과 저장 위치: {os.path.abspath(SAVE_DIR)}")
        print("  - 4_watershed_result.jpg : Watershed 분리 결과")
    else:
        print(f"❌ 에러: {result['message']}")