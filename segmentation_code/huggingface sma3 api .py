import cv2
import numpy as np
import io
import logging
import os
import torch
import warnings
import json  # ★ JSON 저장을 위해 추가
import tempfile
import time  # ★ 시간 측정을 위해 추가
from PIL import Image, ImageOps
from ultralytics import YOLO

# ★ Hugging Face API 클라이언트
from gradio_client import Client, handle_file

# ==========================================
# [설정] 로그 및 경고 설정
# ==========================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# ==========================================
# [설정] 모델 및 파일 경로
# ==========================================
# 1. YOLO 모델 (로컬 실행)
DET_MODEL_PATH = r"runs\detect\det_exp1\weights\best.pt"
CLS_MODEL_PATH = r"runs\classify\test1\weights\best.pt"

# 2. SAM 3 API 설정
SAM3_API_URL = "akhaliq/sam3"  # Hugging Face Space ID
# 수정: 'basil leaves' -> 'leaf' (더 일반적인 단어가 인식률이 높음)
SAM_TEXT_PROMPT = "leaf" 

# 3. 테스트할 이미지 경로
TEST_IMAGE_PATH = r"C:\Users\sega0\Desktop\chytonpide-ai\predict_image\test6.jpg"

# 4. 기타 설정
SCALE_REAL_DIAMETER_MM = 16.0
GREEN_HSV_LOWER = [35, 40, 40]
GREEN_HSV_UPPER = [85, 255, 255]


class HybridBasilAnalyzer:
    """
    [하이브리드 분석기]
    - 로컬: YOLO (탐지, 분류, PLA)
    - 클라우드 API: SAM 3 (잎 정밀 분할)
    """

    def __init__(self):
        logger.info("🤖 AI 모델 로딩 시작...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            # 1. YOLO 모델 로딩 (로컬)
            self.det_model = YOLO(DET_MODEL_PATH)
            self.cls_model = YOLO(CLS_MODEL_PATH)
            logger.info("✅ YOLO 모델 로딩 완료")

            # 2. SAM 3 API 클라이언트 연결
            logger.info(f"☁️ SAM 3 API 서버 연결 중... ({SAM3_API_URL})")
            self.sam_client = Client(SAM3_API_URL)
            logger.info("✅ API 서버 연결 성공!")

        except Exception as e:
            logger.error(f"❌ 초기화 실패: {e}")
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

    def _call_sam3_api(self, basil_crop_pil):
        """
        바질 Crop 이미지를 API로 보내 분석 결과를 받아옴
        """
        temp_path = "temp_crop_for_api.jpg"
        
        try:
            # 1. API 전송을 위해 임시 파일로 저장
            basil_crop_pil.save(temp_path)
            
            logger.info(f"🚀 API 요청 전송 (Prompt: '{SAM_TEXT_PROMPT}')... 대기 중...")
            
            # 2. API 호출 (감도 조절)
            result = self.sam_client.predict(
                image=handle_file(temp_path),
                text=SAM_TEXT_PROMPT,
                threshold=0.4,      
                mask_threshold=0.4,
                api_name="/segment"
            )
            
            logger.info(f"📡 API Raw Result: {result}")

            # 3. 결과 파싱 (업데이트된 로직)
            # 구조: ({'image': '...', 'annotations': [{'image': 'path/to/mask.png', 'label': ...}, ...]}, "Message")
            
            combined_mask = None
            leaf_count = 0
            
            # 튜플/리스트이고 첫 번째 요소가 딕셔너리인 경우 (정상 응답)
            if isinstance(result, (tuple, list)) and len(result) > 0 and isinstance(result[0], dict):
                data = result[0]
                
                # 'annotations' 키 확인
                if 'annotations' in data and isinstance(data['annotations'], list):
                    annotations = data['annotations']
                    leaf_count = len(annotations)
                    logger.info(f"✅ API에서 {leaf_count}개의 잎(annotations)을 발견했습니다.")
                    
                    # 각 마스크 이미지 로드 및 병합
                    for i, item in enumerate(annotations):
                        mask_path = item.get('image')
                        if mask_path and os.path.exists(mask_path):
                            # 마스크 로드 (흑백)
                            part_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                            
                            if part_mask is not None:
                                # 캔버스 초기화 (첫 마스크 크기에 맞춤)
                                if combined_mask is None:
                                    combined_mask = np.zeros_like(part_mask)
                                
                                # 크기가 다를 경우 안전장치 (Resize)
                                if part_mask.shape != combined_mask.shape:
                                    part_mask = cv2.resize(part_mask, (combined_mask.shape[1], combined_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
                                    
                                # 마스크 합치기 (OR 연산)
                                combined_mask = np.maximum(combined_mask, part_mask)
            
            # 만약 위 구조가 아니라면 (예전 방식 Fallback)
            elif isinstance(result, (tuple, list)) and len(result) > 1 and isinstance(result[1], str) and os.path.exists(result[1]):
                mask_path = result[1]
                logger.info("👉 Fallback: 단일 마스크 파일 경로 사용")
                mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask_img is not None:
                    _, combined_mask = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)
                    num_labels, _, _, _ = cv2.connectedComponentsWithStats(combined_mask, connectivity=8)
                    leaf_count = num_labels - 1

            if combined_mask is None:
                logger.warning(f"⚠️ 유효한 마스크를 생성하지 못했습니다. (Result: {result})")
                return None

            # 4. 결과 정리
            stage_name, message = self._determine_stage(leaf_count)
            
            return {
                "leaf_count": leaf_count,
                "stage": stage_name,
                "message": message,
                "mask": combined_mask # 시각화용
            }

        except Exception as e:
            logger.error(f"❌ API 통신 중 오류: {e}")
            return None
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

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
        # 전체 프로세스 시작 시간 측정
        total_start_time = time.time()
        
        try:
            # 1. 이미지 로드
            origin_img_pil = Image.open(io.BytesIO(image_bytes))
            origin_img_pil = ImageOps.exif_transpose(origin_img_pil).convert("RGB")
            origin_img_bgr = cv2.cvtColor(np.array(origin_img_pil), cv2.COLOR_RGB2BGR)

            # 2. YOLO 탐지 (Crop & Scale) - 로컬 수행
            yolo_start_time = time.time()
            results = self.det_model(origin_img_pil, conf=0.15, verbose=False)
            yolo_end_time = time.time()
            yolo_duration = yolo_end_time - yolo_start_time
            
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

            # 3. API 호출 (시간 측정)
            api_start_time = time.time()
            growth_info = self._call_sam3_api(basil_crop_pil)
            api_end_time = time.time()
            api_duration = api_end_time - api_start_time

            # 4. 기타 분석 (PLA, Health)
            pla_result = self._calculate_pla(basil_crop_bgr, mm_per_pixel)
            
            cls_res = self.cls_model(basil_crop_pil, verbose=False)[0]
            health = cls_res.names[cls_res.probs.top1]
            conf = float(cls_res.probs.top1conf) * 100

            # 5. 시각화 저장
            self._save_visualization(basil_crop_bgr, growth_info)

            # 6. JSON 저장을 위해 mask 데이터(numpy 배열)는 제거
            growth_data_for_json = None
            if growth_info:
                growth_data_for_json = growth_info.copy()
                if 'mask' in growth_data_for_json:
                    del growth_data_for_json['mask']  # numpy 배열은 JSON 저장 불가하므로 제거

            # 전체 종료 시간 측정
            total_end_time = time.time()
            total_duration = total_end_time - total_start_time
            
            # 로그 출력
            logger.info(f"⏱️ 실행 시간 - 총 소요: {total_duration:.2f}초 (YOLO: {yolo_duration:.2f}초, API: {api_duration:.2f}초)")

            return {
                "status": "success",
                "data": {
                    "health": {"status": health, "confidence": f"{conf:.2f}%"},
                    "pla": pla_result,
                    "growth": growth_data_for_json,
                    "execution_time": {
                        "total_seconds": round(total_duration, 2),
                        "yolo_seconds": round(yolo_duration, 2),
                        "api_seconds": round(api_duration, 2)
                    }
                }
            }

        except Exception as e:
            logger.error(f"오류: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    def _save_visualization(self, crop_img, growth_info):
        """결과 저장"""
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
                
                cv2.imwrite("result_api_hybrid.jpg", overlay)
                logger.info("💾 결과 이미지 저장 완료: result_api_hybrid.jpg")
        except:
            pass

if __name__ == "__main__":
    analyzer = HybridBasilAnalyzer()
    print(f"\n🚀 하이브리드 분석 시작: {TEST_IMAGE_PATH}")
    
    # 분석 실행 및 결과 받기
    final_result = analyzer.process_file(TEST_IMAGE_PATH)
    
    # 결과 처리
    if final_result:
        # 1. JSON 파일로 저장
        json_filename = "result.json"
        try:
            with open(json_filename, "w", encoding="utf-8") as f:
                json.dump(final_result, f, ensure_ascii=False, indent=4)
            print(f"\n💾 결과 데이터(JSON) 저장 완료: {os.path.abspath(json_filename)}")
        except Exception as e:
            print(f"❌ JSON 저장 실패: {e}")

        # 2. 콘솔 로그에 출력 (복사해서 쓰기 좋게)
        print("\n📊 [분석 결과 JSON 출력]")
        print("="*40)
        print(json.dumps(final_result, ensure_ascii=False, indent=4))
        print("="*40)
    else:
        print("❌ 분석 결과가 없습니다.")