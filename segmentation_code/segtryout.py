import torch
import cv2
import numpy as np
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from PIL import Image
import os
import warnings
import logging

# ==========================================
# [설정] 경고 무시
# ==========================================
warnings.filterwarnings("ignore") 
logging.getLogger("transformers").setLevel(logging.ERROR)

# ==========================================
# 1. 설정 (파일 경로 확인 필수)
# ==========================================
MODEL_NAME = "nvidia/mit-b0" 
WEIGHT_PATH = r"runs\seg\PA\PA_MIOU.pth" 
IMAGE_PATH = r"C:\Users\sega0\Desktop\chytonpide-ai\predict_image\test4.jpg"

# ★★★ 핵심 수정: 클래스 개수를 2개(배경, 잎)로 고정 ★★★
NUM_CLASSES = 2 

# ==========================================
# 2. 생육 단계 판별 로직
# ==========================================
def determine_stage(leaf_count):
    stage_name = "알 수 없음"
    msg = ""
    if leaf_count <= 2:
        stage_name = "🌱 떡잎 단계"
        msg = "떡잎만 존재하거나, 본엽이 나오기 직전입니다."
    elif 3 <= leaf_count <= 4:
        stage_name = "🌿 본엽 2매"
        msg = "본엽이 1쌍(2장) 전개된 상태입니다."
    elif 5 <= leaf_count <= 8:
        stage_name = "🌿 본엽 4매 ~ 8매"
        msg = "본엽이 2쌍에서 4쌍까지 활발히 자라는 중입니다."
    elif 9 <= leaf_count <= 10:
        stage_name = "🌿 본엽 8매 ~ 10매"
        msg = "본엽 성장이 거의 완료되어 가며, 곧 분지가 예상됩니다."
    else:
        stage_name = "🌳 분지 발생"
        msg = "잎이 10매 이상이며, 곁가지(분지)가 발달하는 단계입니다."
    return stage_name, msg

# ==========================================
# 3. 모델 추론 엔진
# ==========================================
def run_inference(image_path, weight_path):
    if not os.path.exists(image_path):
        print(f"❌ 에러: 이미지를 찾을 수 없습니다 -> {image_path}")
        return None, None

    print(f"▶ AI 분석 시작... ({os.path.basename(image_path)})")
    
    try:
        # [수정] num_labels=NUM_CLASSES (2)를 반드시 넣어줘야 가중치가 맞습니다.
        id2label = {0: "background", 1: "leaf"}
        label2id = {"background": 0, "leaf": 1}
        
        model = SegformerForSemanticSegmentation.from_pretrained(
            MODEL_NAME, 
            num_labels=NUM_CLASSES, 
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True
        )
    except Exception as e:
        print(f"❌ 모델 초기화 실패: {e}")
        return None, None

    try:
        checkpoint = torch.load(weight_path, map_location="cpu")
        # 가중치 로드 시도
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
             model.load_state_dict(checkpoint, strict=False)
        print("▶ 가중치 적용 성공!")
    except Exception as e:
        print(f"❌ 가중치 파일 로드 실패: {e}")
        return None, None

    model.eval()
    image_processor = SegformerImageProcessor.from_pretrained(MODEL_NAME)

    # 이미지 리사이징 (속도 향상 & 메모리 절약)
    raw_image = Image.open(image_path).convert("RGB")
    image = raw_image.resize((640, 640)) 

    inputs = image_processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        
    logits = outputs.logits
    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=image.size[::-1], 
        mode="bilinear",
        align_corners=False,
    )
    
    pred_mask = upsampled_logits.argmax(dim=1)[0].numpy().astype(np.uint8)
    return pred_mask, np.array(image)

# ==========================================
# 4. 결과 분석 및 저장
# ==========================================
def analyze_and_show(mask, original_image):
    if mask is None: return

    # 마스크에서 잎(1) 부분만 추출
    leaf_mask = (mask == 1).astype(np.uint8) * 255
    
    # 잎 개수 세기
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(leaf_mask, connectivity=8)
    leaf_count = num_labels - 1 

    current_stage, message = determine_stage(leaf_count)

    print("\n" + "="*40)
    print(f"🌱 [AI 스마트팜 분석 결과]")
    print(f"="*40)
    print(f"📸 잎 개수      : {leaf_count} 장")
    print(f"🏆 현재 단계    : {current_stage}")
    print(f"💬 상세 설명    : {message}")
    print("="*40)

    # 결과 이미지 생성 (초록색 마스킹)
    color_mask = np.zeros_like(original_image)
    color_mask[mask == 1] = [0, 255, 0] 
    
    result_img = cv2.addWeighted(original_image, 0.7, color_mask, 0.3, 0)
    
    # 텍스트 그리기
    cv2.putText(result_img, f"Stage: {current_stage.split(' ')[-1]}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(result_img, f"Leaf Count: {leaf_count}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # 파일로 저장 (에러 방지)
    save_filename = "result_analysis.jpg"
    final_img_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_filename, final_img_bgr)
    
    print(f"\n💾 결과 이미지가 저장되었습니다: {os.path.abspath(save_filename)}")

if __name__ == "__main__":
    mask_result, img_result = run_inference(IMAGE_PATH, WEIGHT_PATH)
    analyze_and_show(mask_result, img_result)