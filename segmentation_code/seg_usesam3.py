from gradio_client import Client, handle_file
import shutil
import os
import cv2
import numpy as np

# ==========================================
# 1. 설정
# ==========================================
IMAGE_PATH = r"C:\Users\sega0\Desktop\chytonpide-ai\predict_image\test6.jpg"  # 본인 경로 확인
TEXT_PROMPT = "leaf"    # 찾을 대상
SAVE_DIR = "sam_results" # 결과 저장할 폴더

# 결과 저장 폴더 생성
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def run_sam3_and_count():
    print(f"▶ SAM 3 서버(Hugging Face)에 요청 중... [찾을 것: {TEXT_PROMPT}]")
    print("   (서버 상황에 따라 시간이 조금 걸릴 수 있습니다...)")
    
    # 서버 연결
    client = Client("akhaliq/sam3")
    
    try:
        # ---------------------------------------------------------
        # 2. 예측 요청
        # ---------------------------------------------------------
        # result는 보통 (원본+마스크 이미지 경로, 마스크 이미지 경로, ...) 형태의 튜플로 반환됩니다.
        # 반환 형식은 스페이스 업데이트에 따라 다를 수 있어 확인이 필요합니다.
        result = client.predict(
            image=handle_file(IMAGE_PATH),
            text=TEXT_PROMPT,
            threshold=0.7,
            mask_threshold=0.5,
            api_name="/segment"
        )
        
        print("✅ 서버 응답 완료!")
        
        # ---------------------------------------------------------
        # 3. 결과 파싱 및 이미지 로드
        # ---------------------------------------------------------
        mask_path = None
        
        # result가 튜플이나 리스트인 경우 (보통 [이미지, 마스크] 순서)
        if isinstance(result, (tuple, list)):
            # 보통 두 번째 요소나 세 번째 요소가 순수한 흑백 마스크일 확률이 높습니다.
            # 여기서는 반환된 모든 경로를 확인해 봅니다.
            print(f"반환된 데이터 개수: {len(result)}")
            
            # 마스크 경로 찾기 (보통 끝쪽이나 1번 인덱스)
            # API마다 다르므로, 여기서는 result[1]이 마스크라고 가정하고 처리합니다.
            # 만약 result[1]이 없다면 result[0]을 사용합니다.
            target_index = 1 if len(result) > 1 else 0
            mask_path = result[target_index]
            
        elif isinstance(result, str):
            mask_path = result
            
        elif isinstance(result, dict) and 'image' in result:
             mask_path = result['image']
        
        if not mask_path or not os.path.exists(mask_path):
            print("❌ 유효한 마스크 이미지 경로를 찾지 못했습니다.")
            print(f"전체 결과: {result}")
            return

        print(f"📂 마스크 이미지 경로: {mask_path}")

        # ---------------------------------------------------------
        # 4. 잎 개수 세기 (OpenCV)
        # ---------------------------------------------------------
        # 이미지 읽기 (흑백 모드)
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if mask_img is None:
            print("❌ 이미지를 읽을 수 없습니다.")
            return

        # 이진화 (혹시 모를 노이즈 제거)
        # 127보다 밝은 픽셀(잎)은 255(흰색), 나머지는 0(검은색)
        _, binary_mask = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)

        # 연결된 성분(덩어리) 찾기
        # num_labels: 덩어리 개수 (배경 포함)
        # stats: 각 덩어리의 위치 및 크기 정보
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        
        # 배경(0번 레이블)을 제외한 개수
        leaf_count = num_labels - 1
        
        print("\n" + "="*40)
        print(f"🌿 분석 결과")
        print(f"="*40)
        print(f"👉 텍스트 프롬프트 : {TEXT_PROMPT}")
        print(f"👉 추정된 잎 개수  : {leaf_count} 개")
        print("="*40)

        # ---------------------------------------------------------
        # 5. 결과 시각화 저장
        # ---------------------------------------------------------
        # 원본 이미지 읽기
        original_img = cv2.imread(IMAGE_PATH)
        if original_img is not None:
            # 마스크 크기 맞추기
            binary_mask_resized = cv2.resize(binary_mask, (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            # 초록색 오버레이 만들기
            color_mask = np.zeros_like(original_img)
            color_mask[binary_mask_resized == 255] = [0, 255, 0] # 초록색

            # 원본 + 마스크 합성
            result_img = cv2.addWeighted(original_img, 0.7, color_mask, 0.3, 0)
            
            # 텍스트 쓰기
            cv2.putText(result_img, f"Count: {leaf_count}", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            save_path = os.path.join(SAVE_DIR, "final_count_result.jpg")
            cv2.imwrite(save_path, result_img)
            print(f"💾 결과 이미지 저장됨: {os.path.abspath(save_path)}")

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_sam3_and_count()