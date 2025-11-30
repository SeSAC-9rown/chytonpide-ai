from gradio_client import Client, handle_file
import shutil
import os

# 1. 설정
IMAGE_PATH = r"C:\Users\sega0\Desktop\chytonpide-ai\predict_image\test4.jpg" # 본인 경로로 수정
TEXT_PROMPT = "leaf" # 찾을 대상

def run_sam3():
    print(f"▶ SAM 3 서버(Hugging Face)에 요청 중... [찾을 것: {TEXT_PROMPT}]")
    
    # 서버 연결
    client = Client("akhaliq/sam3")
    
    try:
        # 2. 예측 요청 (문서에 나온대로 파라미터 수정됨)
        result = client.predict(
            image=handle_file(IMAGE_PATH), # 이미지
            text=TEXT_PROMPT,              # 텍스트 (예: leaf)
            threshold=0.7,                 # 감도 (0.5보다 낮추면 더 많이 찾음)
            mask_threshold=0.5,            # 마스크 정확도
            api_name="/segment"            # ★ 여기가 핵심 수정 사항 ★
        )
        
        # 3. 결과 확인
        # result는 (dict, str) 형태의 튜플로 옵니다.
        # result[0]에 이미지 경로와 마스크 정보가 들어있습니다.
        print("✅ 성공!")
        print(f"결과 데이터: {result}")
        
        # 결과물 경로 추출 (딕셔너리 형태일 수 있음)
        output_data = result[0]
        
        # 만약 결과가 파일 경로라면
        if isinstance(output_data, str) and os.path.exists(output_data):
            print(f"저장된 이미지 경로: {output_data}")
        # 만약 결과가 딕셔너리라면 (AnnotatedImage 형식)
        elif isinstance(output_data, dict):
            # 원본+마스크가 합쳐진 이미지 혹은 마스크 경로
            print("마스크 정보가 담긴 딕셔너리를 받았습니다.")
            print(output_data)

    except Exception as e:
        print(f"❌ 오류 발생: {e}")

if __name__ == "__main__":
    run_sam3()