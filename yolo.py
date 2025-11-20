from ultralytics import YOLO

def main():
    # 1. 모델 로드
    model = YOLO('yolo11n.pt')

    # 2. 모델 학습
    results = model.train(
        # 👇 여기에 r을 붙여서 경로를 넣었습니다 (복사해서 붙여넣기 하세요)
        data=r"C:\Users\sega0\Desktop\grown\basil_yolov11\data.yaml",
        
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        
        # 저장 경로 설정
        project=r'C:\Users\sega0\Desktop\grown',
        name='result',
        exist_ok=True
    )

    print("학습 완료!")
    print(f"결과 저장 위치: {results.save_dir}")

if __name__ == '__main__':
    main()