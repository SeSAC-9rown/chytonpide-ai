from ultralytics import YOLO
from pathlib import Path
import argparse
import os
import cv2

def predict(model_path, source, save=True, save_crop=True, save_dir='runs/predict_det', conf=0.25, imgsz=640):
    """모델로 예측 수행"""

    # 모델 로드
    print("=" * 60)
    print("🔍 YOLO Object Detection 모델 예측")
    print("=" * 60)

    print(f"\n📦 모델 로드: {model_path}")
    model = YOLO(model_path)

    print(f"🖼️  입력: {source}")
    print(f"📊 신뢰도 임계값: {conf}")
    print(f"📐 이미지 크기: {imgsz}")
    print(f"💾 결과 저장: {'활성화' if save else '비활성화'}")
    print(f"✂️  Crop 저장: {'활성화' if save_crop else '비활성화'}")
    print()
    
    # 예측 수행
    results = model.predict(
        source=source,
        save=save,
        save_txt=save,
        project=save_dir,
        conf=conf,
        imgsz=imgsz
    )

    # YOLO가 생성한 저장 폴더 찾기 (predict, predict2, predict3...)
    if save_crop and len(results) > 0:
        # results의 save_dir에서 실제 저장 경로 추출
        result_save_dir = Path(results[0].save_dir)
        crop_dir = result_save_dir / "crop"
        crop_dir.mkdir(parents=True, exist_ok=True)
    
    # 결과 출력
    print("\n" + "=" * 60)
    print("📋 예측 결과")
    print("=" * 60)
    
    total_crops = 0
    
    for i, result in enumerate(results):
        print(f"\n🖼️  이미지 {i+1}: {Path(result.path).name}")
        
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes
            names = result.names
            
            print(f"   탐지된 객체: {len(boxes)}개")
            
            # 클래스별 카운트
            class_counts = {}
            for box in boxes:
                cls_id = int(box.cls.item())
                cls_name = names[cls_id]
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
            
            for cls_name, count in class_counts.items():
                print(f"      • {cls_name}: {count}개")
            
            # Crop 저장
            if save_crop:
                # 원본 이미지 로드
                img = cv2.imread(result.path)
                img_name = Path(result.path).stem
                
                for j, box in enumerate(boxes):
                    cls_id = int(box.cls.item())
                    cls_name = names[cls_id]
                    conf_val = box.conf.item()
                    
                    # 바운딩박스 좌표
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    
                    # 이미지 crop
                    crop_img = img[y1:y2, x1:x2]
                    
                    # 저장
                    crop_filename = f"{img_name}_{cls_name}_{j+1}_{conf_val*100:.0f}.jpg"
                    crop_path = crop_dir / crop_filename
                    cv2.imwrite(str(crop_path), crop_img)
                    total_crops += 1
            
            # 상세 정보 (상위 5개)
            print(f"\n   상세 정보 (상위 5개):")
            for j, box in enumerate(boxes[:5]):
                cls_id = int(box.cls.item())
                cls_name = names[cls_id]
                conf_val = box.conf.item()
                xyxy = box.xyxy[0].tolist()
                print(f"      {j+1}. {cls_name} ({conf_val*100:.1f}%) - [{xyxy[0]:.0f}, {xyxy[1]:.0f}, {xyxy[2]:.0f}, {xyxy[3]:.0f}]")
        else:
            print("   탐지된 객체 없음")
    
    if save:
        print(f"\n💾 결과 저장됨: {save_dir}")
    
    if save_crop and total_crops > 0:
        print(f"✂️  Crop 이미지 저장됨: {crop_dir}")
        print(f"   총 {total_crops}개 이미지 저장")
    
    print("\n" + "=" * 60)
    print("✅ 예측 완료!")
    print("=" * 60)
    
    return results

def main():
    parser = argparse.ArgumentParser(description='YOLO Object Detection 모델 예측')
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='모델 경로 (예: runs/detect/exp1/weights/best.pt)'
    )
    parser.add_argument(
        '--source',
        type=str,
        required=True,
        help='입력 이미지 또는 폴더 경로'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='결과 저장 비활성화 (기본: 저장)'
    )
    parser.add_argument(
        '--no-crop',
        action='store_true',
        help='Crop 이미지 저장 비활성화 (기본: 저장)'
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        default='runs/predict_det',
        help='결과 저장 경로 (default: runs/predict_det)'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='신뢰도 임계값 (default: 0.25)'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='이미지 크기 (default: 640)'
    )
    
    args = parser.parse_args()
    
    # 모델 파일 확인
    if not os.path.exists(args.model):
        print(f"❌ 모델 파일을 찾을 수 없습니다: {args.model}")
        return
    
    # 소스 확인
    if not os.path.exists(args.source):
        print(f"❌ 입력 파일/폴더를 찾을 수 없습니다: {args.source}")
        return
    
    # 예측 실행
    predict(
        model_path=args.model,
        source=args.source,
        save=not args.no_save,
        save_crop=not args.no_crop,
        save_dir=args.save_dir,
        conf=args.conf,
        imgsz=args.imgsz
    )

if __name__ == '__main__':
    main()
