from ultralytics import YOLO
from pathlib import Path
import argparse
import os

def predict(model_path, source, save=False, save_dir='runs/predict', conf=0.25, imgsz=224):
    """모델로 예측 수행"""
    
    # 모델 로드
    print("=" * 60)
    print("🔍 YOLO 분류 모델 예측")
    print("=" * 60)
    
    print(f"\n📦 모델 로드: {model_path}")
    model = YOLO(model_path)
    
    print(f"🖼️  입력: {source}")
    print(f"📊 신뢰도 임계값: {conf}")
    print(f"📐 이미지 크기: {imgsz}")
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
    
    # 결과 출력
    print("\n" + "=" * 60)
    print("📋 예측 결과")
    print("=" * 60)
    
    for i, result in enumerate(results):
        if hasattr(result, 'probs') and result.probs is not None:
            probs = result.probs
            top1_idx = probs.top1
            top1_conf = probs.top1conf.item()
            top5_idx = probs.top5
            top5_conf = probs.top5conf.tolist()
            
            # 클래스 이름 가져오기
            names = result.names
            
            print(f"\n🖼️  이미지 {i+1}: {Path(result.path).name}")
            print(f"   Top-1: {names[top1_idx]} ({top1_conf*100:.2f}%)")
            print(f"   Top-5:")
            for idx, conf_val in zip(top5_idx, top5_conf):
                print(f"      • {names[idx]}: {conf_val*100:.2f}%")
    
    if save:
        print(f"\n💾 결과 저장됨: {save_dir}")
    
    print("\n" + "=" * 60)
    print("✅ 예측 완료!")
    print("=" * 60)
    
    return results

def main():
    parser = argparse.ArgumentParser(description='YOLO 분류 모델 예측')
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='모델 경로 (예: runs/classify/exp1/weights/best.pt)'
    )
    parser.add_argument(
        '--source',
        type=str,
        required=True,
        help='입력 이미지 또는 폴더 경로'
    )
    parser.add_argument(
        '--save',
        action='store_true',
        help='결과 저장'
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        default='runs/predict',
        help='결과 저장 경로 (default: runs/predict)'
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
        default=224,
        help='이미지 크기 (default: 224)'
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
        save=args.save,
        save_dir=args.save_dir,
        conf=args.conf,
        imgsz=args.imgsz
    )

if __name__ == '__main__':
    main()
