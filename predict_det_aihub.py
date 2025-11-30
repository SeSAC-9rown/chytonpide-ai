from ultralytics import YOLO
from pathlib import Path
import argparse
import os
import cv2
import torch
import torch.nn as nn
from ultralytics.nn.modules import conv

# ==============================================================
# [긴급 처방] Triple_Conv 모듈 강제 주입
# 모델이 찾고 있는 'Triple_Conv'라는 부품을 여기서 즉석에서 만들어줍니다.
# ==============================================================
class Triple_Conv(nn.Module):
    """
    YOLO Custom Module: Triple_Conv
    (Conv -> Conv -> Conv 3단 연결 구조)
    """
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        # c1: 입력채널, c2: 출력채널, k: 커널크기
        self.cv1 = conv.Conv(c1, c2, k, s, p, g, d, act)
        self.cv2 = conv.Conv(c2, c2, k, s, p, g, d, act)
        self.cv3 = conv.Conv(c2, c2, k, s, p, g, d, act)

    def forward(self, x):
        return self.cv3(self.cv2(self.cv1(x)))

# ultralytics 패키지가 이 클래스를 자신의 식구로 착각하게 만듭니다.
setattr(conv, 'Triple_Conv', Triple_Conv)
# ==============================================================


def predict(model_path, source, save=True, save_crop=True, save_dir='runs/predict_det', conf=0.25, imgsz=640):
    """모델로 예측 수행"""

    # 모델 로드
    print("=" * 60)
    print("🔍 YOLO Object Detection 모델 예측")
    print("=" * 60)

    print(f"\n📦 모델 로드: {model_path}")
    # 위에서 Triple_Conv를 등록했으므로 이제 에러 없이 로드됩니다.
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

    # YOLO가 생성한 저장 폴더 찾기
    if save_crop and len(results) > 0:
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
                img = cv2.imread(result.path)
                img_name = Path(result.path).stem
                
                for j, box in enumerate(boxes):
                    cls_id = int(box.cls.item())
                    cls_name = names[cls_id]
                    conf_val = box.conf.item()
                    
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    
                    # 좌표 유효성 검사 (이미지 범위 벗어남 방지)
                    h, w, _ = img.shape
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)

                    crop_img = img[y1:y2, x1:x2]
                    
                    if crop_img.size > 0:
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
    
    parser.add_argument('--model', type=str, required=True, help='모델 경로')
    parser.add_argument('--source', type=str, required=True, help='입력 이미지/폴더')
    parser.add_argument('--no-save', action='store_true', help='저장 안함')
    parser.add_argument('--no-crop', action='store_true', help='Crop 안함')
    parser.add_argument('--save-dir', type=str, default='runs/predict_det', help='저장 경로')
    parser.add_argument('--conf', type=float, default=0.25, help='신뢰도')
    parser.add_argument('--imgsz', type=int, default=640, help='이미지 크기')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"❌ 모델 파일을 찾을 수 없습니다: {args.model}")
        return
    
    if not os.path.exists(args.source):
        print(f"❌ 입력 파일/폴더를 찾을 수 없습니다: {args.source}")
        return
    
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