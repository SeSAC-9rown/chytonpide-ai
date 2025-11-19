from pathlib import Path
from PIL import Image, ImageOps
import random
import shutil

# --- 설정 ---
# 스크린샷에 보이는 폴더 경로 (healthy, diseased 폴더가 들어있는 곳)
SRC_ROOT = Path(r"Folder_Path_Here")

# 결과가 저장될 폴더
DST_ROOT = Path(r"Folder_Path_Here")

TARGET_SIZE = (384, 384)
VAL_SPLIT = 0.2
RANDOM_SEED = 42

# ★ 무시할 폴더 이름 목록 (여기에 포함되면 처리 안 함) ★
IGNORE_DIRS = {".vscode", "dataset", ".git", "__pycache__"}
# ------------

random.seed(RANDOM_SEED)

def resize_with_padding(img, target_size):
    img.thumbnail(target_size, Image.BICUBIC)
    delta_w = target_size[0] - img.size[0]
    delta_h = target_size[1] - img.size[1]
    padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
    return ImageOps.expand(img, padding, fill=(0, 0, 0))

print(f"작업 시작: {SRC_ROOT} -> {DST_ROOT}")

if not SRC_ROOT.exists():
    print(f"오류: 원본 폴더가 없습니다: {SRC_ROOT}")
else:
    for class_dir in SRC_ROOT.iterdir():
        # 1. 폴더가 아니면 건너뜀
        if not class_dir.is_dir():
            continue
            
        # 2. [핵심] 무시할 폴더 리스트에 있거나, 이름이 점(.)으로 시작하면 건너뜀 (.vscode 등 방지)
        if class_dir.name in IGNORE_DIRS or class_dir.name.startswith("."):
            print(f"  [Skip] 시스템 폴더 건너뜀: {class_dir.name}")
            continue
            
        # 3. 'healthy', 'diseased' 같은 실제 데이터 폴더만 처리
        print(f"\nProcessing class: {class_dir.name}")

        image_paths = []
        for img_path in class_dir.iterdir():
            if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                image_paths.append(img_path)
        
        if not image_paths:
            print(f"  이미지가 없어 건너뜀 (혹은 이미지가 없는 폴더입니다)")
            continue

        # 8:2 분할
        random.shuffle(image_paths)
        split_idx = int(len(image_paths) * (1 - VAL_SPLIT))
        train_paths = image_paths[:split_idx]
        val_paths = image_paths[split_idx:]
        
        print(f"  Total: {len(image_paths)} -> Train: {len(train_paths)}, Val: {len(val_paths)}")

        # 처리 및 저장 함수
        def process_and_save(paths, split_name):
            save_dir = DST_ROOT / split_name / class_dir.name
            save_dir.mkdir(parents=True, exist_ok=True)

            for img_path in paths:
                try:
                    with Image.open(img_path) as img:
                        img = img.convert("RGB")
                        new_img = resize_with_padding(img, TARGET_SIZE)
                        dst_path = save_dir / img_path.name
                        new_img.save(dst_path, quality=95)
                except Exception as e:
                    print(f"  Error: {img_path.name} - {e}")

        process_and_save(train_paths, "train")
        process_and_save(val_paths, "val")

    print("\n--- 모든 작업 완료 ---")
    print(f"저장 위치: {DST_ROOT}") 