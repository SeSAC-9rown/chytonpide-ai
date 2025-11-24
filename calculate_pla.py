import cv2
import numpy as np
import argparse
import json
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime


def calculate_pla_workflow(image_path, model_path, output_dir=None):
    """
    PLA(엽면적) 계산 함수

    Args:
        image_path: 입력 이미지 경로
        model_path: YOLO 모델 경로
        output_dir: 결과 저장 디렉토리 (기본값: runs/pla/predict)

    Returns:
        dict: 계산 결과 (각 식물의 PLA 값 포함)
    """

    # 기본 출력 디렉토리 설정
    if output_dir is None:
        output_dir = Path("runs/pla/predict")
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 결과 저장 디렉토리 (실행마다 새로운 폴더 생성)
    run_count = 1
    while (output_dir / f"predict{run_count if run_count > 1 else ''}").exists():
        run_count += 1

    result_dir = output_dir / f"predict{run_count if run_count > 1 else ''}"
    result_dir.mkdir(parents=True, exist_ok=True)
    crop_dir = result_dir / "crop"
    crop_dir.mkdir(parents=True, exist_ok=True)
    scale_dir = result_dir / "scale"
    scale_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = result_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    # 1. 이미지 로드
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[Error] 이미지를 불러올 수 없습니다: {image_path}")
        return None

    original_h, original_w = img.shape[:2]
    image_name = Path(image_path).stem

    # 노이즈 제거를 위한 모폴로지 연산 (공통 kernel 정의)
    kernel = np.ones((3, 3), np.uint8)

    # ---------------------------------------------------------
    # STEP 1: YOLO로 scale 마커 검출
    # ---------------------------------------------------------
    model = YOLO(str(model_path))
    results = model(img)

    mm_per_pixel = 0
    scale_marker_info = None

    # scale 클래스 찾기
    for result in results:
        boxes = result.boxes
        # 모델의 클래스명 확인
        class_names = result.names  # {0: 'class_name', ...}

        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = class_names.get(cls_id, "unknown")

            # "scale" 클래스 찾기
            if cls_name.lower() == "scale":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])

                # scale 마커의 중심과 크기
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                width = x2 - x1
                height = y2 - y1
                diameter_pixel = max(width, height)  # 더 긴 쪽을 지름으로 사용

                # 스케일 계산 (실제 지름 16mm / 픽셀 지름)
                real_diameter_mm = 16.0
                mm_per_pixel = real_diameter_mm / diameter_pixel

                print(f"[Info] Scale 마커 감지됨: 지름 {diameter_pixel:.2f}px, 신뢰도 {confidence:.2%}")
                print(f"[Scale] 1 Pixel = {mm_per_pixel:.4f} mm")

                # scale 마커 크롭
                margin = int(diameter_pixel * 0.2) + 10
                crop_x1 = max(0, int(cx - diameter_pixel / 2 - margin))
                crop_y1 = max(0, int(cy - diameter_pixel / 2 - margin))
                crop_x2 = min(original_w, int(cx + diameter_pixel / 2 + margin))
                crop_y2 = min(original_h, int(cy + diameter_pixel / 2 + margin))

                scale_crop = img[crop_y1:crop_y2, crop_x1:crop_x2]
                scale_marker_info = {
                    "class_name": "scale",
                    "confidence": float(confidence),
                    "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "center_x": float(cx),
                    "center_y": float(cy),
                    "diameter_pixel": float(diameter_pixel),
                    "mm_per_pixel": float(mm_per_pixel),
                    "crop_box": {"x1": crop_x1, "y1": crop_y1, "x2": crop_x2, "y2": crop_y2}
                }

                # scale 마커 크롭 이미지 저장
                scale_marker_path = scale_dir / f"{image_name}_scale_marker.jpg"
                cv2.imwrite(str(scale_marker_path), scale_crop)
                print(f"[Save] Scale 마커 저장: {scale_marker_path}")

                break

        if mm_per_pixel > 0:
            break

    if mm_per_pixel == 0:
        print("[Error] Scale 마커를 찾을 수 없습니다. 계산을 중단합니다.")
        print(f"[Info] 사용 가능한 클래스: {class_names if 'class_names' in locals() else 'N/A'}")
        return None

    # 결과 저장용 리스트
    pla_results = []
    plant_count = 0

    for result in results:
        boxes = result.boxes
        class_names = result.names

        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = class_names.get(cls_id, "unknown")

            # scale 마커는 건너뛰고 식물만 처리
            if cls_name.lower() == "scale":
                continue

            plant_count += 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])

            # 이미지 Crop
            plant_crop = img[y1:y2, x1:x2]

            # ---------------------------------------------------------
            # STEP 3: PLA 계산 (초록색 추출)
            # ---------------------------------------------------------
            crop_hsv = cv2.cvtColor(plant_crop, cv2.COLOR_BGR2HSV)

            # 초록색 범위 설정
            lower_green = np.array([35, 40, 40])
            upper_green = np.array([85, 255, 255])

            green_mask = cv2.inRange(crop_hsv, lower_green, upper_green)

            # 노이즈 제거
            green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)

            # 초록색 픽셀 수 계산
            green_pixel_count = cv2.countNonZero(green_mask)

            # 면적 계산: 픽셀 수 * (mm/pixel)^2
            area_mm2 = green_pixel_count * (mm_per_pixel ** 2)
            area_cm2 = area_mm2 / 100.0

            # 결과 출력
            print(f"\n--- 식물 #{plant_count} (Box: {x1},{y1} ~ {x2},{y2}) ---")
            print(f"신뢰도: {confidence:.2%}")
            print(f"녹색 픽셀 수: {green_pixel_count}")
            print(f"엽면적(PLA): {area_mm2:.2f} mm² ({area_cm2:.2f} cm²)")

            # 결과 저장
            result_data = {
                "plant_id": plant_count,
                "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "confidence": confidence,
                "green_pixels": green_pixel_count,
                "pla_mm2": area_mm2,
                "pla_cm2": area_cm2
            }
            pla_results.append(result_data)

            # Crop 이미지 저장
            crop_filename = f"{image_name}_plant_{plant_count}_{confidence:.0%}.jpg"
            crop_path = crop_dir / crop_filename
            cv2.imwrite(str(crop_path), plant_crop)

            # HSV 마스크 이미지 저장 (디버그용)
            # 원본 crop 이미지
            debug_crop_path = debug_dir / f"{image_name}_plant_{plant_count}_crop.jpg"
            cv2.imwrite(str(debug_crop_path), plant_crop)

            # 초록색 마스크 이미지 (흰색 = 초록색 감지)
            debug_mask_path = debug_dir / f"{image_name}_plant_{plant_count}_green_mask.jpg"
            cv2.imwrite(str(debug_mask_path), green_mask)

            # 마스크를 원본 이미지에 오버레이 (초록색 감지 영역 시각화)
            mask_overlay = plant_crop.copy()
            mask_overlay[green_mask > 0] = [0, 255, 0]  # 초록색 감지 영역을 초록색으로 표시
            alpha = 0.5
            debug_overlay_path = debug_dir / f"{image_name}_plant_{plant_count}_overlay.jpg"
            overlay_result = cv2.addWeighted(plant_crop, 1 - alpha, mask_overlay, alpha, 0)
            cv2.imwrite(str(debug_overlay_path), overlay_result)

    # ---------------------------------------------------------
    # 결과 JSON 저장
    # ---------------------------------------------------------
    summary = {
        "timestamp": datetime.now().isoformat(),
        "image": str(image_path),
        "model": str(model_path),
        "scale_marker": scale_marker_info,
        "scale": {
            "mm_per_pixel": mm_per_pixel,
            "sticker_diameter_mm": 16.0
        },
        "total_plants": plant_count,
        "plants": pla_results
    }

    # 요약 통계
    if pla_results:
        pla_values = [p["pla_cm2"] for p in pla_results]
        summary["statistics"] = {
            "total_pla_cm2": sum(pla_values),
            "average_pla_cm2": sum(pla_values) / len(pla_values),
            "min_pla_cm2": min(pla_values),
            "max_pla_cm2": max(pla_values)
        }

    # JSON 파일 저장
    json_path = result_dir / f"{image_name}_results.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n[Success] 결과 저장 완료!")
    print(f"저장 위치: {result_dir}")
    print(f"결과 파일: {json_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="YOLO를 사용하여 식물의 엽면적(PLA) 계산"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="입력 이미지 경로 (예: images/plant.jpg)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="runs/detect/det_exp1/weights/best.pt",
        help="YOLO 모델 경로 (기본값: runs/detect/det_exp1/weights/best.pt)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="결과 저장 디렉토리 (기본값: runs/pla/predict)"
    )

    args = parser.parse_args()

    # 경로 존재 확인
    image_path = Path(args.source)
    model_path = Path(args.model)

    if not image_path.exists():
        print(f"[Error] 이미지를 찾을 수 없습니다: {image_path}")
        return

    if not model_path.exists():
        print(f"[Error] 모델을 찾을 수 없습니다: {model_path}")
        return

    # PLA 계산 실행
    calculate_pla_workflow(image_path, model_path, args.output)


if __name__ == "__main__":
    main()
