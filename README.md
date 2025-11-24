# YOLO Project

YOLO 모델을 사용한 Classification과 Object Detection 학습/예측 프로젝트

## 폴더 구조

```
yolo_project/
├── configs/                    # 설정 파일
│   ├── cls_default.yaml        # 분류 기본 설정
│   ├── cls_basil.yaml          # 분류 바질 실험 설정
│   ├── det_default.yaml        # 탐지 기본 설정
│   ├── aug_light.yaml          # 가벼운 augmentation
│   ├── aug_heavy.yaml          # 강한 augmentation
│   └── experiment/             # 오버라이드된 설정 저장
│       ├── cls_experiment/     # 분류 실험 설정
│       │   ├── 1/
│       │   │   ├── cls_experiment1.yaml
│       │   │   └── cls_experiment1.log
│       │   └── 2/
│       │       └── ...
│       └── det_experiment/     # 탐지 실험 설정
│           ├── 1/
│           │   ├── det_experiment1.yaml
│           │   └── det_experiment1.log
│           └── 2/
│               └── ...
│
├── utils/                      # 유틸리티
│   └── config_loader.py        # YAML 설정 로더
│
├── runs/                       # 학습/예측 결과 저장
│   ├── classify/               # 분류 학습 결과
│   ├── detect/                 # 탐지 학습 결과
│   ├── predict_cls/            # 분류 예측 결과
│   └── predict_det/            # 탐지 예측 결과
│
├── train_cls.py                # 분류 학습 스크립트
├── train_det.py                # 탐지 학습 스크립트
├── predict_cls.py              # 분류 예측 스크립트
├── predict_det.py              # 탐지 예측 스크립트
│
├── yolo11n-cls.pt              # 분류 모델 (사전학습)
├── yolo11n.pt                  # 탐지 모델 (사전학습)
│
├── CHANGELOG.md                # 변경 기록
└── README.md                   # 프로젝트 설명
```

## 사용 방법

### 1. Classification (분류)

#### 학습
```bash
# 기본 설정으로 학습
python train_cls.py

# 특정 설정 파일로 학습
python train_cls.py --config configs/cls_basil.yaml

# 설정 오버라이드
python train_cls.py --epochs 100 --batch 16

# Augmentation 설정 적용
python train_cls.py --config configs/cls_default.yaml --aug configs/aug_heavy.yaml
```

#### 예측
```bash
# 단일 이미지 예측
python predict_cls.py --model runs/classify/exp1/weights/best.pt --source image.jpg

# 폴더 예측 + 결과 저장
python predict_cls.py --model runs/classify/exp1/weights/best.pt --source images/ --save
```

### 2. Object Detection (탐지)

#### 학습
```bash
# 기본 설정으로 학습
python train_det.py

# 설정 오버라이드
python train_det.py --epochs 100 --batch 8 --imgsz 640
```

#### 예측
```bash
# 기본: 바운딩박스 + crop 둘 다 저장
python predict_det.py --model runs/detect/exp1/weights/best.pt --source image.jpg

# 저장 비활성화
python predict_det.py --model runs/detect/exp1/weights/best.pt --source image.jpg --no-save

# crop만 비활성화
python predict_det.py --model runs/detect/exp1/weights/best.pt --source image.jpg --no-crop
```

## 설정 오버라이드 옵션

| 옵션 | 설명 | 예시 |
|------|------|------|
| `--config` | 설정 파일 경로 | `--config configs/cls_basil.yaml` |
| `--epochs` | 학습 에포크 수 | `--epochs 100` |
| `--batch` | 배치 크기 | `--batch 16` |
| `--imgsz` | 이미지 크기 | `--imgsz 640` |
| `--device` | 학습 디바이스 | `--device 0` 또는 `--device cpu` |
| `--augment` | 데이터 증강 활성화 | `--augment` |
| `--name` | 실험 이름 접두사 | `--name my_exp` |
| `--dataset` | 데이터셋 경로 | `--dataset path/to/data` |
| `--aug` | Augmentation 설정 파일 | `--aug configs/aug_heavy.yaml` |

## 설정 파일 자동 생성

커맨드라인에서 설정을 오버라이드하면 자동으로 새 설정 파일과 로그가 생성됩니다:

```bash
python train_cls.py --epochs 100 --batch 16
```

생성되는 파일:
- `configs/experiment/cls_experiment/1/cls_experiment1.yaml` - 변경된 설정
- `configs/experiment/cls_experiment/1/cls_experiment1.log` - 변경 로그

## Augmentation 설정

`aug_light.yaml`, `aug_heavy.yaml` 등의 augmentation 전용 설정 파일을 만들어서 사용할 수 있습니다.

```bash
# 가벼운 augmentation
python train_cls.py --aug configs/aug_light.yaml

# 강한 augmentation
python train_det.py --aug configs/aug_heavy.yaml
```

### 주요 Augmentation 옵션

| 옵션 | 설명 | 범위 |
|------|------|------|
| `hsv_h` | 색상 변환 | 0.0-1.0 |
| `hsv_s` | 채도 변환 | 0.0-1.0 |
| `hsv_v` | 명도 변환 | 0.0-1.0 |
| `degrees` | 회전 각도 | -180~180 |
| `translate` | 이동 | 0.0-1.0 |
| `scale` | 스케일 | 0.0-1.0 |
| `flipud` | 상하 반전 확률 | 0.0-1.0 |
| `fliplr` | 좌우 반전 확률 | 0.0-1.0 |
| `erasing` | 랜덤 지우기 | 0.0-1.0 |
| `auto_augment` | 자동 증강 | randaugment, autoaugment, augmix |

## 예측 결과 저장

### Classification
```
runs/predict_cls/
└── predict/
    └── image.jpg
```

### Detection
```
runs/predict_det/
├── predict/                       # 첫 번째 실행
│   ├── image.jpg                  # 바운딩박스 그려진 이미지
│   └── crop/                      # Crop 이미지
│       ├── image_person_1_95.jpg
│       └── image_car_2_87.jpg
├── predict2/                      # 두 번째 실행
│   ├── image.jpg
│   └── crop/
│       └── ...
└── predict3/                      # 세 번째 실행
    └── ...
```

**파일명 형식**: `{원본이름}_{클래스명}_{번호}_{신뢰도}.jpg`
