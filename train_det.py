from ultralytics import YOLO
from pathlib import Path
import torch
import os
import argparse
from datetime import datetime
from utils.config_loader import load_config, save_config

def get_next_experiment_name(base_name, project_dir='runs/detect'):
    """다음 실험 번호를 찾아 이름을 반환"""
    i = 1
    while True:
        exp_name = f"{base_name}{i}"
        if not os.path.exists(os.path.join(project_dir, exp_name)):
            return exp_name
        i += 1

def get_next_config_number(config_dir='configs/experiment/det_experiment'):
    """다음 설정 파일 번호를 찾아 반환"""
    i = 1
    while True:
        exp_folder = os.path.join(config_dir, str(i))
        if not os.path.exists(exp_folder):
            return i
        i += 1

def train_model(config):
    """설정을 기반으로 모델 학습"""
    
    # GPU 사용 가능 여부 확인
    if not torch.cuda.is_available() and config.training.device != 'cpu':
        print("⚠️  경고: CUDA를 사용할 수 없습니다. CPU로 학습을 진행합니다.")
        config._config['training']['device'] = 'cpu'
    
    print("=" * 60)
    print("🚀 YOLO Object Detection 모델 학습 시작")
    print("=" * 60)
    
    # 모델 로드
    print(f"\n📦 모델 로드: {config.model.name}")
    model = YOLO(config.model.name)
    
    # 다음 실험 이름 생성
    next_experiment_name = get_next_experiment_name(
        config.experiment.base_name,
        config.experiment.project_dir
    )
    
    print(f"📁 실험 이름: {next_experiment_name}")
    print(f"💾 저장 경로: {config.experiment.project_dir}/{next_experiment_name}")
    print(f"📊 데이터셋: {config.dataset.path}")
    print(f"🔢 Epochs: {config.training.epochs}, Batch: {config.training.batch}")
    print(f"🖼️  이미지 크기: {config.dataset.imgsz}")
    print()
    
    # 학습 파라미터 설정
    train_params = {
        'data': config.dataset.path,
        'epochs': config.training.epochs,
        'imgsz': config.dataset.imgsz,
        'batch': config.training.batch,
        'workers': config.training.workers,
        'device': config.training.device,
        'seed': config.training.seed,
        'cache': config.training.cache,
        'augment': config.training.augment,
        'verbose': config.training.verbose,
        'name': next_experiment_name,
        'project': config.experiment.project_dir
    }

    # augmentation 설정이 있으면 추가
    if config.get('augmentation'):
        aug = config.augmentation
        aug_params = {
            'hsv_h': aug.get('hsv_h', 0.015),
            'hsv_s': aug.get('hsv_s', 0.7),
            'hsv_v': aug.get('hsv_v', 0.4),
            'degrees': aug.get('degrees', 0.0),
            'translate': aug.get('translate', 0.1),
            'scale': aug.get('scale', 0.5),
            'shear': aug.get('shear', 0.0),
            'perspective': aug.get('perspective', 0.0),
            'flipud': aug.get('flipud', 0.0),
            'fliplr': aug.get('fliplr', 0.5),
            'mosaic': aug.get('mosaic', 1.0),
            'mixup': aug.get('mixup', 0.0)
        }
        train_params.update(aug_params)
        print("🎨 Augmentation 설정 적용됨")

    # 학습 시작
    results = model.train(**train_params)
    
    # 학습에 사용된 설정 저장
    save_path = Path(config.experiment.project_dir) / next_experiment_name / "config.yaml"
    save_config(config, str(save_path))
    print(f"\n💾 설정 파일 저장됨: {save_path}")
    
    print("\n" + "=" * 60)
    print("✅ 학습 완료!")
    print("=" * 60)
    
    return results

def main():
    parser = argparse.ArgumentParser(description='YOLO Object Detection 모델 학습')
    
    # 기본 설정
    parser.add_argument(
        '--config',
        type=str,
        default='configs/det_default.yaml',
        help='설정 파일 경로 (default: configs/det_default.yaml)'
    )
    
    # 설정 오버라이드 옵션들
    parser.add_argument('--epochs', type=int, help='학습 에포크 수')
    parser.add_argument('--batch', type=int, help='배치 크기')
    parser.add_argument('--imgsz', type=int, help='이미지 크기')
    parser.add_argument('--device', help='학습 디바이스 (0, 1, cpu 등)')
    parser.add_argument('--augment', action='store_true', help='데이터 증강 활성화')
    parser.add_argument('--name', type=str, help='실험 이름 접두사')
    parser.add_argument('--dataset', type=str, help='데이터셋 경로')
    parser.add_argument('--aug', type=str, help='Augmentation 설정 파일 경로')
    
    args = parser.parse_args()
    
    # 설정 파일 로드
    print(f"📄 설정 파일 로드: {args.config}")
    config = load_config(args.config)

    # 커맨드라인 인자로 설정 오버라이드
    overrides = []

    if args.epochs is not None:
        old_val = config._config['training']['epochs']
        config._config['training']['epochs'] = args.epochs
        overrides.append(f"epochs: {old_val} → {args.epochs}")
    if args.batch is not None:
        old_val = config._config['training']['batch']
        config._config['training']['batch'] = args.batch
        overrides.append(f"batch: {old_val} → {args.batch}")
    if args.imgsz is not None:
        old_val = config._config['dataset']['imgsz']
        config._config['dataset']['imgsz'] = args.imgsz
        overrides.append(f"imgsz: {old_val} → {args.imgsz}")
    if args.device is not None:
        old_val = config._config['training']['device']
        config._config['training']['device'] = args.device
        overrides.append(f"device: {old_val} → {args.device}")
    if args.augment:
        old_val = config._config['training']['augment']
        config._config['training']['augment'] = True
        overrides.append(f"augment: {old_val} → True")
    if args.name is not None:
        old_val = config._config['experiment']['base_name']
        config._config['experiment']['base_name'] = args.name
        overrides.append(f"base_name: {old_val} → {args.name}")
    if args.dataset is not None:
        old_val = config._config['dataset']['path']
        config._config['dataset']['path'] = args.dataset
        overrides.append(f"dataset: {old_val} → {args.dataset}")

    # augmentation 설정 파일 로드
    if args.aug is not None:
        aug_config = load_config(args.aug)
        if aug_config.get('augmentation'):
            config._config['augmentation'] = aug_config._config['augmentation']
            overrides.append(f"augmentation: {args.aug} 적용")
            print(f"🎨 Augmentation 설정 로드: {args.aug}")

    # 오버라이드가 있으면 새 yaml 파일과 로그 파일 생성
    if overrides:
        config_num = get_next_config_number()
        exp_dir = Path("configs") / "experiment" / "det_experiment" / str(config_num)
        exp_dir.mkdir(parents=True, exist_ok=True)
        new_config_path = exp_dir / f"det_experiment{config_num}.yaml"
        new_log_path = exp_dir / f"det_experiment{config_num}.log"

        # 설정 파일 저장
        save_config(config, str(new_config_path))

        # 로그 파일 저장
        with open(new_log_path, 'w', encoding='utf-8') as f:
            f.write(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"기본 설정: {args.config}\n")
            f.write(f"생성된 설정: {new_config_path}\n\n")
            f.write("변경된 설정:\n")
            for override in overrides:
                f.write(f"  • {override}\n")

        print(f"\n📝 새 설정 파일 생성: {new_config_path}")
        print(f"📋 로그 파일 생성: {new_log_path}")
        print("🔧 변경된 설정:")
        for override in overrides:
            print(f"   • {override}")

    # 학습 실행
    train_model(config)

if __name__ == '__main__':
    main()
