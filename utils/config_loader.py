import yaml
from pathlib import Path
from typing import Dict, Any

class Config:
    """설정 관리 클래스"""
    
    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict
    
    def __getattr__(self, key):
        if key in self._config:
            value = self._config[key]
            # 중첩된 딕셔너리도 Config 객체로 변환
            if isinstance(value, dict):
                return Config(value)
            return value
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")
    
    def get(self, key, default=None):
        return self._config.get(key, default)
    
    def to_dict(self):
        return self._config

def load_config(config_path: str) -> Config:
    """YAML 설정 파일 로드"""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    return Config(config_dict)

def save_config(config: Config, save_path: str):
    """설정을 YAML 파일로 저장"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, allow_unicode=True)
