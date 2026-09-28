import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABEL = 'PROJECT INVISIBLE — MiniMax H3 Still'
PRESET = 'H3'


def settings():
    defaults = {'model_roots': [], 'preview_interval': 1.0, 'keep_loaded': False}
    path = ROOT / 'config.json'
    if path.exists():
        defaults.update(json.loads(path.read_text(encoding='utf-8')))
    return defaults


def models_root():
    from modules import paths
    return Path(paths.models_path)


def roots():
    base = models_root()
    return [base / 'MiniMax-H3', base / 'diffusion_models', base / 'Stable-diffusion',
            base / 'text_encoders', base / 'text_encoder', base / 'VAE', base / 'Lora',
            *[Path(p) for p in settings()['model_roots']]]
