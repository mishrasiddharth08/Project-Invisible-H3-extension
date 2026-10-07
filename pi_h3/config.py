import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABEL = 'PROJECT INVISIBLE — MiniMax H3 Still'
PRESET = 'H3'
LEGACY_VIDEO_PRESET = 'H3 Video'
# Import compatibility only; this value is never registered as a preset choice.
VIDEO_PRESET = LEGACY_VIDEO_PRESET
OUTPUT_KEY = 'pi_h3_output'
STILL_OUTPUT = 'Still image'
VIDEO_OUTPUT = 'Video'
OUTPUTS = (STILL_OUTPUT, VIDEO_OUTPUT)


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
