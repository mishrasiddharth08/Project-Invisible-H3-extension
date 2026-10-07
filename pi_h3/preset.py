"""Add the H3 still-worker and native-video presets without changing Forge core."""
from copy import copy, deepcopy
from pathlib import Path

from .assets import scan
from .config import (LABEL, PRESET, LEGACY_VIDEO_PRESET, OUTPUT_KEY, OUTPUTS,
                     STILL_OUTPUT, VIDEO_OUTPUT, models_root, settings)

_NATIVE_CHECKPOINT = None
_NATIVE_MODULES = []


def _physical_checkpoint(path):
    if not path:
        return None
    from modules import sd_models, shared
    wanted = str(Path(path).resolve()).casefold()
    for item in sd_models.checkpoints_list.values():
        filename = getattr(item, 'filename', None)
        if (filename and not getattr(item, '_pi_h3', False)
                and str(Path(filename).resolve()).casefold() == wanted):
            return item.short_title if getattr(shared.opts, 'sd_checkpoint_dropdown_use_short', False) else item.name
    return None


def _clone(template, name, section, defaults):
    result = {}
    for key, info in template.items():
        if key.startswith('sd_'):
            target = name + key[2:]
        elif key in ('forge_checkpoint_sd', 'forge_additional_modules_sd', 'forge_unet_storage_dtype_sd'):
            target = key[:-2] + name
        else:
            continue
        item = copy(info)
        item.default = deepcopy(info.default)
        if getattr(info, 'section', (None,))[0] == 'ui_sd':
            item.section = section
        if target in defaults:
            item.default = deepcopy(defaults[target])
        if target.endswith(('_width', '_height')):
            item.component_args = {'minimum': 64, 'maximum': 4096, 'step': 32}
        result[target] = item
    return result


def _native_roots():
    base = models_root()
    return [base / 'MiniMax-H3', base / 'text_encoders', base / 'text_encoder', base / 'VAE',
            *[Path(path) for path in settings()['model_roots']]]


def _native_modules(inventory, search_roots=None):
    """Choose only components the native loader can identify and safely use."""
    try:
        from forge_h3.contracts import H3Error
        from forge_h3.models import inspect_model
    except ImportError:
        return []
    candidates = {}
    seen = set()
    paths = [Path(path) for role in ('clip', 'vae') for path in inventory.get(role, [])]
    for root in _native_roots() if search_roots is None else search_roots:
        root = Path(root)
        if not root.is_dir():
            continue
        paths.extend(root.rglob('*.safetensors'))
    for path in paths:
        resolved = str(path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        try:
            item = inspect_model(path)
        except (H3Error, OSError, ValueError):
            continue
        if item is None or item.role not in ('text_encoder', 'video_vae', 'audio_vae'):
            continue
        if item.role == 'text_encoder' and item.quantization == 'nvfp4':
            continue
        candidates.setdefault(item.role, []).append(item)
    selected = []
    for role in ('text_encoder', 'video_vae', 'audio_vae'):
        choices = candidates.get(role, [])
        if choices:
            def priority(item):
                quantization = str(item.quantization).lower()
                if role != 'text_encoder':
                    return 0
                if quantization.startswith('int4') or 'w4a4' in quantization:
                    return 0
                if quantization.startswith('int8'):
                    return 1
                if quantization in ('plain', 'bf16'):
                    return 2
                return 3
            choices.sort(key=lambda item: (priority(item), str(item.path).casefold()))
            selected.append(str(choices[0].path))
    return selected


def native_defaults():
    global _NATIVE_CHECKPOINT, _NATIVE_MODULES
    if _NATIVE_CHECKPOINT is None:
        inventory = scan()
        _NATIVE_CHECKPOINT = _physical_checkpoint(inventory['dit'][0] if inventory['dit'] else None)
        _NATIVE_MODULES = _native_modules(inventory)
    return _NATIVE_CHECKPOINT, list(_NATIVE_MODULES)


def native_modules_valid(paths):
    from forge_h3.contracts import H3Error
    from forge_h3.models import inspect_model
    try:
        items = [inspect_model(path) for path in paths]
    except (H3Error, OSError, ValueError):
        return False
    roles = {item.role for item in items if item is not None}
    return ({'text_encoder', 'video_vae', 'audio_vae'}.issubset(roles)
            and all(item.quantization != 'nvfp4' for item in items
                    if item is not None and item.role == 'text_encoder'))


def install():
    global _NATIVE_CHECKPOINT, _NATIVE_MODULES
    from modules import shared
    from modules_forge import presets

    inventory = scan()
    _NATIVE_CHECKPOINT = _physical_checkpoint(inventory['dit'][0] if inventory['dit'] else None)
    _NATIVE_MODULES = _native_modules(inventory)
    template = {}
    presets.register(template)
    still = {
        f'{PRESET}_t2i_sampler': 'ER SDE', f'{PRESET}_i2i_sampler': 'ER SDE',
        f'{PRESET}_t2i_scheduler': 'Simple', f'{PRESET}_i2i_scheduler': 'Simple',
        f'{PRESET}_t2i_step': 50, f'{PRESET}_t2i_hr_step': 50, f'{PRESET}_i2i_step': 50,
        f'{PRESET}_t2i_cfg': 1.0, f'{PRESET}_t2i_hr_cfg': 1.0, f'{PRESET}_i2i_cfg': 1.0,
        f'{PRESET}_t2i_width': 1536, f'{PRESET}_i2i_width': 1536,
        f'{PRESET}_t2i_height': 1536, f'{PRESET}_i2i_height': 1536,
        'forge_checkpoint_' + PRESET: LABEL,
        'forge_additional_modules_' + PRESET: [],
    }
    ours = {}
    ours.update(_clone(template, PRESET, ('ui_h3', 'MINIMAX H3 STILL'), still))
    required = {'forge_checkpoint_' + PRESET, PRESET + '_t2i_step', PRESET + '_t2i_cfg'}
    if not required.issubset(ours):
        raise RuntimeError('Neo preset schema changed; H3 preset registration skipped')
    for key, info in ours.items():
        if key not in shared.opts.data_labels:
            shared.opts.add_option(key, info)
    if OUTPUT_KEY not in shared.opts.data_labels:
        shared.opts.add_option(OUTPUT_KEY, shared.OptionInfo(
            STILL_OUTPUT, 'H3 Output', component_args={'choices': list(OUTPUTS)},
            section=('forge_h3', 'MiniMax H3')))
    if getattr(shared.opts, 'forge_preset', None) == LEGACY_VIDEO_PRESET:
        shared.opts.set('forge_preset', PRESET)
        shared.opts.set(OUTPUT_KEY, VIDEO_OUTPUT)

    original = presets.PresetArch.choices
    if not getattr(original, '_pi_h3', False):
        def choices(*args, **kwargs):
            values = list(original(*args, **kwargs))
            if PRESET not in values:
                values.append(PRESET)
            return values
        choices._pi_h3 = True
        presets.PresetArch.choices = staticmethod(choices)
