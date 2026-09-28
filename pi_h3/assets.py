"""Read tensor headers before accepting a component; never load unknown pickle files."""
import json
import struct
from pathlib import Path
from .config import roots


def header(path):
    path = Path(path)
    if path.suffix.lower() != '.safetensors':
        raise ValueError('H3 accepts .safetensors model files only.')
    with path.open('rb') as stream:
        size = stream.read(8)
        if len(size) != 8:
            raise ValueError('Incomplete model: ' + str(path))
        n = struct.unpack('<Q', size)[0]
        if not 2 <= n <= 64 * 1024 * 1024:
            raise ValueError('Invalid model header: ' + str(path))
        raw = stream.read(n)
        if len(raw) != n:
            raise ValueError('Incomplete model header: ' + str(path))
    try:
        data = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError('Invalid model header: ' + str(path)) from error
    if not isinstance(data, dict):
        raise ValueError('Invalid model header: ' + str(path))
    tensors = {k: v for k, v in data.items() if k != '__metadata__'}
    if not tensors:
        raise ValueError('Empty model: ' + str(path))
    spans = []
    for name, value in tensors.items():
        if not isinstance(name, str) or not isinstance(value, dict):
            raise ValueError('Invalid tensor header: ' + str(path))
        offsets = value.get('data_offsets')
        shape = value.get('shape')
        if (not isinstance(offsets, list) or len(offsets) != 2 or
                any(type(x) is not int for x in offsets) or offsets[0] < 0 or offsets[1] < offsets[0] or
                not isinstance(shape, list) or any(type(x) is not int or x < 0 for x in shape) or
                not isinstance(value.get('dtype'), str)):
            raise ValueError('Invalid tensor header: ' + str(path))
        spans.append(tuple(offsets))
    cursor = 0
    for start, end in sorted(spans):
        if start != cursor:
            raise ValueError('Invalid tensor offsets: ' + str(path))
        cursor = end
    if path.stat().st_size < 8 + n + cursor:
        raise ValueError('Download is incomplete: ' + str(path))
    return tensors


def classify(path):
    name = Path(path).name.lower().replace('-', '_')
    if 'minimax' not in name or 'h3' not in name:
        return None
    if 'video_vae' in name:
        return 'vae'
    if 'qwen3vl32b' in name.replace('_', ''):
        return 'clip'
    if 'turbo' in name or 'lora' in name:
        return 'lora'
    if ('fl2va' in name or 'ref2va' in name) and 'adapter' not in name:
        return 'dit'
    return None


def scan(search_roots=None):
    found = {k: [] for k in ('dit', 'clip', 'vae', 'lora')}
    seen = set()
    for root in search_roots if search_roots is not None else roots():
        root = Path(root)
        if not root.is_dir():
            continue
        for path in root.rglob('*.safetensors'):
            value = str(path.resolve())
            role = classify(path)
            if role and value not in seen:
                seen.add(value)
                found[role].append(value)
    return {k: sorted(v) for k, v in found.items()}


def resolve(selection, role, inventory):
    if selection and selection not in ('Auto', '(none)'):
        path = str(Path(selection).resolve())
        if path not in inventory[role]:
            raise ValueError(f'Selected {role} is missing or outside the H3 model folders: {path}')
    else:
        choices = inventory[role]
        if not choices:
            raise ValueError(f'H3 {role} is missing. Open H3 controls → Models for download links and folder instructions.')
        # Prefer community reference formats; user selections always win.
        path = sorted(choices, key=lambda p: ('nvfp4_awq' not in p if role == 'clip' else 'int8_convrot' not in p,
                                              'pruned' not in p, 'fl2va' not in p, p))[0]
    tensors = header(path)
    required = {'dit': ('video_patch_proj.weight', 'audio_patch_proj.weight'),
                'clip': ('visual.deepstack_merger_list.0.norm.weight', 'model.layers.49.self_attn.q_proj.weight'),
                'vae': ('decoder.transformer_blocks.0.scale1', 'encoder.down.5.block.0.conv1.weight')}
    if not all(any(key.endswith(suffix) for key in tensors) for suffix in required.get(role, ())):
        raise ValueError(f'Incompatible H3 {role} tensor structure: {path}')
    return path


def fingerprint(paths):
    return tuple((str(p), Path(p).stat().st_size, Path(p).stat().st_mtime_ns) for p in paths)
