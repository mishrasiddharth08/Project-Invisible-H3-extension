"""Dedicated same-interpreter worker. Never import Comfy into the Forge process."""
import base64
import contextlib
import io
import json
import logging
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'vendor' / 'python'), str(ROOT / 'vendor' / 'ComfyUI'), str(ROOT)]
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
WIRE = sys.stdout
sys.stdout = sys.stderr
logging.basicConfig(level=logging.INFO, stream=sys.stderr, format='%(message)s')
mode = sys.argv[1] if len(sys.argv) > 1 else 'auto'
sys.argv = [sys.argv[0]]


def emit(kind, **values):
    WIRE.write(json.dumps({'type': kind, **values}) + '\n')
    WIRE.flush()


def startup():
    import comfy.options
    comfy.options.enable_args_parsing()
    sys.argv += ['--disable-auto-launch', '--disable-dynamic-vram']
    if mode == 'cpu':
        sys.argv.append('--cpu')
    else:
        # VRAM-tier auto mode: full speed on large cards, staged offload on small ones.
        tier = mode
        if tier == 'auto':
            try:
                import torch
                if torch.cuda.is_available():
                    gigabytes = torch.cuda.get_device_properties(0).total_memory / 2**30
                    tier = 'full' if gigabytes >= 20 else 'lowvram' if gigabytes >= 10 else 'novram'
                else:
                    tier = 'cpu'
            except Exception:
                tier = 'full'
        if tier == 'lowvram':
            sys.argv.append('--lowvram')
        elif tier == 'novram':
            sys.argv.append('--novram')
        elif tier == 'cpu':
            sys.argv.append('--cpu')
    import comfy.sd
    import comfy.sample
    import comfy.samplers
    import comfy.model_management as mm
    from pi_h3.vendor.fizgig import FizgigH3StillLatent, FizgigH3StillDecode
    emit('ready', device=str(mm.get_torch_device()), samplers=comfy.samplers.KSampler.SAMPLERS,
         schedulers=comfy.samplers.KSampler.SCHEDULERS)


def preview(x0, width, height):
    import numpy as np
    import torch
    from PIL import Image
    from comfy.latent_formats import MiniMaxH3Video
    import comfy.utils
    # Sampler callbacks receive a NestedTensor, or the unpacked video tensor.
    if getattr(x0, 'is_nested', False):
        streams = x0.unbind()
        if not streams:
            raise ValueError('H3 preview contains no latent streams.')
        x0 = streams[0]
    if x0.ndim != 5 or x0.shape[1] < 24 or x0.shape[2] < 1:
        raise ValueError('H3 preview did not contain a video latent.')
    z = x0[0, :24, 0].float().cpu().movedim(0, -1)
    factors = torch.tensor(MiniMaxH3Video.latent_rgb_factors)
    bias = torch.tensor(MiniMaxH3Video.latent_rgb_factors_bias)
    pixels = ((z @ factors + bias + 1) * 127.5).clamp(0, 255).byte().numpy()
    image = Image.fromarray(pixels)
    scale = min(1, 384 / max(width, height))
    image = image.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.BILINEAR)
    stream = io.BytesIO()
    image.save(stream, format='JPEG', quality=80)
    return base64.b64encode(stream.getvalue()).decode('ascii')


def lora_patches(model, weights, source, lora_module=None):
    """Load an adapter only when every tensor belongs to this H3 model."""
    if lora_module is None:
        import comfy.lora
        lora_module = comfy.lora
    unmatched = []

    class Missing(logging.Filter):
        def filter(self, record):
            message = record.getMessage()
            if message.startswith('lora key not loaded: '):
                unmatched.append(message.removeprefix('lora key not loaded: '))
                return False
            return True

    missing_filter = Missing()
    root_logger = logging.getLogger()
    root_logger.addFilter(missing_filter)
    try:
        keys = lora_module.model_lora_keys_unet(model.model, {})
        patches = lora_module.load_lora(weights, keys, log_missing=True)
    finally:
        root_logger.removeFilter(missing_filter)
    if unmatched:
        sample = ', '.join(unmatched[:3])
        raise ValueError(f'LoRA has {len(unmatched)} unsupported tensor(s): {sample} ({source})')
    if not patches:
        raise ValueError('LoRA contains no weights matching this H3 model: ' + source)
    return patches


class Engine:
    def __init__(self):
        self.key = None
        self.model = self.clip = self.vae = None
        self.cond_cache = {}

    def load(self, paths):
        import comfy.sd
        import comfy.utils
        from pi_h3.assets import fingerprint
        key = fingerprint([paths[k] for k in ('dit', 'clip', 'vae')])
        if self.key == key:
            return
        import comfy.model_management as mm
        mm.unload_all_models()
        self.model = self.clip = self.vae = None
        import gc
        gc.collect()
        emit('status', text='loading H3 model')
        model = comfy.sd.load_diffusion_model(paths['dit'])
        if type(model.model).__name__ != 'MiniMaxH3':
            raise ValueError('Selected checkpoint is not a MiniMax H3 model.')
        emit('status', text='loading H3 text encoder')
        clip = comfy.sd.load_clip([paths['clip']], clip_type=comfy.sd.CLIPType.MINIMAX)
        if 'MiniMax' not in type(clip.tokenizer).__name__:
            raise ValueError('Selected encoder is not the H3 Qwen3-VL-32B encoder.')
        emit('status', text='loading H3 video VAE')
        vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(paths['vae'], safe_load=True))
        fsm = vae.first_stage_model
        if not all(hasattr(fsm, name) for name in ('_adaptive_decode', '_finalize_pixels', 'latents_mean', 'latents_std')):
            raise ValueError('Select the MiniMax H3 video VAE, not the audio or single-frame VAE.')
        self.model, self.clip, self.vae, self.key = model, clip, vae, key

    def condition(self, prompt, references, width, height):
        import math
        import numpy as np
        import torch
        from PIL import Image
        import comfy.utils
        items, blocks = [], []
        for filename in references:
            with Image.open(filename) as image:
                image = image.convert('RGB')
                image = torch.from_numpy(np.array(image).astype(np.float32) / 255).unsqueeze(0)
            h, w = image.shape[1:3]
            scale = min(1.0, math.sqrt(width * height / (w * h)))
            tw, th = max(32, round(w * scale / 32) * 32), max(32, round(h * scale / 32) * 32)
            image = comfy.utils.common_upscale(image.movedim(-1, 1), tw, th, 'lanczos', 'disabled').movedim(1, -1)
            items.append({'type': 'image', 'data': image})
            blocks.append({'kind': 'image', 'latent_h': th // 16, 'latent_w': tw // 16, 'latent': self.vae.encode(image)})
        tokens = self.clip.tokenize(prompt, **({'minimax_ref_items': items} if items else {}))
        cond = self.clip.encode_from_tokens_scheduled(tokens)
        if blocks:
            cond = [[value, {**metadata, 'minimax_refs': blocks}] for value, metadata in cond]
        return cond

    def cached_condition(self, positive_prompt, negative_prompt, references, width, height, cfg):
        """Encode each prompt once per batch; identical requests reuse the conditioning."""
        import hashlib
        from .assets import fingerprint
        def encode(prompt):
            key = hashlib.sha256(repr((prompt, width, height, fingerprint(references) if references else ())).encode('utf-8', 'replace')).hexdigest()
            cached = self.cond_cache.get(key)
            if cached is None:
                cached = self.condition(prompt, references, width, height)
                if len(self.cond_cache) >= 4:
                    self.cond_cache.clear()
                self.cond_cache[key] = cached
            return cached
        return encode(positive_prompt), encode(positive_prompt if cfg == 1 else negative_prompt)

    def run(self, request):
        import gc
        import torch
        import comfy.sd
        import comfy.utils
        import comfy.sample
        import comfy.model_management as mm
        from pi_h3.vendor.fizgig import FizgigH3StillLatent, FizgigH3StillDecode
        self.load(request['paths'])
        width, height = request['width'], request['height']
        model = self.model
        candidate = None
        try:
            emit('status', text='encoding prompt and references')
            positive, negative = self.cached_condition(request['prompt'], request['negative'], request['references'], width, height, request['cfg'])
            if request['adapters']:
                # One disposable clone owns every adapter, including quantized patches.
                candidate = self.model.clone()
                model = candidate
                for adapter in request['adapters']:
                    weights = comfy.utils.load_torch_file(adapter['path'], safe_load=True)
                    patches = lora_patches(self.model, weights, adapter['path'])
                    applied = candidate.add_patches(patches, adapter['strength'])
                    if len(applied) != len(patches):
                        raise ValueError('LoRA has unsupported H3 targets: ' + adapter['path'])
            latent = FizgigH3StillLatent().make(width, height, 1)[0]['samples']
            noise = comfy.sample.prepare_noise(latent, request['seed'])
            last_preview = 0.0
            preview_failed = False
            def callback(step, x0, x, total):
                nonlocal last_preview, preview_failed
                data = {'step': step + 1, 'steps': total}
                now = time.monotonic()
                if request['preview'] and not preview_failed and now - last_preview >= request['preview_interval']:
                    try:
                        data['preview'] = preview(x0, width, height)
                    except Exception as error:
                        print('[H3] Preview disabled for this image:', error, file=sys.stderr)
                        preview_failed = True
                    last_preview = now
                emit('progress', **data)
            emit('status', text='sampling')
            with torch.inference_mode():
                output = comfy.sample.sample(model, noise, request['steps'], request['cfg'], request['sampler'],
                    request['scheduler'], positive, negative, latent, callback=callback,
                    disable_pbar=True, seed=request['seed'])
                emit('status', text='decoding final image')
                image = FizgigH3StillDecode().decode(self.vae, {'samples': output})[0][0]
                from PIL import Image
                pixels = image.detach().float().clamp(0, 1).cpu().numpy()
                Image.fromarray((pixels * 255).round().astype('uint8')).save(request['output'])
                emit('result', path=request['output'])
        finally:
            # Release request-only patches and restore quantized base weights.
            # Models stay resident when the next request uses the same files (batch mode);
            # the Forge side decides when to truly release via the shutdown request.
            if candidate is not None:
                candidate.unpatch_model()
                candidate.patches.clear()
                candidate.parent = None
            del model
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception:
                pass


if __name__ == '__main__':
    try:
        startup()
        engine = Engine()
        for line in sys.stdin:
            request = json.loads(line)
            if request.get('type') == 'shutdown':
                break
            try:
                engine.run(request)
            except Exception as error:
                traceback.print_exc(file=sys.stderr)
                emit('error', message=str(error))
                break  # failed/partial state is never cached
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        emit('error', message=str(error))
