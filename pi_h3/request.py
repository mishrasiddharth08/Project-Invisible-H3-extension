"""Validate inexpensive request settings before starting the model worker."""
import math
import re

SAMPLERS = {'ER SDE': 'er_sde', 'Euler': 'euler', 'Heun': 'heun', 'DPM++ 2M': 'dpmpp_2m',
            'Res Multistep': 'res_multistep'}
SCHEDULERS = {'Simple': 'simple', 'Normal': 'normal', 'Beta': 'beta', 'Karras': 'karras'}


def validate(width, height, steps, cfg, sampler, scheduler, editing=False, image=None, mask=None,
             denoise=1.0, hires=False):
    if any(int(v) != v or not 64 <= v <= 4096 or v % 32 for v in (width, height)):
        raise ValueError('H3 width and height must be multiples of 32, between 64 and 4096.')
    if int(steps) != steps or not 1 <= steps <= 150:
        raise ValueError('Choose 1–150 steps.')
    if not math.isfinite(cfg) or not 1 <= cfg <= 20:
        raise ValueError('H3 CFG must be between 1 and 20; use 1 for the reference workflow.')
    if sampler not in SAMPLERS:
        raise ValueError('Unsupported H3 sampler. Choose ER SDE, Euler, Heun, DPM++ 2M or Res Multistep.')
    if scheduler not in SCHEDULERS and scheduler != 'Automatic':
        raise ValueError('Unsupported H3 schedule. Choose Simple, Normal, Beta or Karras.')
    if editing and image is None:
        raise ValueError('Add an image in img2img before generating an edit.')
    if mask is not None:
        raise ValueError('H3 still reference editing does not support masks. Use plain img2img.')
    if editing and abs(float(denoise) - 1.0) > 1e-6:
        raise ValueError('Set img2img Denoising strength to 1. H3 uses reference conditioning, not SD denoising.')
    if hires:
        raise ValueError('Disable Hires fix for H3. Set the desired final width and height directly.')
    return SAMPLERS[sampler], SCHEDULERS.get(scheduler, 'simple')


def parse_loras(prompt):
    adapters = []
    def take(match):
        name, strength = match.group(1), float(match.group(2))
        if not math.isfinite(strength) or not -2 <= strength <= 2:
            raise ValueError('LoRA strength must be between -2 and 2.')
        adapters.append((name, strength))
        return ''
    prompt = re.sub(r'<lora:([^:<>]+):([-+\d.eE]+)>', take, prompt)
    if re.search(r'<(?:lora|lyco|hypernet):', prompt, re.I):
        raise ValueError('Unsupported or malformed adapter tag. Use <lora:H3-filename:strength>.')
    return prompt.strip(), adapters
