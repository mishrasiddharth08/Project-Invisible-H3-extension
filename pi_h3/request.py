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
    if width * height < 3 * 1024 * 1024:
        # Fizgig Still guidance: below ~3 MP the decoder produces waxy/plastic-looking skin.
        print('[PI-H3] Warning: ' + str(width * height // 1024) + ' KP is below the recommended 3 MP; '
              'skin may look plastic. Use e.g. 1536x2048.')
    if int(steps) != steps or not 1 <= steps <= 150:
        raise ValueError('Choose 1–150 steps.')
    if not math.isfinite(cfg) or not 1 <= cfg <= 20:
        raise ValueError('H3 CFG must be between 1 and 20; use 1 for the reference workflow.')
    if sampler not in SAMPLERS:
        # A config preset from another model may leave its sampler selected; H3 falls back to its recommended sampler.
        print('[PI-H3] Unsupported sampler "' + str(sampler) + '" ignored; using ER SDE.')
        sampler = 'ER SDE'
    if scheduler not in SCHEDULERS and scheduler != 'Automatic':
        # A config preset from another model may leave its scheduler selected; H3 falls back to its recommended schedule.
        print('[PI-H3] Unsupported scheduler "' + str(scheduler) + '" ignored; using Simple.')
    if editing and image is None:
        raise ValueError('Add an image in img2img before generating an edit.')
    if mask is not None:
        raise ValueError('H3 still reference editing does not support masks. Use plain img2img.')
    if editing:
        try:
            denoise = float(denoise)
        except (TypeError, ValueError):
            raise ValueError('Set img2img Denoising strength to 1. H3 uses reference conditioning, not SD denoising.') from None
        if not math.isfinite(denoise) or abs(denoise - 1.0) > 1e-6:
            raise ValueError('Set img2img Denoising strength to 1. H3 uses reference conditioning, not SD denoising.')
    if hires:
        raise ValueError('Disable Hires fix for H3. Set the desired final width and height directly.')
    return SAMPLERS[sampler], SCHEDULERS.get(scheduler, 'simple')


def parse_loras(prompt):
    adapters = []
    def take(match):
        name, raw_strength = match.group(1), match.group(2)
        strength = validate_lora_strength(raw_strength, '<lora:' + name + ':' + raw_strength + '>')
        adapters.append((name, strength))
        return ''
    prompt = re.sub(r'<lora:([^:<>]+):([-+\d.eE]+)>', take, prompt)
    if re.search(r'<(?:lora|lyco|hypernet):', prompt, re.I):
        raise ValueError('Unsupported or malformed adapter tag. Use <lora:H3-filename:strength>.')
    return prompt.strip(), adapters


def validate_lora_strength(value, source='selected LoRA'):
    try:
        strength = float(value)
    except (TypeError, ValueError):
        raise ValueError(f'Invalid LoRA strength for {source}. Use a number between -2 and 2.') from None
    if not math.isfinite(strength) or not -2 <= strength <= 2:
        raise ValueError('LoRA strength must be between -2 and 2.')
    return strength
