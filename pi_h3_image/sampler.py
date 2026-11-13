"""The "H3-Image LCM" sampler: LCM with a noise scale that falls across the steps. Torch only, so CPU tests cover it.

Each step takes the model's clean estimate and noises it afresh to the next sigma, as Forge's LCM does; the fresh noise
is scaled from S_NOISE_START at the first step to S_NOISE_END at the last, as ComfyUI's SamplerLCM does with s_noise and
s_noise_end. On MiniMax-H3-Image this is what removes the speckle and grit that Euler and Heun leave at the checkpoint's
own last sigmas below about 16 MP (at 16 MP Euler is clean too): Euler carries the model's own noise estimate into the
next step, and the low-sigma output heads turn it into speckle, while fresh noise is what they were distilled on. 1.10
to 0.90 is the community workflow's setting (Simple, 6 steps).
"""

import torch

SAMPLER = "H3-Image LCM"
S_NOISE_START = 1.10
S_NOISE_END = 0.90


def noise_scale(step: int, steps: int, start: float = S_NOISE_START, end: float = S_NOISE_END) -> float:
    """The noise multiplier after step `step` of `steps`: linear from start (first step) to end (last step)."""
    if steps <= 1:
        return start
    return start + (end - start) * step / (steps - 1)


@torch.no_grad()
def sample_h3_image_lcm(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None):
    """k-diffusion sampler: Forge passes the model wrapper, the start latent and the sigma list. Without a
    noise_sampler it draws from k-diffusion's default one, which Forge points at the image's seeded RNG, so a seed gives
    the same picture every time."""
    extra_args = {} if extra_args is None else extra_args
    if noise_sampler is None:
        from k_diffusion.sampling import default_noise_sampler

        noise_sampler = default_noise_sampler(x)
    predictor = model.inner_model.predictor
    s_in = x.new_ones([x.shape[0]])
    steps = len(sigmas) - 1
    for i in range(steps):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
        x = denoised
        if sigmas[i + 1] > 0:
            noise = noise_sampler(sigmas[i], sigmas[i + 1]) * noise_scale(i, steps)
            x = predictor.noise_scaling(sigmas[i + 1], noise, x)
    return x
