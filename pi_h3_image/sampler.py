"""H3-Image LCM variants with a noise ramp or fixed 0.9 multiplier."""

import torch

SAMPLER = "H3-Image LCM"
SAMPLER_FIXED = "H3-Image LCM 0.9"
S_NOISE_START = 1.10
S_NOISE_END = 0.90


def noise_scale(step, steps, start=S_NOISE_START, end=S_NOISE_END):
    return start if steps <= 1 else start + (end - start) * step / (steps - 1)


@torch.no_grad()
def sample_h3_image_lcm(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None,
                        *, noise_start=S_NOISE_START, noise_end=S_NOISE_END):
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
            noise = noise_sampler(sigmas[i], sigmas[i + 1]) * noise_scale(i, steps, noise_start, noise_end)
            x = predictor.noise_scaling(sigmas[i + 1], noise, x)
    return x
