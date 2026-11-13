"""The [H3] Resolution shift setting: MiniMax-H3-Image's sigma grid moved with the picture size, as Krea 2 and Flux
move their shift. Plain math, so CPU tests cover it.

MiniMax's grid is a time-shifted curve: logit(sigma) = ln 3 + 1.42 logit(t) at t = 5/6 ... 1/6, within 0.02, so it is
shift 3 with a steeper slope. A flow shift multiplies the odds sigma / (1 - sigma) of every point by one factor, so a
resolution shift is a factor on the odds of the grid, (tokens / 16 MP tokens) ** strength: below 16 MP it lowers every
sigma, above it raises them. Strength 0.5 is SD3's rule (shift in proportion to the square root of the pixel count);
0 keeps MiniMax's fixed grid.
"""

# 16 MP (4096x4096), the size the checkpoint is best at: a 16x VAE and 2x2 patches make one token of 32x32 pixels
TOKEN_PIXELS = 32
REFERENCE_TOKENS = (4096 // TOKEN_PIXELS) ** 2


def tokens(width: int, height: int) -> int:
    return max(int(width) // TOKEN_PIXELS, 1) * max(int(height) // TOKEN_PIXELS, 1)


def odds_factor(width: int, height: int, strength: float) -> float:
    """The factor on the odds of each grid sigma for a picture of this size: 1 at 16 MP or at strength 0."""
    if not strength:
        return 1.0
    return (tokens(width, height) / REFERENCE_TOKENS) ** float(strength)


def shift_sigma(sigma: float, factor: float) -> float:
    """sigma with its odds multiplied by factor (a flow shift of factor); 0 and 1 stay where they are."""
    sigma = float(sigma)
    if factor == 1.0 or sigma <= 0.0 or sigma >= 1.0:
        return sigma
    return factor * sigma / (1.0 + (factor - 1.0) * sigma)


def shift_sigmas(sigmas, factor: float):
    """A schedule (a tensor) with each sigma shifted by factor."""
    if factor == 1.0:
        return sigmas
    return factor * sigmas / (1.0 + (factor - 1.0) * sigmas)
