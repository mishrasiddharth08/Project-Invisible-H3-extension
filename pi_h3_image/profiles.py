"""The kinds of still-image H3 model, their sampling recipes and their schedules. Torch only, so CPU tests cover it."""

import functools
from dataclasses import dataclass

from .resolution_shift import shift_sigmas
from .sampler import SAMPLER

# the ManualSigmas grid of Pruna's workflows: 8 Euler steps at shift 1, no CFG
PH3_IMAGE_SIGMAS = (1.0, 0.984753, 0.967905, 0.951688, 0.925372, 0.881923, 0.810083, 0.528718, 0.0)
PH3_IMAGE_SCHEDULER = "P-H3-Image"
# the hidden states of the 4B text encoder that Pruna's text connector reads (36 = after the final norm), as Forge's
# clip wrapper asks the encoder for them
PH3_IMAGE_HIDDEN_STATES = [13, 19, 25, 31, 36]
# how the prompt names a picture before its vision block
PICTURE_LABEL = "<Picture %d>: "
H3_IMAGE_PICTURE_LABEL = "Picture %d: "

# the sigma grid of MiniMax-H3-Image's six output heads (the checkpoint's pdd_config.json)
H3_IMAGE_PDD_SIGMAS = (1.0, 0.9675236940383911, 0.8875485062599182, 0.75, 0.5327739119529724, 0.23200887441635132, 0.0)

# The MiniMax-H3-Image schedules for Euler and Heun (the default sampler, H3-Image LCM, runs MiniMax's own grid
# through Simple and needs none of them). Same-seed tests on an RTX 5090 (Euler, CFG 1, 6.4 MP): the picture is clean when
# the last model evaluation is at sigma 0.27 or above and the sampler jumps from there to 0, and comes out gritty when
# the model is evaluated lower: the grid's own last point, 0.232, already leaves speckle, and Simple at 8 steps (0.38,
# then 0.17) is far worse. Anything from 0.27 to 0.42 gives the same picture, 0.53 a softer one, 0.76 an under-denoised
# one. So each schedule ends at a fixed sigma well inside the clean range, whatever the step count.
H3_IMAGE_TAIL = 0.42
# "H3-Image 8-step": a shifted flow curve, sigma = s t / (1 + (s - 1) t) with t uniform from 1 down to 1/8, where s
# puts the last point on the tail (s = 5.069). The crispest of the tested lists; more steps sample the same curve.
H3_IMAGE_8_STEP = "H3-Image 8-step"
H3_IMAGE_CURVE_END = 1.0 / 8.0
H3_IMAGE_CURVE_SHIFT = H3_IMAGE_TAIL * (1.0 / H3_IMAGE_CURVE_END - 1.0) / (1.0 - H3_IMAGE_TAIL)
# "H3-Image 8-step softer": the same eight steps with a slower start, each of the first seven sigmas halfway between
# the 8-step curve and Pruna's grid (the list as tested). The first steps decide how smooth the picture is: with Heun
# it measures between the two on sharpness, fine grain and dark specks; the last sigma barely matters.
H3_IMAGE_8_STEP_SOFTER = "H3-Image 8-step softer"
H3_IMAGE_8_STEP_SOFTER_SIGMAS = (1.0, 0.9787, 0.9531, 0.923, 0.8803, 0.8172, 0.7192, H3_IMAGE_TAIL)
# "H3-Image 6-step": the checkpoint's grid as MiniMax made it, with the last point lifted from 0.232 to the tail
H3_IMAGE_6_STEP = "H3-Image 6-step"
H3_IMAGE_6_STEP_SIGMAS = H3_IMAGE_PDD_SIGMAS[:5] + (H3_IMAGE_TAIL,)
# (the grid without its last point, 5 steps ending at 0.53, also runs clean but with about a fifth more fine grain;
# it was a schedule in 0.10.0 and is left out)


@dataclass(frozen=True)
class Profile:
    name: str          # the UI preset, and the key of the Settings options
    label: str
    sampler: str
    scheduler: str
    steps: int
    cfg: float
    shift: float       # the flow shift of both streams
    size: int          # the recommended side of a square picture
    text_encoder: str  # the text encoder variant it needs (models.ModelInfo.variant)
    vae: str           # the VAE variant it needs
    dit: str = "h3"    # the checkpoint variant: "h3" is the audio-video layout P-H3-Image is released in
    # the edit mode of img2img: how many pictures it takes (0 = none), how the prompt names them, and whether a short
    # instruction gets "Edit <Picture 1>. " in front
    max_pictures: int = 0
    picture_label: str = PICTURE_LABEL
    edit_prefix: bool = False
    # whether the Shift slider does anything (a model on its own sigma grid ignores it)
    shift_used: bool = True


# Pruna's P-H3-Image: its 4B text encoder and single-frame VAE in front of an H3 audio-video DiT (Pruna's own
# checkpoint, or MiniMax H3 with Pruna's LoRA), 8 MP output
PH3_IMAGE = Profile(name="ph3", label="P-H3-Image", sampler="Euler", scheduler=PH3_IMAGE_SCHEDULER, steps=8, cfg=1.0, shift=1.0,
                    size=2912, text_encoder="ph3-image", vae="ph3-image", max_pictures=8, edit_prefix=True)
# MiniMax's own MiniMax-H3-Image: the H3 DiT without the audio stream, six output heads distilled on the checkpoint's
# sigma grid, the H3 32B text encoder and a 2D image VAE; no CFG, about 4 to 20 MP, best around 16 MP.
# MiniMax's reference recipe is Euler, Simple, 6 steps (Simple lands on the checkpoint's grid); its last step, at sigma
# 0.232, leaves speckle below about 16 MP, in ComfyUI too, and is clean at 16 MP. The default keeps the grid and swaps
# the sampler: LCM noises each clean estimate afresh, and the same six sigmas come out clean at any size (sampler.py). The H3-Image schedules above are for Euler and
# Heun, which need a last sigma of 0.27 or more.
H3_IMAGE = Profile(name="h3image", label="MiniMax-H3-Image", sampler=SAMPLER, scheduler="Simple", steps=6, cfg=1.0,
                   shift=1.0, size=4096, text_encoder="h3", vae="h3-image", dit="h3-image", max_pictures=16,
                   picture_label=H3_IMAGE_PICTURE_LABEL, shift_used=False)
# MiniMax's reference recipe for it, for the panel and the documentation
H3_IMAGE_REFERENCE = ("Euler", "Simple", 6)

ALL = (PH3_IMAGE, H3_IMAGE)


def for_components(dit: str, text_encoder: str) -> Profile | None:
    """The image checkpoint is told by its own tensors; an audio-video checkpoint is a still-image model only behind
    P-H3-Image's text encoder. None for any other set (MiniMax H3 with its own text encoder is a video model)."""
    if dit == H3_IMAGE.dit:
        return H3_IMAGE
    return PH3_IMAGE if text_encoder == PH3_IMAGE.text_encoder else None


def resample(points, n: int, device=None):
    """Forge scheduler body: the sigma list as it is at its own step count, and for another count the same shape
    resampled (linear interpolation over the step index); the first and the last sigma never move, then 0."""
    import torch
    points = torch.tensor(points, dtype=torch.float64)
    if n != len(points):
        positions = torch.linspace(0.0, len(points) - 1.0, max(int(n), 1), dtype=torch.float64)
        lower = positions.floor().long().clamp(max=len(points) - 2)
        points = torch.lerp(points[lower], points[lower + 1], positions - lower)
    return torch.cat([points, points.new_zeros(1)]).to(dtype=torch.float32, device=device)


def ph3_image_sigmas(n: int, sigma_min=0.0, sigma_max=1.0, device=None, **kwargs):
    """Forge scheduler function: Pruna's 8-step grid as it is, and for another step count the same curve resampled
    (linear interpolation over the step index, the closing 0 included, so a 4-step run keeps the grid's shape). Shift
    does not apply: the grid is given in the model's own sigma space, as ComfyUI's ManualSigmas is."""
    import torch
    grid = torch.tensor(PH3_IMAGE_SIGMAS, dtype=torch.float64)
    if n == len(grid) - 1:
        sigmas = grid
    else:
        positions = torch.linspace(0.0, len(grid) - 1, n + 1, dtype=torch.float64)
        lower = positions.floor().long().clamp(max=len(grid) - 2)
        sigmas = torch.lerp(grid[lower], grid[lower + 1], positions - lower)
    return sigmas.to(dtype=torch.float32, device=device)


def h3_image_8_step_sigmas(n: int, sigma_min=0.0, sigma_max=1.0, device=None, **kwargs):
    """The shifted flow curve from 1 to the tail over n steps, then 0: at 8 steps the tested list, at any other count
    the same curve sampled coarser or finer. The model's sigma bounds and the Shift slider do not apply."""
    import torch
    t = torch.linspace(1.0, H3_IMAGE_CURVE_END, max(int(n), 1), dtype=torch.float64)
    sigmas = H3_IMAGE_CURVE_SHIFT * t / (1.0 + (H3_IMAGE_CURVE_SHIFT - 1.0) * t)
    return torch.cat([sigmas, sigmas.new_zeros(1)]).to(dtype=torch.float32, device=device)


def h3_image_8_step_softer_sigmas(n: int, sigma_min=0.0, sigma_max=1.0, device=None, **kwargs):
    return resample(H3_IMAGE_8_STEP_SOFTER_SIGMAS, n, device)


def h3_image_6_step_sigmas(n: int, sigma_min=0.0, sigma_max=1.0, device=None, **kwargs):
    return resample(H3_IMAGE_6_STEP_SIGMAS, n, device)


def follow_grid(function):
    """A schedule of fixed sigmas that moves with the [H3] Resolution shift: Forge passes the model's schedule linker
    (inner_model) to a schedule that asks for it, and when its predictor has moved MiniMax-H3-Image's grid by an odds
    factor, every sigma moves by the same factor. Other predictors have no factor and keep the list as it is."""
    @functools.wraps(function)
    def schedule(n, *args, inner_model=None, **kwargs):
        factor = getattr(getattr(inner_model, "predictor", None), "odds_factor", 1.0)
        return shift_sigmas(function(n, *args, **kwargs), float(factor))
    return schedule


# Forge schedules: (name, label, function); each follows the [H3] Resolution shift
SCHEDULERS = (
    ("ph3_image", PH3_IMAGE_SCHEDULER, follow_grid(ph3_image_sigmas)),
    ("h3_image_8", H3_IMAGE_8_STEP, follow_grid(h3_image_8_step_sigmas)),
    ("h3_image_8_softer", H3_IMAGE_8_STEP_SOFTER, follow_grid(h3_image_8_step_softer_sigmas)),
    ("h3_image_6", H3_IMAGE_6_STEP, follow_grid(h3_image_6_step_sigmas)),
)
