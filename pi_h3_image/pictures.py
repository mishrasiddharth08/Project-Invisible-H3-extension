"""The pictures of an edit, from Forge Neo's own image inputs (ComfyUI PH3ImageConditioning and
MiniMaxH3ImageConditioning).

In img2img the input image is <Picture 1>, the picture to edit; the gallery of Forge Neo's built-in ImageStitch
Integrated holds the references, <Picture 2> onwards. The original picture is taken, not the copy Forge resized to
the output size, and each model sizes it its own way, aspect kept, sides on the 32 grid:

- P-H3-Image: within 1 MP for the text encoder and within the output area for the VAE, never enlarged;
- MiniMax-H3-Image: a 1024 short edge for the text encoder, a short edge between 1024 and 2048 for the VAE.
"""

import base64
import io
import math
import re

import numpy as np
from PIL import Image

from .contracts import GRID, H3Error

IMAGE_STITCH = "ImageStitch Integrated"
# P-H3-Image: what the text encoder sees of a picture
VISION_PIXELS = 1_000_000
# MiniMax-H3-Image: the short edge the text encoder sees, and the range the VAE keeps it in
H3_IMAGE_VISION_SHORT_EDGE = 1024
H3_IMAGE_VAE_SHORT_EDGE = (1024, 2048)
H3_IMAGE_PROFILE = "h3image"
# a short instruction gets the edit prefix; an edit brief from Pruna's prompt upsampler, or an instruction that
# already names the picture to edit, is encoded as it is
EDIT_PREFIX = "Edit <Picture 1>. "
EDIT_VERBATIM = re.compile(r"\s*(Edit <Picture \d+>|subject_definitions:)")

_warned_extra = False


def load(value):
    # a gallery item is (image, caption) in the UI; through the API it arrives as base64, with or without a data: prefix
    if isinstance(value, (tuple, list)):
        value = value[0] if value else None
    if value is None or isinstance(value, Image.Image):
        return value
    if isinstance(value, str) and value:
        return Image.open(io.BytesIO(base64.b64decode(value.split(",", 1)[-1])))
    return None


def stitch_gallery(p) -> list:
    """The images in ImageStitch's gallery, when its accordion is on."""
    runner = getattr(p, "scripts", None)
    for script in getattr(runner, "alwayson_scripts", []):
        if script.title() == IMAGE_STITCH:
            enable, gallery = (list(p.script_args[script.args_from:script.args_to]) + [False, None])[:2]
            if not enable or not gallery:
                return []
            images = [load(item) for item in gallery]
            return [image for image in images if image is not None]
    return []


def collect(p, limit: int, label: str = "H3") -> list:
    """<Picture 1> is the img2img input image, the gallery follows; at most `limit` pictures in all."""
    images = list(getattr(p, "init_images", None) or [])[:1] + stitch_gallery(p)
    if len(images) > limit:
        raise H3Error(f"{label} edits with up to {limit} pictures, the input image included; "
                      f"{len(images)} were given.")
    return images


def within_area(width: int, height: int, pixels: int) -> tuple[int, int]:
    """Aspect-preserving scale (down only) to `pixels`, rounded to the canvas grid."""
    scale = min(1.0, math.sqrt(pixels / (width * height)))
    return tuple(max(GRID, round(side * scale / GRID) * GRID) for side in (width, height))


def short_edge(width: int, height: int, low: int, high: int) -> tuple[int, int]:
    """Aspect-preserving scale putting the short edge inside [low, high], rounded to the canvas grid."""
    scale = min(max(min(width, height), low), high) / min(width, height)
    return tuple(max(GRID, round(side * scale / GRID) * GRID) for side in (width, height))


def to_tensor(image: Image.Image, size: tuple[int, int]):
    """(1, h, w, 3) float in [0, 1] at `size` (width, height), the form the text encoder and the VAE take."""
    import torch
    image = image.convert("RGB")
    if tuple(size) != image.size:
        image = image.resize(tuple(size), Image.Resampling.LANCZOS)
    return torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).unsqueeze(0)


def prepare(image: Image.Image, pixels: int):
    """A picture within the pixel budget, never enlarged."""
    return to_tensor(image, within_area(image.width, image.height, pixels))


def edit_inputs(profile, images, width: int, height: int) -> tuple[list, list]:
    """The pictures of an edit as the VAE takes them and as the text encoder sees them, in <Picture i> order."""
    vae, vision = [], []
    for image in images:
        if profile.name == H3_IMAGE_PROFILE:
            vae.append(to_tensor(image, short_edge(image.width, image.height, *H3_IMAGE_VAE_SHORT_EDGE)))
            vision.append(to_tensor(image, short_edge(image.width, image.height, H3_IMAGE_VISION_SHORT_EDGE,
                                                      H3_IMAGE_VISION_SHORT_EDGE)))
        else:
            vae.append(prepare(image, width * height))
            vision.append(prepare(image, VISION_PIXELS))
    return vae, vision


def edit_prompt(prompt: str) -> str:
    """The instruction as the model reads it when pictures are present."""
    if EDIT_VERBATIM.match(prompt):
        return prompt.strip()
    return EDIT_PREFIX + prompt
