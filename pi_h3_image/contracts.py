"""Request validation shared by the UI, the API and the integration."""

from dataclasses import dataclass

# the sides H3 takes: 16x spatial compression and a 2x2 patch
GRID = 32
MIN_SIDE = 64
# a still image is one video latent frame; its audio stub spans the two audio latents of a single frame at 40 per
# second (ComfyUI PH3ImageConditioning), sampled with the picture and never decoded
AUDIO_LATENTS = 2
# the most pictures an edit takes, the input image included (MiniMax-H3-Image; P-H3-Image takes 8): each profile
# has its own limit (profiles.Profile.max_pictures)
MAX_REFERENCES = 16


class H3Error(RuntimeError):
    """An actionable configuration or generation failure."""


# Forge logs and swallows exceptions raised in script callbacks; a rejected H3 request is kept here and raised
# again from the model, which Forge calls directly
_pending_error = None


def set_pending_error(error):
    global _pending_error
    _pending_error = error


def raise_pending_error():
    if _pending_error is not None:
        raise _pending_error


def integer(value, label):
    try:
        result = int(value)
        if float(value) != result:
            raise ValueError
        return result
    except (TypeError, ValueError, OverflowError):
        raise H3Error(f"{label} must be an integer.") from None


@dataclass
class GenerationRequest:
    """What H3 adds to Forge's own generation settings: the picture size and the pictures it is conditioned on."""
    width: int = 1024
    height: int = 1024
    references: int = 0

    def __post_init__(self):
        self.width = integer(self.width, "Width")
        self.height = integer(self.height, "Height")
        if min(self.width, self.height) < MIN_SIDE or self.width % GRID or self.height % GRID:
            raise H3Error(f"H3 width and height must be multiples of {GRID}, at least {MIN_SIDE}.")
        self.references = integer(self.references, "References")
        if not 0 <= self.references <= MAX_REFERENCES:
            raise H3Error(f"H3 takes up to {MAX_REFERENCES} pictures; {self.references} were given.")
