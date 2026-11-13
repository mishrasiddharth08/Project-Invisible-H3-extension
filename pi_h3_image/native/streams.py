"""The packed latent of an H3 still image: video [1, 24, 1, H/16, W/16] and the audio stub [1, 32, 2, 2] in one tensor.

Forge's sampler works on one tensor; ComfyUI packs the two streams the same way (comfy.utils.pack_latents).
MiniMax-H3-Image has no audio stream: its audio part is empty, [1, 32, 2, 0], and the packed latent is the picture's.
"""

import math
from dataclasses import dataclass, field

import torch

from ..contracts import AUDIO_LATENTS

SPATIAL = 16
LATENT_CHANNELS = 24
AUDIO_CHANNELS = 32


@dataclass(frozen=True)
class StreamShapes:
    video: tuple[int, ...]
    audio: tuple[int, ...]

    @property
    def video_size(self) -> int:
        return math.prod(self.video[1:])

    @property
    def size(self) -> int:
        return self.video_size + math.prod(self.audio[1:])

    def pack(self, video: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        # [B, 1, 1, N]: Forge's sampler reads the latent's dims 2 and 3 as an (H, W) area, so it must be 4D
        flat = torch.cat([video.reshape(video.shape[0], -1), audio.reshape(audio.shape[0], -1)], dim=-1)
        return flat.reshape(flat.shape[0], 1, 1, -1)

    def unpack(self, packed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = packed.shape[0]
        flat = packed.reshape(b, -1)
        n = self.video_size
        return flat[:, :n].reshape(b, *self.video[1:]), flat[:, n:].reshape(b, *self.audio[1:])


def still_shapes(width: int, height: int, audio: bool = True) -> StreamShapes:
    return StreamShapes(video=(1, LATENT_CHANNELS, 1, height // SPATIAL, width // SPATIAL),
                        audio=(1, AUDIO_CHANNELS, 2, AUDIO_LATENTS if audio else 0))


def text_token_tags(text_len: int, spans) -> torch.Tensor | None:
    """adaLN modality tag of every prompt token: 1 for text, 0 inside the vision blocks; None without images."""
    if not spans:
        return None
    tags = torch.ones(text_len, dtype=torch.long)
    for start, stop in spans:
        tags[start:min(stop, text_len)] = 0
    return tags


def preview_frame(packed: torch.Tensor, shapes: StreamShapes | None) -> torch.Tensor:
    """The video frame [1, 24, H/16, W/16] of a packed latent, for Forge's RGB live preview; anything that is not
    the current generation's packed latent comes back unchanged."""
    if shapes is None or packed.shape[-1] != shapes.size:
        return packed
    video, _ = shapes.unpack(packed[:1])
    return video[:, :, 0]


@dataclass
class Generation:
    """What the transformer and the decoder need for the current request; set by prepare()."""
    shapes: StreamShapes
    seed: int
    # reference pictures: {"kind": "image", "latent_h", "latent_w", "latent": [1, 24, 1, h, w]}, each at its own size,
    # in "<Picture i>" order
    refs: list = field(default_factory=list)
    # vision block token ranges of the prompt, which the DiT tags as video modality
    vision_spans: list = field(default_factory=list)
    frame: torch.Tensor | None = None  # [1, 3, H, W] float in [0, 1], after decoding
    memory_reported: bool = False      # the first sampling step printed its working memory
