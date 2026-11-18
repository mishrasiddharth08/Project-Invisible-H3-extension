"""The H3 image autoencoders as a Forge component, and the state-dict conversion they accept.

Accepted files: the two image VAEs, the H3 encoder with a single-frame decoder: P-H3-Image's (bf16 or int8; its
decoder transformer linears may be quantized, Forge's mixed-precision operations load them) and MiniMax-H3-Image's 2D
one, which has no latent statistics buffers. The video VAE of MiniMax H3 has the same encoder and a decoder that makes
four frames per latent: it is told apart here only to be refused, as the audio VAE is in models.py.
"""

import torch

from .image_vae import H3_IMAGE_DECODER_DIM, IMAGE_PROJ_OUT, LATENTS_MEAN, LATENTS_STD

H3_KEYS = ("decoder.transformer_blocks.0.scale1", "encoder.down.5.block.0.conv1.weight")
PROJ_OUT = "decoder.proj_out.weight"
X_EMBEDDER = "decoder.x_embedder.weight"
QUANT_SUFFIXES = (".comfy_quant", ".weight_scale")

MISSING_VAE = ("MiniMax H3 needs an image autoencoder: select the P-H3-Image VAE or the MiniMax-H3-Image VAE under "
               "VAE / Text Encoder")
VIDEO_VAE = ("This is the video VAE of MiniMax H3, which the extension does not load: select the P-H3-Image VAE or the "
             "MiniMax-H3-Image VAE under VAE / Text Encoder")


def _shape(value):
    # a torch tensor, or a safetensors header entry {"dtype", "shape", "data_offsets"}
    return tuple(value["shape"]) if isinstance(value, dict) else tuple(value.shape)


def is_h3_vae(sd) -> bool:
    """Any autoencoder of the H3 family: the shared encoder and a ViT decoder."""
    return all(k in sd for k in H3_KEYS)


def is_image_vae(sd) -> bool:
    """An image VAE: its decoder makes one frame per latent (ComfyUI's test on the output projection)."""
    return is_h3_vae(sd) and PROJ_OUT in sd and _shape(sd[PROJ_OUT])[0] == IMAGE_PROJ_OUT


def is_h3_image_vae(sd) -> bool:
    """MiniMax-H3-Image's VAE: the 16-head 2D decoder (P-H3-Image's has 32 heads)."""
    return is_image_vae(sd) and X_EMBEDDER in sd and _shape(sd[X_EMBEDDER])[0] == H3_IMAGE_DECODER_DIM


def is_ph3_vae(sd) -> bool:
    return is_image_vae(sd) and not is_h3_image_vae(sd)


def is_quantized(sd: dict, prefix: str = "") -> bool:
    return any(k.startswith(prefix) and k.endswith(QUANT_SUFFIXES) for k in sd)


def with_latent_stats(sd: dict) -> dict:
    """The state dict with H3's latent statistics, for a file that leaves them out (MiniMax-H3-Image's)."""
    sd = dict(sd)
    sd.setdefault("latents_mean", torch.tensor(LATENTS_MEAN))
    sd.setdefault("latents_std", torch.tensor(LATENTS_STD))
    return sd


def decoder_layers(sd: dict, prefix: str = "") -> int:
    return sum(k.startswith(f"{prefix}decoder.transformer_blocks.") and k.endswith(".scale1") for k in sd)
