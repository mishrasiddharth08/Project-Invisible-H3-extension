"""MiniMax H3 diffusion engine for Forge Neo: text-to-image, with reference pictures.

The video latent [1, 24, 1, H/16, W/16] of a still image and its audio stub [1, 32, 2, 2] travel through Forge's
sampler packed into one flat tensor [1, 1, 1, N], as ComfyUI does (comfy.utils.pack_latents). The script sets up a
generation with prepare(), which returns the packed shape; the transformer unpacks it every step; decode_first_stage
decodes the video latent and hands Forge the picture. The audio stub is sampled with the picture and never decoded.
"""

from fractions import Fraction

import torch
from backend import memory_management
from backend.args import args
from backend.diffusion_engine.base import ForgeDiffusionEngine, ForgeObjects
from backend.patcher.clip import CLIP
from backend.patcher.unet import UnetPatcher
from backend.patcher.vae import VAE

from ..contracts import H3Error, raise_pending_error
from ..profiles import PICTURE_LABEL
from ..resize import area_resize
from .layout import prompt_memory
from .model import SHIFT, MiniMaxH3
from .streams import Generation, still_shapes
from .text_engine import MiniMaxH3TextEngine


def encode_memory(shape, dtype) -> int:
    """Working memory to keep free for one picture's encode, [1, 3, 1, h, w] (Forge's figure for its temporal video
    VAEs, Wan's)."""
    return 6000 * shape[-2] * shape[-1] * memory_management.dtype_size(dtype)


# Forge sizes the sampling headroom from the packed latent's element count (about 100k for a 1 MP still, halved
# again for INT8 models), which left 1.2 GB free next to a resident text encoder on a 32 GB card. The DiT's working
# memory scales with its sequence instead: fp32 rows, the 28672-wide MLP activations, INT8 activation buffers and
# the allocator's slack: a few hundred KB per token by the layers' widths, kept with a margin until a measured run
# says more (the first sampling step prints it). The reservation below is what Forge keeps free when it loads the DiT
# for sampling, so the text encoder leaves VRAM (partly or wholly) when the two do not fit together. MiniMax-H3-Image
# at 16 MP is 16k image tokens: twice the figure would push most of the DiT itself out of a 32 GB card.
SAMPLING_BYTES_PER_TOKEN = 768 * 1024
SAMPLING_FIXED_BYTES = 1024 * 1024 * 1024
# the prompt's share of the sequence is not known when the reservation is made; a long edit brief is a few hundred
TEXT_TOKENS_RESERVED = 1024
# the ViT decoder: one token per 16x16 pixels, a 16384-wide feed-forward and full attention over them
DECODE_BYTES_PER_TOKEN = 96 * 1024
DECODE_FIXED_BYTES = 256 * 1024 * 1024


def sampling_tokens(generation: "Generation") -> int:
    """The DiT's sequence length for a generation, the prompt's share estimated."""
    shapes = generation.shapes
    _, _, t, h, w = shapes.video
    rows = t * ((h + 1) // 2) * ((w + 1) // 2) + shapes.audio[2] * shapes.audio[3]
    rows += sum(((r["latent_h"] + 1) // 2) * ((r["latent_w"] + 1) // 2) for r in generation.refs)
    return rows + TEXT_TOKENS_RESERVED


def sampling_memory(generation: "Generation") -> int:
    return sampling_tokens(generation) * SAMPLING_BYTES_PER_TOKEN + SAMPLING_FIXED_BYTES


def decode_memory(shapes) -> int:
    _, _, t, h, w = shapes.video
    return t * h * w * DECODE_BYTES_PER_TOKEN + DECODE_FIXED_BYTES


def finite(tensor: torch.Tensor) -> bool:
    """No NaN or Inf: max and min carry them through, without a mask the size of the picture (one sync per call)."""
    return bool(torch.isfinite(tensor.amax()) and torch.isfinite(tensor.amin()))


def vision_copy(image: torch.Tensor) -> torch.Tensor:
    """A (1, h, w, 3) picture scaled within VISION_PIXELS, aspect kept, sides on the 32 grid, never enlarged."""
    from ..pictures import VISION_PIXELS, within_area
    height, width = image.shape[1], image.shape[2]
    w, h = within_area(width, height, VISION_PIXELS)
    if (w, h) == (width, height):
        return image
    resized = torch.nn.functional.interpolate(image.movedim(-1, 1), size=(h, w), mode="bicubic", antialias=True, align_corners=False)
    return resized.movedim(1, -1).clamp(0.0, 1.0)


def _vae_dtype() -> torch.dtype:
    # ComfyUI runs the H3 VAE in fp16 or fp32 only
    if args.fp32_vae:
        return torch.float32
    if memory_management.should_use_fp16(memory_management.vae_device()):
        return torch.float16
    return torch.float32


class MiniMaxH3Engine(ForgeDiffusionEngine):
    matched_guesses = [MiniMaxH3]

    def __init__(self, estimated_config, huggingface_components):
        super().__init__(estimated_config, huggingface_components)

        clip = CLIP(model_dict={"qwen3vl_32b": huggingface_components["text_encoder"]},
                    tokenizer_dict={"qwen3vl_32b": huggingface_components["tokenizer"]})

        vae = VAE(model=huggingface_components["vae"], dtype=_vae_dtype(), is_wan=True)
        vae.latent_channels = 24

        unet = UnetPatcher.from_model(model=huggingface_components["transformer"], diffusers_scheduler=None,
                                      k_predictor=self._h3_predictor(estimated_config), config=estimated_config)

        self.text_processing_engine_h3 = MiniMaxH3TextEngine(text_encoder=clip.cond_stage_model.qwen3vl_32b,
                                                             tokenizer=clip.tokenizer.qwen3vl_32b)

        self.forge_objects = ForgeObjects(unet=unet, clip=clip, vae=vae, clipvision=None)
        self.forge_objects_original = self.forge_objects.shallow_copy()
        self.forge_objects_after_applying_lora = self.forge_objects.shallow_copy()

        self.is_h3_image = True
        # Forge's Shift slider sets the flow shift of both streams (P-H3-Image; MiniMax-H3-Image is on its own grid)
        self.use_shift = True
        self.shift = SHIFT
        self.generation: Generation | None = None
        # reference pictures, (1, h, w, 3) in [0, 1] at their own size, in "<Picture i>" order; set by the script for
        # every generation
        self.references: list[torch.Tensor] = []
        # standard img2img: Forge's input image is encoded and noised at Denoising strength (the [H3] Enable
        # Reference setting off); otherwise the input image is <Picture 1> and encode_first_stage is a placeholder
        self.img2img_input = False
        # the [H3] Output scale setting: the fraction of each side of the decoded picture that is kept (area filter)
        self.output_scale = Fraction(1)

    def _h3_predictor(self, estimated_config):
        """MiniMax-H3-Image samples on the sigma grid its output heads were distilled on; the others on Forge's
        shifted flow schedule."""
        config = getattr(estimated_config, "unet_config", None) or {}
        if config.get("pdd_steps", 1) > 1 and config.get("pdd_sigmas"):
            from .sampling import PredictionH3Image
            return PredictionH3Image(estimated_config, config["pdd_sigmas"])
        return self._get_predictor()

    @property
    def has_audio(self) -> bool:
        """Whether the checkpoint has H3's audio stream (MiniMax-H3-Image has none): the packed latent then carries
        the two-step audio stub next to the picture."""
        return getattr(self.forge_objects.unet.model.diffusion_model, "audio_latents_dim", 32) is not None

    def set_references(self, references, vision=None, picture_label: str = PICTURE_LABEL) -> None:
        """Called by the script for every generation, before Forge encodes the prompt: (1, h, w, 3) pictures in
        [0, 1] as the VAE takes them, in "<Picture i>" order, the copies the text encoder sees (made within 1 MP
        when not given) and the label the prompt gives each."""
        self.references = list(references or ())
        self.vision_pictures = [vision_copy(image) for image in self.references] if vision is None else list(vision)
        self.picture_label = picture_label

    def set_img2img(self, standard: bool) -> None:
        """Called by the script for every generation: whether Forge's img2img input image starts the sampling."""
        self.img2img_input = bool(standard)

    def set_output_scale(self, scale) -> None:
        """Called by the script for every generation: 1 keeps the decoded picture; 3/4, 2/3, 1/2 or 1/4 shrinks each
        side to that fraction with an exact area filter (resize.area_resize), which averages away the pixel-level
        texture the 16x decoder invents. A side that does not divide is rounded to the nearest pixel (2/3 of 4096 is
        2731)."""
        self.output_scale = min(Fraction(scale), Fraction(1))

    def set_resolution_shift(self, factor: float) -> bool:
        """Called by the script for every generation, before Forge makes the schedule: move MiniMax-H3-Image's sigma
        grid by an odds factor (resolution_shift.odds_factor, 1 for the stock grid), both for the schedule and for
        picking the output heads, so each head still runs at its own step. Returns whether the model has the grid."""
        # sampling.PredictionH3Image, the only predictor with a grid to move
        predictor = getattr(self.forge_objects.unet.model, "predictor", None)
        if not hasattr(predictor, "set_odds_factor"):
            return False
        predictor.set_odds_factor(factor)
        self.forge_objects.unet.model.diffusion_model.pdd_sigmas = predictor.grid
        return True

    def condition_images(self) -> list[torch.Tensor]:
        """The pictures the text encoder sees before the prompt as "<Picture i>" (ComfyUI's vision copies)."""
        return list(getattr(self, "vision_pictures", ()))

    def set_shift(self, shift, *args, **kwargs):
        # Forge Neo after d70373e also passes the size (width, height); H3's shift does not depend on it
        shift = float(shift) if shift and shift > 0 else SHIFT
        super().set_shift(shift)
        self.shift = shift

    def prepare(self, width: int, height: int, seed: int) -> tuple[int, ...]:
        """Start a generation: remember its shapes; returns the shape of the packed latent (without the batch)."""
        shapes = still_shapes(width, height, audio=self.has_audio)
        # each reference on its own grid (ComfyUI MiniMaxH3ReferenceToVideo's ref_blocks)
        refs = [{"kind": "image", "latent_h": image.shape[1] // 16, "latent_w": image.shape[2] // 16, "latent": latent}
                for image, latent in zip(self.references, self._encode_images(self.references))]
        self.generation = Generation(shapes=shapes, seed=seed, refs=refs,
                                     vision_spans=list(self.text_processing_engine_h3.vision_spans))
        self.forge_objects.unet.model.diffusion_model.generation = self.generation
        self._reserve_working_memory()
        return (1, 1, shapes.size)

    def _reserve_working_memory(self) -> None:
        """Forge keeps this much VRAM free when it loads the DiT for sampling, on this generation's copy of the
        UNet patcher (Forge takes a fresh copy for every generation)."""
        unet = self.forge_objects.unet.clone()
        reserved = sampling_memory(self.generation)
        unet.add_extra_preserved_memory_during_sampling(reserved)
        self.forge_objects.unet = unet
        print(f"[MiniMax H3] keeping {reserved / 2**30:.1f} GiB of VRAM free for sampling "
              f"({sampling_tokens(self.generation)} tokens at most)")

    @torch.inference_mode()
    def _encode_images(self, images: list[torch.Tensor]) -> list[torch.Tensor]:
        # each picture on its own, one latent frame [1, 24, 1, h/16, w/16] (ComfyUI vae.encode of one image)
        if not images:
            return []
        vae = self.forge_objects.vae
        # the largest picture's working memory too: with the text encoder and the DiT resident, a 1536x1024 sheet left
        # the VAE's attention to fall back to slices
        largest = max(images, key=lambda image: image.shape[1] * image.shape[2])
        memory_management.load_models_gpu([vae.patcher], memory_required=encode_memory(largest.shape[:3], vae.vae_dtype))
        latents = []
        for image in images:
            pixels = image.movedim(-1, 1).unsqueeze(2).mul(2.0).sub(1.0)  # [1, 3, 1, h, w] in [-1, 1]
            latent = vae.first_stage_model.encode(pixels.to(vae.device, vae.vae_dtype))
            latents.append(latent.float().cpu())
        return latents

    def release_generation(self) -> None:
        """Drop generation data and the temporary adapter weights after saving."""
        from .lora import release_cached_terms

        self.generation = None
        model = self.forge_objects.unet.model.diffusion_model
        model.generation = None
        release_cached_terms(model)

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: list[str]):
        raise_pending_error()
        images = self.condition_images()
        # room for the activations too: a picture's vision tokens did not fit next to the resident DiT
        pictures = [tuple(image.shape[-3:-1]) for image in images]
        memory_management.load_models_gpu([self.forge_objects.clip.patcher], memory_required=prompt_memory(pictures))
        # the same pictures for the prompt and the negative prompt: they come before either text
        return self.text_processing_engine_h3(prompt, images=images, picture_label=getattr(self, "picture_label", PICTURE_LABEL))

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt: str) -> tuple[int, int]:
        token_count = len(self.text_processing_engine_h3.tokenize(prompt))
        return token_count, max(999, token_count)

    @torch.inference_mode()
    def encode_first_stage(self, x: torch.Tensor):
        """Forge's img2img init hands over the input image, resized to the output, [B, 3, H, W] in [-1, 1]. Standard
        img2img encodes it into the packed latent (the audio stub at zero, as ComfyUI's empty latent) for Forge to
        noise at Denoising strength; an edit takes the picture elsewhere and gets a placeholder here."""
        raise_pending_error()
        if not getattr(self, "img2img_input", False):
            return torch.zeros((1, 1, 1, 1), device=x.device)
        height, width = x.shape[-2], x.shape[-1]
        shapes = still_shapes(width, height, audio=self.has_audio)
        vae = self.forge_objects.vae
        memory_management.load_models_gpu([vae.patcher], memory_required=encode_memory((1, 3, 1, height, width), vae.vae_dtype))
        pixels = x[:1].unsqueeze(2).to(vae.device, vae.vae_dtype)  # [1, 3, 1, H, W]
        video = vae.first_stage_model.encode(pixels).float()
        if tuple(video.shape) != shapes.video:
            raise H3Error(f"The H3 VAE encoded the input image to {tuple(video.shape)}, not {shapes.video}")
        audio = torch.zeros(shapes.audio, dtype=video.dtype, device=video.device)
        return shapes.pack(video, audio).to(x.device)

    @torch.inference_mode()
    def decode_first_stage(self, x: torch.Tensor):
        generation = self.generation
        if generation is None or x.shape[-1] != generation.shapes.size:
            raise RuntimeError("[MiniMax H3] the latent does not belong to the current H3 generation")
        if not finite(x[:1]):
            raise H3Error("H3 sampling produced NaN or Inf values: the picture would come out black or as noise. Try "
                          "another sampler, scheduler or attention option.")
        video, _ = generation.shapes.unpack(x[:1].float())

        vae = self.forge_objects.vae
        # room for the decoder's activations too, so the DiT or the text encoder make way when they must
        memory_management.load_models_gpu([vae.patcher], memory_required=decode_memory(generation.shapes))
        pixels = vae.first_stage_model.decode(video.to(vae.device, vae.vae_dtype))  # [1, 3, 1, H, W] in [0, 1]
        if not finite(pixels):
            advice = " Start Forge Neo with --fp32-vae." if vae.vae_dtype != torch.float32 else ""
            raise H3Error(f"The H3 VAE produced NaN or Inf values: the picture would come out black.{advice}")
        frame = pixels[:, :, 0].float()
        if self.output_scale < 1:
            frame = area_resize(frame, self.output_scale)
        generation.frame = frame.cpu()
        return generation.frame.mul(2.0).sub(1.0)
