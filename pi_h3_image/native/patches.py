"""Runtime hooks into Forge Neo's loader and sampler.

Every hook hands anything that is not MiniMax H3 to the original Forge Neo function untouched.
"""

import contextlib
import functools
import json
import logging
import os
from contextvars import ContextVar
from types import SimpleNamespace

import torch
from backend import loader, memory_management
from backend.nn import krea
from backend.nn.llm import llama
from backend.operations import using_forge_operations
from backend.sampling import condition
from backend.state_dict import load_state_dict
from huggingface_guess import detection, model_list
from transformers.modeling_utils import no_init_weights

from ..contracts import H3Error
from ..models import PH3_IMAGE, inspect_model, is_image_checkpoint, is_pruna_name
from . import filebacked, gguf_vision, model, release, vae
from .engine import MiniMaxH3Engine
from .image_vae import H3ImageVAE, MiniMaxH3ImageVAE, PH3ImageVAE
from .islands import floating_dtype, fp32_islands, restore_fp32
from .text_encoder import PH3ImageQwen3VL, Qwen3VL32B
from .transformer import MiniMaxH3Model

logger = logging.getLogger("forge_h3")

# the vision tower marks a Qwen3-VL file (Comfy-Org's files start at "visual.", Pruna's at "model.visual."); H3's
# 32B encoder is truncated after layer 50, P-H3-Image's 4B one carries Pruna's text connector
TE_VISION_KEY = "visual.deepstack_merger_list.0.norm.weight"
TE_32B_KEYS = ("model.layers.49.self_attn.q_proj.weight", "model.language_model.layers.49.self_attn.q_proj.weight")
TE_CONNECTOR_PREFIX = "text_connector."
TE_PREFIX = "qwen3vl_32b.transformer."

MISSING_TE = ("MiniMax H3 needs its text encoder: select a qwen3vl_32b_minimax_h3 or a p-h3-image_text_encoder file "
              "under VAE / Text Encoder")

_applied = False
_sampling_h3 = False
_loading_pruna = ContextVar("h3_image_loading_pruna", default=False)


@contextlib.contextmanager
def _swapped(module, name: str, value):
    # the loader imports these classes inside the function, so a temporary swap routes one call to our class
    original = getattr(module, name)
    setattr(module, name, value)
    try:
        yield
    finally:
        setattr(module, name, original)


def begin_sampling() -> None:
    """While H3 samples, prompt and negative prompt never share a batch (see _hook_conditions)."""
    global _sampling_h3
    _sampling_h3 = True


def end_sampling() -> None:
    global _sampling_h3
    _sampling_h3 = False


@contextlib.contextmanager
def _file_dtype_kept(state_dict: dict):
    """A bf16 text encoder file stays bf16, as ComfyUI runs it (llama_detect reads the file's dtype): Forge's default
    casts every plain text encoder to fp16, which costs a 9 GB copy in RAM for the P-H3-Image file and leaves the
    file-backed views behind. A text encoder dtype flag on Forge's command line still wins."""
    from backend.args import args
    dtype = floating_dtype(state_dict)
    flags = ("fp32_text_enc", "bf16_text_enc", "fp16_text_enc", "fp8_e4m3fn_text_enc", "fp8_e5m2_text_enc")
    if dtype != torch.bfloat16 or any(getattr(args, flag, False) for flag in flags):
        yield
        return
    with _swapped(memory_management, "text_encoder_dtype", lambda device=None: torch.bfloat16):
        yield


def text_encoder_class(asd: dict):
    """The encoder class a text encoder state dict needs, None for any other file."""
    if not any(k.endswith(TE_VISION_KEY) for k in asd):
        return None
    if any(k.startswith(TE_CONNECTOR_PREFIX) for k in asd):
        return PH3ImageQwen3VL
    if any(k in asd for k in TE_32B_KEYS):
        return Qwen3VL32B
    return None


def _is_h3_text_encoder(asd: dict) -> bool:
    return text_encoder_class(asd) is not None


def _hook_detection() -> None:
    original = detection.detect_unet_config

    def detect_unet_config(state_dict, key_prefix, *args, **kwargs):
        dit_config = model.detect(state_dict, key_prefix, pruna=_loading_pruna.get())
        if dit_config is not None:
            return dit_config
        return original(state_dict, key_prefix, *args, **kwargs)

    detection.detect_unet_config = detect_unet_config


def _hook_metadata() -> None:
    # Forge reads a checkpoint's metadata but detects the model from its tensors alone; MiniMax-H3-Image keeps the
    # sigma grid of its output heads in the metadata ("config": {"transformer": {"pdd_steps", "pdd_sigmas"}})
    original = loader._load_unet

    @functools.wraps(original)
    def _load_unet(path):
        pruna = False
        if is_pruna_name(path):
            try:
                item = inspect_model(path)
                pruna = is_image_checkpoint(item) and item.variant == PH3_IMAGE
            except (H3Error, OSError, ValueError):
                pass
        token = _loading_pruna.set(pruna)
        try:
            sd, metadata, guess = original(path)
        finally:
            _loading_pruna.reset(token)
        if isinstance(guess, model.MiniMaxH3) and metadata and "config" in metadata:
            try:
                guess.unet_config.update(json.loads(metadata["config"]).get("transformer", {}))
            except (ValueError, AttributeError, TypeError) as e:
                logger.warning(f"[MiniMax H3] could not read the checkpoint's config metadata: {e}")
        return sd, metadata, guess

    loader._load_unet = _load_unet


def _replace(sd: dict, prefix: str, asd: dict) -> None:
    for k in [k for k in sd if k.startswith(prefix)]:
        del sd[k]
    for k, v in asd.items():
        sd[prefix + k] = v


def _hook_replace_state_dict() -> None:
    original = loader.replace_state_dict

    def replace_state_dict(sd: dict, asd: dict, guess, path):
        if not isinstance(guess, model.MiniMaxH3):
            return original(sd, asd, guess, path)
        if _is_h3_text_encoder(asd):
            _replace(sd, f"{guess.text_encoder_key_prefix[0]}{TE_PREFIX}", asd)
        elif vae.is_h3_vae(asd):
            _replace(sd, guess.vae_key_prefix[0], vae.with_latent_stats(asd))
        else:
            return original(sd, asd, guess, path)
        return sd

    loader.replace_state_dict = replace_state_dict


def _load_vae(state_dict: dict) -> H3ImageVAE:
    if not isinstance(state_dict, dict) or not vae.is_h3_vae(state_dict):
        raise ValueError(vae.MISSING_VAE)
    if not vae.is_image_vae(state_dict):
        raise ValueError(vae.VIDEO_VAE)
    layers = vae.decoder_layers(state_dict)
    # one frame per latent: MiniMax-H3-Image's with a 2D decoder, P-H3-Image's at a fixed time coordinate
    cls = MiniMaxH3ImageVAE if vae.is_h3_image_vae(state_dict) else PH3ImageVAE
    with no_init_weights():
        if vae.is_quantized(state_dict):
            # an int8 VAE (Pruna's): Forge's mixed-precision operations read each layer's comfy_quant
            with using_forge_operations(device=memory_management.cpu, dtype=torch.float32, manual_cast_enabled=True,
                                        extra_dtype={"mixed_ops": True, "TE": False}):
                autoencoder = cls(num_layers=layers)
        else:
            with using_forge_operations(device=memory_management.cpu, dtype=torch.float32, extra_dtype="vae"):
                autoencoder = cls(num_layers=layers)
    # read by Forge's VAE patcher: z_dim for the 3D path (the latent keeps a frame axis of 1)
    autoencoder.config = SimpleNamespace(z_dim=24)
    load_state_dict(autoencoder, state_dict, log_name="MiniMax H3 VAE")
    return autoencoder


def _file_backed(name: str, module, storages: set[int]) -> None:
    # see filebacked.py: weights that return from VRAM become file views again instead of anonymous copies
    try:
        backed, total = filebacked.keep_file_backed(module, storages)
    except Exception as e:
        # the model works without it, with Forge's own copies in system RAM
        logger.warning(f"[MiniMax H3] {name}: weights will be copied to system RAM when they leave VRAM: {e}")
        return
    print(f"[MiniMax H3] {name}: {backed / 2**30:.1f} of {total / 2**30:.1f} GiB stay file-backed in system RAM")


def _hook_components() -> None:
    original = loader.load_huggingface_component

    # wraps leaves __wrapped__ behind, which inspect.getsource follows: the compat checks of the other extensions
    # that read Forge's loader source still reach it through this hook
    @functools.wraps(original)
    def load_huggingface_component(guess, component_name, lib_name, cls_name, repo_path, state_dict):
        if not isinstance(guess, model.MiniMaxH3):
            return original(guess, component_name, lib_name, cls_name, repo_path, state_dict)

        if cls_name == "AutoencoderMiniMaxH3":
            return _load_vae(state_dict)
        if cls_name == "MiniMaxH3Transformer3DModel":
            # the Krea 2 branch is a generic single-stream DiT load: dtype, quantization and device handling
            islands = fp32_islands(state_dict, curve="adaln_t_table" in state_dict)
            storages = filebacked.file_storages(state_dict)
            with _swapped(krea, "SingleStreamDiT", MiniMaxH3Model):
                dit = original(guess, component_name, lib_name, "Krea2Transformer2DModel", repo_path, state_dict)
            restore_fp32(dit, islands)
            _file_backed("diffusion model", dit, storages)
            return dit
        if cls_name == "Qwen3VLModel":
            if not isinstance(state_dict, dict) or len(state_dict) <= 16:
                raise ValueError(MISSING_TE)
            cls = text_encoder_class(state_dict)
            if cls is None:
                raise ValueError(MISSING_TE)
            gguf = gguf_vision.is_gguf(state_dict)
            if gguf:
                from backend.loader_gguf import dequantize

                gguf_vision.plain_vision(state_dict, dequantize)
            # Pruna's text connector is fp32 and never quantized; the loader stores every plain layer at one dtype
            connector = {k: v for k, v in state_dict.items() if k.startswith(TE_CONNECTOR_PREFIX)}
            storages = filebacked.file_storages(state_dict)
            with _swapped(llama, "Qwen3VL", cls), _file_dtype_kept(state_dict):
                text_encoder = original(guess, component_name, lib_name, cls_name, repo_path, state_dict)
            if gguf:
                gguf_vision.cast_to_input(text_encoder.visual)
            if connector:
                restore_fp32(text_encoder, connector)
            _file_backed("text encoder", text_encoder, storages)
            return text_encoder

        return original(guess, component_name, lib_name, cls_name, repo_path, state_dict)

    loader.load_huggingface_component = load_huggingface_component


def _hook_conditions() -> None:
    # Forge batches the prompt with the negative prompt, repeating the shorter one up to a common length; H3 packs
    # the text into its sequence, so a repeated prompt would be a different prompt. ComfyUI runs them apart.
    original = condition.ConditionCrossAttn.can_concat

    def can_concat(self, other):
        if _sampling_h3:
            return False
        return original(self, other)

    condition.ConditionCrossAttn.can_concat = can_concat


def _hook_unload() -> None:
    # see release.py: a replaced H3 model is emptied before Forge detaches it to system RAM
    original = memory_management.unload_all_models
    types = release.h3_module_types()

    def unload_all_models(*args, **kwargs):
        try:
            from modules import sd_models
            if release.release_replaced(memory_management.current_loaded_models, sd_models.model_data.sd_model, types):
                print("[MiniMax H3] released the replaced H3 model without copying it to system RAM")
        except Exception as e:
            logger.warning(f"[MiniMax H3] could not release the replaced H3 model early: {e}")
        return original(*args, **kwargs)

    memory_management.unload_all_models = unload_all_models

    from modules import processing, sd_models
    original_manage = processing.manage_model_and_prompt_cache

    def manage_model_and_prompt_cache(p):
        if release.reload_instead_of_unload(processing.need_global_unload, sd_models.model_data.sd_model):
            # an emptied hash makes forge_model_reload discard H3 (see above) and load it again from disk
            print("[MiniMax H3] settings changed: reloading H3 from disk instead of moving it to system RAM")
            sd_models.model_data.forge_hash = ""
        return original_manage(p)

    processing.manage_model_and_prompt_cache = manage_model_and_prompt_cache


def _hook_taesd() -> None:
    # Forge's TAESD live preview asks decoder_model() for a decoder; H3's packed latent has none, so Forge falls back
    # to its RGB preview (the latent_rgb_factors of model.py, on the picture's latent frame)
    from modules import sd_vae_taesd, shared
    original = sd_vae_taesd.decoder_model

    def decoder_model():
        if getattr(shared.sd_model, "is_h3_image", False):
            return None
        return original()

    sd_vae_taesd.decoder_model = decoder_model

    # with Settings > VAE set to TAESD, Forge would encode with an SD encoder and decode the packed latent with an
    # SD decoder, so H3 always uses its full VAE
    original_encoder = sd_vae_taesd.encoder_model

    def encoder_model():
        if getattr(shared.sd_model, "is_h3_image", False):
            return None
        return original_encoder()

    sd_vae_taesd.encoder_model = encoder_model

    from modules import processing, sd_samplers_common
    warned = []

    def decode_first_stage(model, x):
        if not getattr(model, "is_h3_image", False):
            return original_decode(model, x)
        if shared.opts.sd_vae_decode_method != "Full" and not warned:
            warned.append(True)
            print("[MiniMax H3] Settings > VAE > VAE for Decoding is TAESD: H3 decodes the picture with its full VAE")
        return sd_samplers_common.samples_to_images_tensor(x, 0, model)

    original_decode = sd_samplers_common.decode_first_stage
    sd_samplers_common.decode_first_stage = decode_first_stage
    # processing imported the function by name
    processing.decode_first_stage = decode_first_stage


def apply() -> None:
    global _applied
    if _applied:
        return

    _hook_detection()
    try:
        _hook_metadata()
    except Exception as e:
        # MiniMax-H3-Image then samples on its published sigma grid (profiles.H3_IMAGE_PDD_SIGMAS)
        logger.warning(f"[MiniMax H3] could not read checkpoint metadata through the loader: {e}")
    if model.MiniMaxH3 not in model_list.models:
        model_list.models.append(model.MiniMaxH3)
    if MiniMaxH3Engine not in loader.possible_models:
        loader.possible_models = (*loader.possible_models, MiniMaxH3Engine)
    _hook_replace_state_dict()
    _hook_components()
    _hook_conditions()
    _hook_unload()
    try:
        _hook_taesd()
    except Exception as e:
        # the model works without it: the RGB preview stays
        logger.warning(f"[MiniMax H3] could not route the live preview and the VAE settings: {e}")

    try:
        from . import presets

        presets.register_sampler()
    except Exception as e:
        # MiniMax-H3-Image still runs on Forge's own LCM, at a fixed noise scale
        logger.warning(f"[MiniMax H3] could not add the H3-Image LCM sampler: {e}")
    try:
        from . import presets

        presets.register_scheduler()
    except Exception as e:
        # the models still run on Forge's own schedules, none of which ends where theirs do
        logger.warning(f"[MiniMax H3] could not add the P-H3-Image and H3-Image schedules: {e}")
    try:
        from . import presets

        presets.register()
    except Exception as e:
        # the model still works without the presets: the panel applies each profile's sampler, schedule and CFG
        logger.warning(f"[MiniMax H3] could not add the ph3 and h3image UI presets: {e}")

    _applied = True
    print(f"[MiniMax H3] native backend enabled (torch {torch.__version__}, {os.path.basename(model.CONFIG_DIR)})")
