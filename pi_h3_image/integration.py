"""H3 inside Forge's own txt2img flow, through script callbacks.

Forge loads the model (native/patches.py registers it), samples and decodes as usual. The callbacks here check the
request, give the sampler the packed video+audio noise of one still image, and let Forge save the picture.
"""

from fractions import Fraction
from pathlib import Path

from . import pictures
from .contracts import GenerationRequest, H3Error, set_pending_error
from .models import inspect_model, is_image_checkpoint, resolve_components
from .resolution_shift import odds_factor

# StableDiffusionProcessing.distilled_cfg_scale when a request does not set it
API_DEFAULT_DISTILLED_CFG = 3.5
# img2img Resize mode "Just resize (latent upscale)": it would interpolate the placeholder latent
LATENT_UPSCALE = 3


def checkpoint_info(value):
    # the same lookup Forge applies to a checkpoint override: alias, then title substring, with or without the hash
    from modules import sd_models
    return sd_models.get_closet_checkpoint_match(value)


def select_h3(p):
    """The selected checkpoint when it is an H3 diffusion model; read from its header, before Forge loads it."""
    from modules import shared
    overrides = getattr(p, "override_settings", {})
    info = checkpoint_info(overrides.get("sd_model_checkpoint", shared.opts.sd_model_checkpoint))
    if info is None:
        return None
    item = inspect_model(info.filename)
    return info if is_image_checkpoint(item) else None


def module_paths(values):
    from modules_forge import main_entry
    result = []
    for value in values or []:
        # Forge Neo keys its list by file name; with "Show filenames without folder" off the dropdown sends the path
        # under models/VAE or models/text_encoder, which Forge itself reduces to the file name (modules_change)
        path = main_entry.module_list.get(value) or main_entry.module_list.get(Path(value).name, value)
        if not Path(path).is_file():
            raise H3Error(f"Selected H3 component is unavailable: {Path(value).name}. Refresh the model list.")
        result.append(str(Path(path).resolve()))
    return result


def validate_processing(p):
    if getattr(p, "batch_size", 1) != 1:
        raise H3Error("H3 generates one picture at a time: set Batch Size to 1 and use Batch Count.")
    if getattr(p, "enable_hr", False) or getattr(p, "txt2img_upscale", False):
        raise H3Error("Disable Hires. fix for H3 generation.")
    if getattr(p, "restore_faces", False):
        raise H3Error("Disable Restore faces for H3 generation.")
    if getattr(p, "image_mask", None) is not None:
        raise H3Error("H3 inpainting is not implemented.")
    if (getattr(p, "subseed_strength", 0) > 0 or getattr(p, "seed_resize_from_w", -1) > 0
            or getattr(p, "seed_resize_from_h", -1) > 0):
        raise H3Error("Disable variation seed and seed resize for H3.")
    # Scripts (X/Y/Z plot, Prompts from file and the like) are allowed: they run Forge's processing once per picture,
    # and each run passes the checks above


def before_process(p):
    """Before Forge loads the model: check the request and set up the H3 generation."""
    p.pi_h3_image_request = None
    set_pending_error(None)
    try:
        _before_process(p)
    except H3Error as error:
        set_pending_error(error)
        raise


def script_before_process(p):
    """before_process as Forge's script runner calls it: a rejected request prints one line instead of the traceback
    Forge logs for any exception; the error stays pending and stops the generation when the model is first called."""
    try:
        before_process(p)
    except H3Error as error:
        print(f"[MiniMax H3] {error}")


REFERENCE_SETTING = "h3_do_reference"
OUTPUT_SCALE_SETTING = "h3_output_scale"
RESOLUTION_SHIFT_SETTING = "h3_resolution_shift"
# the setting's choices and the fraction of each side that is kept
OUTPUT_SCALES = {"1": Fraction(1), "3/4": Fraction(3, 4), "2/3": Fraction(2, 3), "1/2": Fraction(1, 2),
                 "1/4": Fraction(1, 4)}


def reference_mode(p=None) -> bool:
    """The [H3] Enable Reference setting (Forge's klein_do_reference and the like): on, img2img is an edit with the
    input image as <Picture 1>; off, img2img is Forge's standard one, the input image noised at Denoising strength."""
    from modules import shared
    overrides = getattr(p, "override_settings", {}) or {}
    return bool(overrides.get(REFERENCE_SETTING, getattr(shared.opts, REFERENCE_SETTING, True)))


def output_scale() -> tuple[str, Fraction]:
    """The [H3] Output scale setting: its label and the fraction of each side of the decoded picture that is kept."""
    from modules import shared
    label = str(getattr(shared.opts, OUTPUT_SCALE_SETTING, "1"))
    return (label, OUTPUT_SCALES[label]) if label in OUTPUT_SCALES else ("1", Fraction(1))


def resolution_shift() -> float:
    """The [H3] Resolution shift setting: the strength of the grid's move with picture size, 0 (off) to 1."""
    from modules import shared
    try:
        strength = float(getattr(shared.opts, RESOLUTION_SHIFT_SETTING, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return min(max(strength, 0.0), 1.0)


def validate_img2img(p, components, reference):
    if not getattr(p, "init_images", None):
        role = "<Picture 1>, the picture to edit" if reference else "the picture to start from"
        raise H3Error(f"Add an input image for H3 in img2img: it becomes {role}.")
    if getattr(p, "resize_mode", 0) == LATENT_UPSCALE:
        raise H3Error("H3 does not take the latent upscale resize mode. Choose another Resize mode.")


_precision_noted = False


def note_precision_flags(components=None):
    """Once per session: where Forge's launch flags differ from the models' ComfyUI reference. A checkpoint that
    declares INT8 attention for its blocks (MiniMax-H3-Image) runs it in ComfyUI by default, and in Forge Neo only
    with --use-ck-attention; for the others the flag runs below the reference's precision."""
    global _precision_noted
    if _precision_noted:
        return
    _precision_noted = True
    try:
        from backend.args import args
    except ImportError:
        return
    notes = []
    declared = components is not None and components.wants_int8_attention
    if declared and not getattr(args, "use_ck_attention", False):
        print(f"[MiniMax H3] note: {components.dit.path.name} asks for comfy-kitchen's INT8 attention, which ComfyUI "
              "then uses; Forge Neo runs it with --use-ck-attention (faster, less VRAM at large sizes).")
    if getattr(args, "use_ck_attention", False) and not declared:
        notes.append("--use-ck-attention runs INT8 attention in the DiT and the VAE decoder")
    if getattr(args, "fast_fp16", False):
        notes.append("--fast-fp16 accumulates fp16 matmuls in fp16 (the VAE decoder, a fp16 text encoder)")
    if notes:
        print("[MiniMax H3] note: " + "; ".join(notes) + ". The reference runs full precision there: compare without "
              "the flag if small details come out garbled.")


def _before_process(p):
    p.pi_h3_image_references = []
    info = select_h3(p)
    if info is None:
        return
    from modules import processing, shared
    validate_processing(p)
    overrides = getattr(p, "override_settings", {})
    components = resolve_components(info.filename, module_paths(overrides.get("forge_additional_modules", shared.opts.forge_additional_modules)))
    note_precision_flags(components)
    profile = components.profile
    from .dimensions import validate_dimensions
    validate_dimensions(profile.name, p.width, p.height)
    refs, vision = [], []
    p.pi_h3_image_standard_img2img = False
    detail = ""
    if isinstance(p, processing.StableDiffusionProcessingImg2Img):
        reference = reference_mode(p)
        validate_img2img(p, components, reference)
        if reference:
            # the original pictures, sized for the VAE and for the text encoder as the model's ComfyUI node does
            images = pictures.collect(p, profile.max_pictures, profile.label)
            refs, vision = pictures.edit_inputs(profile, images, p.width, p.height)
            # the model generates the whole picture from noise, conditioned on <Picture 1>: the input image is never noised
            p.denoising_strength = 1.0
            if profile.edit_prefix:
                p.prompt = pictures.edit_prompt(p.prompt)
            detail = f", editing <Picture 1> with {len(refs)} pictures"
        else:
            # Forge's own img2img: the input image is encoded (engine.encode_first_stage) and noised at Denoising strength
            p.pi_h3_image_standard_img2img = True
            detail = f", img2img at denoising strength {p.denoising_strength:g}"
    request = GenerationRequest(width=p.width, height=p.height, references=len(refs))
    p.pi_h3_image_request = request
    p.pi_h3_image_components = components
    p.pi_h3_image_references = refs
    p.pi_h3_image_vision = vision
    experimental = f" (experimental: {components.experimental})" if components.experimental else ""
    print(f"[MiniMax H3] {profile.label}{experimental}: still image at {request.width}x{request.height}{detail}")


def process(p):
    request = getattr(p, "pi_h3_image_request", None)
    if request is None:
        return
    if not getattr(p.sd_model, "is_h3_image", False):
        raise H3Error("Forge did not load the H3 model. Check the VAE / Text Encoder selection and the console.")
    from modules import shared

    engine = p.sd_model
    refs = getattr(p, "pi_h3_image_references", [])
    # Forge caches the conditioning by prompt, which knows nothing of the pictures before it
    if refs or engine.condition_images():
        p.clear_prompt_cache()
    profile = p.pi_h3_image_components.profile
    engine.set_references(refs, getattr(p, "pi_h3_image_vision", None) if refs else None, profile.picture_label)
    engine.set_img2img(getattr(p, "pi_h3_image_standard_img2img", False))
    scale_label, scale = output_scale()
    engine.set_output_scale(scale)
    if scale < 1:
        p.extra_generation_params["H3 Output scale"] = scale_label
    strength = resolution_shift()
    factor = odds_factor(p.width, p.height, strength)
    if engine.set_resolution_shift(factor) and strength > 0:
        p.extra_generation_params["H3 Resolution shift"] = f"{strength:g} (x{factor:.2f})"
    if request.references:
        p.extra_generation_params["H3 Edit pictures"] = request.references

    p.extra_generation_params["H3 Model"] = profile.label
    if p.pi_h3_image_components.experimental:
        p.extra_generation_params["H3 Experimental"] = p.pi_h3_image_components.experimental
    if getattr(shared.opts, "forge_preset", None) != profile.name:
        # outside the profile's preset the slider is another model's Shift or Distilled CFG; keep the profile's own
        # shift (and infotext)
        p.distilled_cfg_scale = profile.shift
    elif getattr(p, "is_api", False) and p.distilled_cfg_scale == API_DEFAULT_DISTILLED_CFG:
        # an API request that leaves distilled_cfg_scale out gets Forge's Flux default; use the preset's Shift
        p.distilled_cfg_scale = getattr(shared.opts, f"{profile.name}_t2i_dcfg", profile.shift)


def before_sampling(p, noise):
    from .native import patches
    request = getattr(p, "pi_h3_image_request", None)
    if request is None:
        return
    from modules import rng
    shape = p.sd_model.prepare(request.width, request.height, int(p.seeds[0]))
    # Forge made p.rng for an image latent; the samplers that add noise on the way (ancestral, SDE, res_multistep)
    # draw from it too, so it has to give the packed shape
    p.rng = rng.ImageRNG(shape, p.seeds, subseeds=p.subseeds, subseed_strength=p.subseed_strength,
                         seed_resize_from_h=p.seed_resize_from_h, seed_resize_from_w=p.seed_resize_from_w)
    p.modified_noise = p.rng.next().to(device=noise.device, dtype=noise.dtype)
    if getattr(p, "init_latent", None) is not None:
        if getattr(p, "pi_h3_image_standard_img2img", False):
            # standard img2img: Forge noises the encoded input image at Denoising strength
            if tuple(p.init_latent.shape) != tuple(p.modified_noise.shape):
                error = H3Error("The img2img input image did not reach H3. Set Settings > VAE > VAE for Encoding to "
                                "Full, and check the console for an earlier error.")
                set_pending_error(error)
                raise error
        else:
            # an edit samples from init_latent at full denoise: the packed start is pure noise
            import torch
            p.init_latent = torch.zeros_like(p.modified_noise)
    patches.begin_sampling()


def after_sampling(p):
    from .native import patches
    if getattr(p, "pi_h3_image_request", None) is not None:
        patches.end_sampling()


def postprocess(p, processed):
    """After Forge's own outputs: drop the decoded picture."""
    if getattr(p, "pi_h3_image_request", None) is None:
        return
    from .native import patches
    patches.end_sampling()
    p.sd_model.release_generation()
