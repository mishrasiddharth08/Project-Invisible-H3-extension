"""Read only safetensors headers; never deserialize a checkpoint to discover it."""

import json
import struct
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .contracts import H3Error

MAX_HEADER_BYTES = 32 * 1024 * 1024
ROLE_LABELS = {"dit": "diffusion model", "text_encoder": "text encoder", "vae": "VAE", "audio_vae": "audio VAE"}
# what a generation needs under VAE / Text Encoder
REQUIRED_ROLES = ("text_encoder", "vae")


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise H3Error(f"Duplicate key in safetensors header: {key}")
        result[key] = value
    return result


@lru_cache(maxsize=32)
def _read_header(path, size, mtime):
    try:
        with open(path, "rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise H3Error("Invalid safetensors header: missing length.")
            length = struct.unpack("<Q", prefix)[0]
            if not 2 <= length <= MAX_HEADER_BYTES or length > size - 8:
                raise H3Error("Invalid or oversized safetensors header.")
            header = json.loads(stream.read(length), object_pairs_hook=_unique_pairs)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise H3Error(f"Cannot read safetensors header: {exc}") from exc
    if not isinstance(header, dict):
        raise H3Error("Invalid safetensors header object.")
    for name, tensor in header.items():
        if name == "__metadata__":
            if not isinstance(tensor, dict):
                raise H3Error("Invalid safetensors metadata.")
            continue
        if not isinstance(tensor, dict) or not isinstance(tensor.get("shape"), list):
            raise H3Error(f"Invalid tensor in safetensors header: {name}")
        if not all(isinstance(n, int) and n >= 0 for n in tensor["shape"]):
            raise H3Error(f"Invalid tensor shape in safetensors header: {name}")
    return header


GGUF_MAGIC = b"GGUF"
MAX_GGUF_ITEMS = 1_000_000
MAX_GGUF_STRING = 1024 * 1024
# ggml tensor types (ggml.h); Forge Neo dequantizes F32/F16/BF16, Q4_0..Q8_0 and the K-quants, not the IQ family
GGML_TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1", 10: "Q2_K",
              11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
              19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32",
              27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16"}
GGUF_PLAIN = {"F32", "F16", "BF16"}
GGUF_FORGE = GGUF_PLAIN | {"Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K"}
# GGUF metadata value types: fixed-size scalars by struct format, 8 = string, 9 = array
GGUF_SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


class _GGUFStream:
    def __init__(self, stream):
        self.stream = stream

    def unpack(self, fmt):
        size = struct.calcsize("<" + fmt)
        data = self.stream.read(size)
        if len(data) != size:
            raise H3Error("Invalid GGUF header: truncated.")
        return struct.unpack("<" + fmt, data)[0]

    def count(self):
        value = self.unpack("Q")
        if value > MAX_GGUF_ITEMS:
            raise H3Error("Invalid or oversized GGUF header.")
        return value

    def string(self):
        length = self.unpack("Q")
        if length > MAX_GGUF_STRING:
            raise H3Error("Invalid GGUF header: oversized string.")
        data = self.stream.read(length)
        if len(data) != length:
            raise H3Error("Invalid GGUF header: truncated.")
        return data.decode("utf-8")

    def value(self, kind):
        if kind in GGUF_SCALARS:
            return self.unpack(GGUF_SCALARS[kind])
        if kind == 8:
            return self.string()
        if kind == 9:
            item_kind = self.unpack("I")
            return [self.value(item_kind) for _ in range(self.count())]
        raise H3Error(f"Invalid GGUF metadata type {kind}.")


@lru_cache(maxsize=32)
def _read_gguf_header(path, size, mtime):
    """The tensor table of a GGUF file in the shape of a safetensors header: name -> {"dtype", "shape"}."""
    try:
        with open(path, "rb") as raw:
            if raw.read(4) != GGUF_MAGIC:
                raise H3Error("Invalid GGUF file: missing magic.")
            stream = _GGUFStream(raw)
            if stream.unpack("I") not in (2, 3):
                raise H3Error("Unsupported GGUF version.")
            tensor_count, kv_count = stream.count(), stream.count()
            metadata = {}
            for _ in range(kv_count):
                key = stream.string()
                metadata[key] = stream.value(stream.unpack("I"))
            header = {}
            for _ in range(tensor_count):
                name = stream.string()
                dims = [stream.unpack("Q") for _ in range(stream.unpack("I"))]
                kind, _offset = stream.unpack("I"), stream.unpack("Q")
                if name in header:
                    raise H3Error(f"Duplicate tensor in GGUF header: {name}")
                # ggml lists dimensions innermost first; ComfyUI-GGUF stores reshaped tensors' real shape apart
                shape = metadata.get(f"comfy.gguf.orig_shape.{name}") or dims[::-1]
                header[name] = {"dtype": GGML_TYPES.get(kind, f"type{kind}"), "shape": [int(n) for n in shape]}
    except (OSError, UnicodeError, RecursionError) as exc:
        raise H3Error(f"Cannot read GGUF header: {exc}") from exc
    header["__metadata__"] = {k: v for k, v in metadata.items() if not k.startswith("comfy.gguf.orig_shape.")}
    return header


def read_header(path):
    path = Path(path).resolve()
    try:
        stat = path.stat()
    except OSError as exc:
        raise H3Error(f"Model file is unavailable: {path.name}") from exc
    reader = _read_gguf_header if path.suffix.lower() == ".gguf" else _read_header
    return reader(str(path), stat.st_size, stat.st_mtime_ns)


# the variants of a file: P-H3-Image's own text encoder and VAE; MiniMax-H3-Image's checkpoint (no audio stream) and
# 2D image VAE; "h3" for what MiniMax H3 itself ships: the audio-video DiT layout (P-H3-Image's checkpoint has it
# too), the 32B text encoder (MiniMax-H3-Image uses it) and the video VAE, which is recognized only to be refused
PH3_IMAGE = "ph3-image"
H3_IMAGE = "h3-image"
H3 = "h3"
# P-H3-Image's text encoder carries Pruna's text connector; the image decoders make one frame per latent
# (3 x 16 x 16), MiniMax-H3-Image's with 16 heads of 64 (P-H3-Image's has 32)
TEXT_CONNECTOR = "text_connector.proj.weight"
STILL_PROJ_OUT = 768
H3_IMAGE_DECODER_DIM = 1024
DIT_KEYS = {"video_patch_proj.weight", "condition_proj.weight", "final_layer.video_out.weight"}
DIT_AUDIO_KEY = "audio_patch_proj.weight"
# a checkpoint may ask for an attention kernel per block (ComfyUI's comfy_attention); Forge Neo picks one globally
ATTENTION_KEY = "blocks.0.attn.comfy_attention.config"


@dataclass(frozen=True)
class ModelInfo:
    path: Path
    role: str
    quantization: str
    variant: str = H3


def _role(keys):
    clean = {k.removeprefix("model.diffusion_model.") for k in keys}
    if DIT_KEYS <= clean:
        return "dit"
    if (any(k.endswith("embed_tokens.weight") for k in keys) and any("visual." in k for k in keys)
            and (any("layers.49." in k for k in keys) or TEXT_CONNECTOR in keys)):
        return "text_encoder"
    if any(k.startswith("decoder.x_embedder.") for k in keys) and any(k.startswith("decoder.register_tokens") for k in keys):
        return "vae"
    if {"pre_block.attn.q_bias", "pre_block.attn.v_bias", "pre_block.attn.zero_k_bias"} <= keys:
        return "audio_vae"
    return None


def _variant(role, header, keys):
    if role == "text_encoder" and TEXT_CONNECTOR in keys:
        return PH3_IMAGE
    if role == "vae" and header.get("decoder.proj_out.weight", {}).get("shape", [0])[0] == STILL_PROJ_OUT:
        wide = header.get("decoder.x_embedder.weight", {}).get("shape", [0])[0]
        return H3_IMAGE if wide == H3_IMAGE_DECODER_DIM else PH3_IMAGE
    if role == "dit" and DIT_AUDIO_KEY not in {k.removeprefix("model.diffusion_model.") for k in keys}:
        return H3_IMAGE
    return H3


def is_image_checkpoint(item):
    """The two image routes must leave ordinary H3 audio/video checkpoints alone."""
    return item is not None and item.role == "dit" and item.variant in (H3_IMAGE, PH3_IMAGE)


def is_pruna_name(path):
    return Path(path).stem.casefold() in {
        "p-h3-image_bf16", "p-h3-image_int8_convrot", "p-h3-image_w4a8"
    }


# ComfyUI quantization formats as the files name them, and the short label the Components panel shows
QUANT_LABELS = {"int8_tensorwise": "int8", "asym_w4a8_int8": "w4a8", "w6a8_int8": "w6a8", "convrot_w4a4": "int4",
                "float8_e4m3fn": "fp8", "float8_e5m2": "fp8", "mxfp8": "mxfp8", "nvfp4": "nvfp4"}
MAX_COMFY_QUANT_BYTES = 4096


@lru_cache(maxsize=32)
def _declared_formats(path, size, mtime):
    """The per-layer formats a ComfyUI-quantized safetensors file declares: its _quantization_metadata, else the small
    comfy_quant JSON tensors (a few bytes each, the only tensor data read for discovery)."""
    header = _read_header(path, size, mtime)
    raw = header.get("__metadata__", {}).get("_quantization_metadata")
    if raw:
        try:
            layers = json.loads(raw).get("layers", {})
            return frozenset(v.get("format") for v in layers.values() if isinstance(v, dict)) - {None}
        except (ValueError, AttributeError):
            return frozenset()
    entries = [v for k, v in header.items() if k.endswith(".comfy_quant") and isinstance(v, dict)]
    formats = set()
    if entries:
        try:
            with open(path, "rb") as stream:
                base = 8 + struct.unpack("<Q", stream.read(8))[0]
                for entry in entries:
                    start, end = entry.get("data_offsets", [0, 0])
                    if not 0 < end - start <= MAX_COMFY_QUANT_BYTES:
                        continue
                    stream.seek(base + start)
                    try:
                        formats.add(json.loads(stream.read(end - start)).get("format"))
                    except (ValueError, AttributeError):
                        continue
        except OSError as exc:
            raise H3Error(f"Cannot read {Path(path).name}: {exc}") from exc
    return frozenset(formats) - {None}


def _quantization(header, keys, path=None):
    metadata = json.dumps(header.get("__metadata__", {})).lower()
    if "nvfp4" in metadata or "awq" in metadata or any(k.endswith("weight_scale_2") for k in keys):
        return "nvfp4"
    labels = []
    if path is not None:
        stat = path.stat()
        labels = sorted({QUANT_LABELS.get(f, f) for f in _declared_formats(str(path), stat.st_size, stat.st_mtime_ns)})
    if labels:
        # a mixed file such as tsolful's INT4BQ declares int4 and int8 layers
        return " + ".join(labels)
    if any(k.endswith(".weight_s_rel") for k in keys):
        return "w6a8"
    if any("weight_scale" in k or "scale_weight" in k for k in keys):
        return "int8" if any(v.get("dtype") == "I8" for k, v in header.items() if k != "__metadata__") else "fp8"
    return "plain"


def _gguf_quantization(path, header, role):
    types = {v["dtype"] for k, v in header.items() if k != "__metadata__"}
    unsupported = sorted(types - GGUF_FORGE)
    if unsupported:
        raise H3Error(f"{path.name} uses GGUF types Forge Neo cannot dequantize ({', '.join(unsupported)}). "
                      "Use a Q2_K to Q8_0 build.")
    if role not in ("dit", "text_encoder"):
        raise H3Error(f"{path.name}: GGUF is supported for the H3 diffusion model and text encoder only. Select a "
                      f"safetensors {ROLE_LABELS[role]}.")
    counts = {}
    for k, v in header.items():
        if k != "__metadata__" and v["dtype"] not in GGUF_PLAIN:
            counts[v["dtype"]] = counts.get(v["dtype"], 0) + 1
    return f"gguf {max(counts, key=counts.get)}" if counts else "gguf"


def inspect_model(path):
    path = Path(path).resolve()
    suffix = path.suffix.lower()
    if suffix not in (".safetensors", ".gguf"):
        return None
    header = read_header(path)
    keys = set(header) - {"__metadata__"}
    role = _role(keys)
    if role is None:
        if suffix == ".gguf" and any(k.startswith("blk.") for k in keys):
            # llama.cpp keeps Qwen3-VL's vision tower in a separate mmproj file; unsloth's H3 file carries it inside
            raise H3Error(f"{path.name}: llama.cpp GGUF text encoders keep the vision part in a separate file, which "
                          "H3 cannot load. Select unsloth's qwen3vl_32b_minimax_h3 GGUF or a safetensors H3 text "
                          "encoder, such as qwen3vl_32b_minimax_h3_int8_convrot.")
        return None
    variant = _variant(role, header, keys)
    if role == "dit" and variant == H3 and is_pruna_name(path):
        variant = PH3_IMAGE
    if suffix == ".gguf":
        return ModelInfo(path, role, _gguf_quantization(path, header, role), variant)
    return ModelInfo(path, role, _quantization(header, keys, path), variant)


@dataclass(frozen=True)
class Components:
    dit: ModelInfo
    text_encoder: ModelInfo
    vae: ModelInfo

    @property
    def models(self):
        return (self.dit, self.text_encoder, self.vae)

    @property
    def profile(self):
        """The recipe of the selected model: MiniMax-H3-Image is told by its checkpoint, P-H3-Image by its text
        encoder in front of an audio-video checkpoint."""
        from .profiles import for_components
        return for_components(self.dit.variant, self.text_encoder.variant)

    @property
    def experimental(self) -> str:
        """A pairing the models were not released with, in words for the console, the panel and the infotext; empty
        for a released set. Both are MiniMax-H3-Image with a part of P-H3-Image. Its 4B text encoder: Pruna distilled
        that encoder (and its text connector) to stand in for the H3 32B one in front of an H3 DiT, so the
        conditioning has the width and, by reports, the meaning this checkpoint reads; the checkpoint itself was never
        trained on it. Its VAE: the same H3 encoder and latent statistics with another single-frame decoder, so it
        reads the same latents; users report finer detail from it."""
        if self.dit.variant != H3_IMAGE:
            return ""
        parts = [label for variant, label in ((self.text_encoder.variant, "P-H3-Image text encoder"),
                                              (self.vae.variant, "P-H3-Image VAE")) if variant == PH3_IMAGE]
        return " + ".join(parts)

    @property
    def wants_int8_attention(self) -> bool:
        """Whether the checkpoint declares comfy-kitchen's INT8 attention for its blocks, as ComfyUI then uses."""
        return ATTENTION_KEY in read_header(self.dit.path)


def resolve_components(dit_path, module_paths):
    """The H3 checkpoint and the two modules selected under VAE / Text Encoder, checked before Forge loads them."""
    dit = inspect_model(dit_path)
    if dit is None or dit.role != "dit":
        raise H3Error("The selected checkpoint is not a recognized H3 diffusion model.")
    resolved = {"dit": dit}
    for path in dict.fromkeys(str(Path(p).resolve()) for p in module_paths):
        item = inspect_model(path)
        if item is None or item.role == "dit":
            raise H3Error(f"Unrecognized H3 component: {Path(path).name}. Select only the H3 text encoder and VAE.")
        if item.role == "audio_vae":
            raise H3Error(f"{Path(path).name} is the H3 audio VAE: still images do not use it. Deselect it.")
        if item.role == "vae" and item.variant == H3:
            raise H3Error(f"{Path(path).name} is the H3 video VAE, which makes four frames per latent: still images "
                          "use the P-H3-Image VAE (p-h3-image_vae) or the MiniMax-H3-Image VAE "
                          "(minimax_h3_image_vae). Select one of them in its place.")
        if item.role in resolved:
            raise H3Error(f"More than one H3 {ROLE_LABELS[item.role]} is selected.")
        resolved[item.role] = item
    if "text_encoder" in resolved and resolved["text_encoder"].quantization == "nvfp4":
        # Forge Neo loads it without a warning, but the conditioning comes out wrong (a prompt for a bird gave a dog)
        raise H3Error("The NVFP4 AWQ text encoder does not encode prompts correctly in Forge Neo yet. Select qwen3vl_32b_minimax_h3_int8_convrot or the bf16 text encoder.")
    for role in REQUIRED_ROLES:
        if role not in resolved:
            raise H3Error(f"Select the H3 {ROLE_LABELS[role]} in Forge's VAE / Text Encoder field.")
    components = Components(**resolved)
    if components.dit.variant == H3_IMAGE:
        # its own parts are the H3 32B text encoder and the MiniMax-H3-Image VAE; P-H3-Image's 4B encoder and its VAE
        # are let through as experiments (Components.experimental). The video VAE was refused above.
        return components
    if components.vae.variant == H3_IMAGE:
        raise H3Error("The MiniMax-H3-Image VAE goes with the MiniMax-H3-Image checkpoint (minimax_h3_image): select "
                      "that checkpoint, or the VAE of the selected model.")
    if components.text_encoder.variant != PH3_IMAGE:
        # MiniMax H3 with its own 32B text encoder is the video model; as a still-image model the audio-video
        # checkpoints run behind P-H3-Image's text encoder (Pruna's checkpoint, or MiniMax H3 with Pruna's LoRA)
        raise H3Error("An H3 audio-video checkpoint makes still images with the P-H3-Image text encoder "
                      "(p-h3-image_text_encoder) and VAE (p-h3-image_vae): select them in place of the H3 text "
                      "encoder. For MiniMax's own image model, select the MiniMax-H3-Image checkpoint "
                      "(minimax_h3_image).")
    return components
