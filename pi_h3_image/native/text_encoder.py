"""The Qwen3-VL conditioning encoders of MiniMax H3 and P-H3-Image, on Forge Neo's Llama2_.

MiniMax H3: Qwen3-VL 32B truncated to the first 50 of 64 layers and consumed as the unnormalized hidden state after
layer 50: no final norm and no lm_head (ComfyUI comfy/text_encoders/llama.py Qwen3VL_32BConfig). Forge Neo's Qwen3VL
is pinned to the 4B and 8B configs, so the 32B subclass builds the size from the config.

P-H3-Image: Qwen3-VL 4B, Forge Neo's own config, read at five hidden states (the last one after the final norm) that
Pruna's text connector standardizes, projects to the 5120 width the DiT expects and rescales, with the first token
replaced by a learned sink (ComfyUI comfy/text_encoders/minimax.py PH3ImageQwen3VL).

With pictures both run what Forge Neo's Qwen3VL leaves out, interleaved M-RoPE and the DeepStack visual features
(from the Qwen-Image 2.1 extension, where the same Qwen3-VL family is used at 8B).
"""

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
from backend.nn.llm import llama, qwen35
from backend.nn.llm.llama import Llama2_, Qwen3VL, Qwen3VL_4BConfig
from backend.nn.llm.qwen35 import Qwen3VLVisionModel

from ..profiles import PH3_IMAGE_HIDDEN_STATES

VISION_KEYS = ("hidden_size", "intermediate_size", "depth", "num_heads", "num_position_embeddings", "deepstack_visual_indexes")

# P-H3-Image: the text connector's output width
PH3_IMAGE_TEXT_DIM = 5120
PH3_IMAGE_MODEL_TYPE = "qwen3vl_4b"


def vision_defaults() -> dict:
    """Forge Neo's Qwen3-VL vision settings: one dict up to d70373e, a shared part plus one dict per model since its
    Qwen-Image 2.1 update; the 32B tower's own sizes come from the text encoder config (VISION_KEYS)."""
    common = getattr(qwen35, "QWEN3VL_VISION_COMMON", None)
    if common is None:
        return dict(qwen35.QWEN3VL_VISION)
    return {**common, **qwen35.QWEN3VL_VISION["qwen3vl_8b"]}


@dataclass
class Qwen3VL_32BConfig(Qwen3VL_4BConfig):
    hidden_size: int = 5120
    intermediate_size: int = 25600
    num_hidden_layers: int = 50
    num_attention_heads: int = 64
    final_norm: bool = False


def interleaved_mrope(config, position_ids: torch.Tensor, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # transformers Qwen3VLTextRotaryEmbedding: the height / width frequencies interleave with the temporal ones
    # (T H W T H W ... T), while Forge Neo's precompute_freqs_cis lays them out in blocks, as Qwen2.5-VL does
    head_dim = config.head_dim
    inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    freqs = position_ids.to(device).float()[:, :, None] * inv_freq  # (3, seq, head_dim / 2)
    merged = freqs[0].clone()
    for axis in (1, 2):
        section = slice(axis, config.rope_dims[axis] * 3, 3)
        merged[:, section] = freqs[axis, :, section]

    emb = torch.cat((merged, merged), dim=-1)[None, None]  # (1, 1, seq, head_dim)
    cos, sin = emb.cos(), emb.sin()
    half = head_dim // 2
    return cos, sin[..., :half], -sin[..., half:]


def _attention_mask(attention_mask: torch.Tensor | None, x: torch.Tensor) -> torch.Tensor:
    # padding + causal, as Llama2_.forward builds it
    batch, seq_len = x.shape[0], x.shape[1]
    low = torch.finfo(x.dtype).min / 4
    mask = torch.empty(seq_len, seq_len, dtype=x.dtype, device=x.device).fill_(low).triu_(1)
    if attention_mask is not None:
        padding = 1.0 - attention_mask.to(x.dtype).reshape((batch, 1, -1, attention_mask.shape[-1])).expand(batch, 1, seq_len, attention_mask.shape[-1])
        mask = padding.masked_fill(padding.to(torch.bool), low) + mask
    return mask


def forward_with_images(model: Llama2_, x, attention_mask, intermediate_output, final_layer_norm_intermediate,
                        position_ids, visual_mask, deepstack):
    """Llama2_.forward plus transformers Qwen3VLTextModel: interleaved M-RoPE, and the DeepStack features added to the
    image positions after the first decoder layers. intermediate_output is one layer index, or a list of them as
    Llama2_ reads it (the state entering layer i; the count of layers means after the final norm)."""
    freqs_cis = interleaved_mrope(model.config, position_ids, x.device)
    mask = _attention_mask(attention_mask, x)
    taps = None
    if isinstance(intermediate_output, list):
        taps = set(intermediate_output)
    elif intermediate_output is not None and intermediate_output < 0:
        intermediate_output = len(model.layers) + intermediate_output

    intermediate = None
    collected = []
    for i, layer in enumerate(model.layers):
        if taps is not None and i in taps:
            collected.append(x.unsqueeze(1).clone())
        x, _ = layer(x=x, attention_mask=mask, freqs_cis=freqs_cis, optimized_attention=llama.attention_function, past_key_value=None)
        if i < len(deepstack):
            x[:, visual_mask] = x[:, visual_mask] + deepstack[i].to(device=x.device, dtype=x.dtype)
        if taps is None and i == intermediate_output:
            intermediate = x.clone()

    if model.norm is not None:
        x = model.norm(x)
    if taps is not None:
        if len(model.layers) in taps:
            collected.append(x.unsqueeze(1).clone())
        intermediate = torch.cat(collected, dim=1)
    if intermediate is not None and final_layer_norm_intermediate and model.norm is not None:
        intermediate = model.norm(intermediate)
    return x, intermediate


class _H3Qwen3VL(Qwen3VL):
    """The image path shared by the H3 and P-H3-Image encoders."""

    def forward(self, x, attention_mask=None, embeds=None, num_tokens=None, intermediate_output=None, final_layer_norm_intermediate=True, dtype=None, embeds_info=[], **kwargs):
        position_ids, visual_mask, deepstack = (None, None, None) if embeds is None else self.build_image_inputs(embeds, embeds_info)
        if position_ids is None:
            # text only: the three M-RoPE axes coincide and Forge Neo's own path is exact
            return self.model(x, attention_mask=attention_mask, embeds=embeds, num_tokens=num_tokens, intermediate_output=intermediate_output, final_layer_norm_intermediate=final_layer_norm_intermediate, dtype=dtype)
        return forward_with_images(self.model, embeds, attention_mask, intermediate_output, final_layer_norm_intermediate,
                                   position_ids, visual_mask[0], deepstack)


class Qwen3VL32B(_H3Qwen3VL):
    def __init__(self, config_dict: dict, **kwargs):
        # Forge Neo after d70373e passes model_type (its 4B or 8B); the 32B config is our own
        nn.Module.__init__(self)
        text_config: dict = config_dict.get("text_config", {})
        if text_config.get("hidden_size", None) != Qwen3VL_32BConfig.hidden_size:
            raise ValueError("[MiniMax H3] the text encoder config is not Qwen3-VL 32B")
        config = Qwen3VL_32BConfig()

        for key, value in asdict(config).items():
            if key in config_dict:
                assert value == config_dict[key]

        self.num_layers = config.num_hidden_layers
        self.model = Llama2_(config)

        vision_config = {**vision_defaults(), "out_hidden_size": config.hidden_size}
        for key in VISION_KEYS:
            if key in config_dict.get("vision_config", {}):
                vision_config[key] = config_dict["vision_config"][key]
        self.visual = Qwen3VLVisionModel(vision_config)


class TextConnector(nn.Module):
    """Pruna's connector: standardized, concatenated hidden states -> the 5120-wide conditioning, sink token first."""

    def __init__(self, in_channels: int, out_channels: int, dtype=None, device=None):
        super().__init__()
        self.input_mean = nn.Parameter(torch.empty(in_channels, dtype=dtype, device=device), requires_grad=False)
        self.input_std = nn.Parameter(torch.empty(in_channels, dtype=dtype, device=device), requires_grad=False)
        self.proj = nn.Linear(in_channels, out_channels, bias=False, dtype=dtype, device=device)
        self.output_mean = nn.Parameter(torch.empty(out_channels, dtype=dtype, device=device), requires_grad=False)
        self.output_std = nn.Parameter(torch.empty(out_channels, dtype=dtype, device=device), requires_grad=False)
        self.sink_embedding = nn.Parameter(torch.empty(out_channels, dtype=dtype, device=device), requires_grad=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # [B, layers, T, C] -> [B, T, layers * C] standardized, projected and rescaled, in fp32 (the conditioning is
        # taken as fp32 anyway); the first token is the sink
        x = hidden_states.movedim(1, 2).flatten(2).to(torch.float32)
        x = (x - self.input_mean.to(x)) / self.input_std.to(x)
        x = self.proj(x)
        x = torch.addcmul(self.output_mean.to(x), x, self.output_std.to(x))
        sink = self.sink_embedding.to(x).expand(x.shape[0], 1, -1)
        return torch.cat([sink, x[:, 1:]], dim=1)


class PH3ImageQwen3VL(_H3Qwen3VL):
    def __init__(self, config_dict: dict, **kwargs):
        # Forge Neo's own 4B model (Krea 2's), whatever model_type the loader passes
        super().__init__(config_dict, model_type=PH3_IMAGE_MODEL_TYPE)
        self.text_connector = TextConnector(len(PH3_IMAGE_HIDDEN_STATES) * self.model.config.hidden_size, PH3_IMAGE_TEXT_DIM,
                                            dtype=torch.float32)

    def forward(self, *args, **kwargs):
        x, hidden_states = super().forward(*args, **kwargs)
        if hidden_states is None or hidden_states.ndim != 4:
            raise ValueError("[MiniMax H3] the P-H3-Image text encoder needs its five hidden states")
        return x, self.text_connector(hidden_states)
