"""MiniMax H3 transformer: embedding, packing, per-segment timesteps and the block loop.

Ported from ComfyUI comfy/ldm/minimax/model.py (MiniMaxH3Model). See dit.py for the layers.

Timestep domain: the model receives the sigma and derives per-token timesteps t = 1 - sigma internally. A still
image runs both streams on one schedule (ComfyUI's ModelSamplingMiniMaxH3 with the same shift for video and audio),
so the audio stub needs no schedule of its own.

MiniMax-H3-Image (ComfyUI MiniMaxH3ImageModel) is the same stack without the audio stream, [text | reference
pictures | image], with a bank of output heads distilled on the checkpoint's own sigma grid (pdd_sigmas).
"""

import torch
import torch.nn as nn

from ..contracts import raise_pending_error
from . import kernels, lora
from .dit import (
    SEGMENT_TAG,
    DiTBlock,
    FinalLayer,
    TimeEmbedder,
    TokenRefiner,
    rope_rotation_table,
)
from .layout import (
    VISUAL_COND_TIMESTEP,
    PackedLayout,
    pack_audio,
    pad_to_patch_size,
    patchify_video,
    unpack_audio,
    unpatchify_video,
)
from .streams import text_token_tags

# packed layouts kept per text length; the prompt and the negative prompt need one each
LAYOUT_CACHE_SIZE = 4


class MiniMaxH3Model(nn.Module):
    def __init__(self, hidden_size=5376, num_layers=50, token_refiner_num_layers=2,
                 num_attention_heads=56, attention_head_dim=128, ffn_hidden_size=14336,
                 latents_dim=24, audio_latents_dim=32, patch_size=(1, 2, 2), text_dim=5120,
                 timestep_input_dim=256, time_embed_hidden_size=5376, time_embed_dim=2688,
                 rope_inv_freq_len=16, norm_eps=1e-5, qk_norm_eps=1e-5, final_norm_eps=1e-5,
                 adaln_curve_grid=None, gate_compress=False, pdd_steps=1, pdd_sigmas=None,
                 dtype=None, device=None, **kwargs):
        super().__init__()
        self.dtype = dtype
        self.hidden_size = hidden_size
        self.patch_size = tuple(patch_size)
        self.latents_dim = latents_dim
        # None for the image model, which has no audio stream
        self.audio_latents_dim = audio_latents_dim
        # the decreasing sigma grid (pdd_steps + 1 points) a PDD head bank was distilled on
        self.pdd_sigmas = [float(s) for s in pdd_sigmas] if pdd_sigmas else None
        if pdd_steps > 1 and (self.pdd_sigmas is None or len(self.pdd_sigmas) != pdd_steps + 1):
            raise ValueError(f"[MiniMax H3] a checkpoint with {pdd_steps} output heads needs its {pdd_steps + 1} sigmas")
        self.use_adaln_curves = adaln_curve_grid is not None
        # pruned (curve-form) checkpoints replace the time embedder and full-width adaln weights with a small shared
        # basis of the time-embedding curve
        curve = {"apply_silu": not self.use_adaln_curves,
                 "adaln_dtype": torch.float32 if self.use_adaln_curves else dtype}
        video_patch_dim = latents_dim * self.patch_size[0] * self.patch_size[1] * self.patch_size[2]

        self.video_patch_proj = nn.Linear(video_patch_dim, hidden_size, bias=True, dtype=torch.float32, device=device)
        if audio_latents_dim is not None:
            self.audio_patch_proj = nn.Linear(audio_latents_dim, hidden_size, bias=True, dtype=torch.float32, device=device)
        self.condition_proj = nn.Linear(text_dim, hidden_size, bias=True, dtype=dtype, device=device)
        if self.use_adaln_curves:
            self.register_buffer("adaln_t_table", torch.empty(adaln_curve_grid, time_embed_dim, dtype=torch.float32, device=device))
        else:
            self.time_embedder = TimeEmbedder(timestep_input_dim, time_embed_hidden_size, time_embed_dim,
                                              dtype=torch.float32, device=device)
        self.rope = nn.Module()
        self.rope.register_buffer("inv_freq", torch.empty(rope_inv_freq_len, dtype=torch.float32, device=device))
        self.token_refiner = TokenRefiner(token_refiner_num_layers, hidden_size, num_attention_heads,
                                          attention_head_dim, ffn_hidden_size, norm_eps, qk_norm_eps,
                                          final_norm_eps, dtype=dtype, device=device)
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_attention_heads, attention_head_dim, ffn_hidden_size,
                     time_embed_dim, norm_eps, qk_norm_eps, **curve, gate_compress=gate_compress,
                     dtype=dtype, device=device)
            for _ in range(num_layers)])
        self.final_layer = FinalLayer(hidden_size, time_embed_dim, video_patch_dim, audio_latents_dim,
                                      final_norm_eps, **curve, pdd_steps=pdd_steps, dtype=dtype, device=device)

        # set by the engine for each request (streams.Generation): stream shapes, seed, reference pictures
        self.generation = None
        self._layouts = {}

    def preprocess_text_embeds(self, text_states):
        """[B, L, text_dim] Qwen states -> [B, L, hidden] refined text embeds, once per generation."""
        if text_states.shape[-1] == self.hidden_size:
            return text_states
        return self.token_refiner(self.condition_proj(text_states[0])).unsqueeze(0)

    def rope_freqs(self, position_ids, device):
        # [S, 3] float64 -> [S, 96] fp32
        pos = position_ids.to(torch.float32).to(device)
        inv = kernels.cast_to(self.rope.inv_freq, device=device)
        per_axis = pos.unsqueeze(-1) * inv.view(1, 1, -1)      # [S, 3, 16]
        t_f, h_f, w_f = per_axis.unbind(dim=1)
        half = torch.cat((t_f, h_f, w_f), dim=-1)              # [S, 48]
        return torch.cat((half, half), dim=-1)                 # [S, 96]

    def _cond_rows(self, latents, aug, seed, device):
        """Concatenated reference rows with condition noise augmentation; every condition restarts the same RNG."""
        rows = []
        for z in latents:
            r = patchify_video(z.to(torch.float32), self.patch_size)
            if aug < 1.0:
                gen = torch.Generator("cpu").manual_seed(seed)
                noise = torch.randn(r.shape, generator=gen, dtype=torch.float32)
                r = aug * r + (1.0 - aug) * noise.to(r.device)
            rows.append(r.to(device))
        return torch.cat(rows, dim=0) if rows else None

    @staticmethod
    def _merge_rows(target_rows, cond_rows, update):
        if cond_rows is None:
            return target_rows
        merged = torch.empty(update.shape[0], target_rows.shape[1], dtype=torch.float32, device=target_rows.device)
        merged[~update] = cond_rows
        merged[update] = target_rows
        return merged

    def _embed_and_pack(self, video_x, audio_x, context, layout, payload):
        device = video_x.device
        dtype = context.dtype
        seed = int(payload.get("seed", 0))
        cond_video = self._cond_rows(payload.get("cond_video_latents", []),
                                     payload.get("visual_cond_noise_aug", VISUAL_COND_TIMESTEP), seed, device)
        all_video_rows = self._merge_rows(patchify_video(video_x.to(torch.float32), self.patch_size), cond_video,
                                          layout.img_update.to(device))
        video_embed = self.video_patch_proj(all_video_rows).to(dtype)
        audio_embed = None
        if self.audio_latents_dim is not None:
            audio_embed = self.audio_patch_proj(pack_audio(audio_x.to(torch.float32))).to(dtype)
        text_states = context[0]
        if text_states.shape[-1] != self.hidden_size:
            text_states = self.token_refiner(self.condition_proj(text_states))

        # segments are contiguous: assemble by slices, embed rows follow segment order
        h = torch.empty(layout.seq_len, self.hidden_size, dtype=dtype, device=device)
        voff = aoff = 0
        for a, b, kind in layout.segments:
            n = b - a
            if kind == "text":
                h[a:b] = text_states
            elif kind in ("ref_img", "video"):
                h[a:b] = video_embed[voff:voff + n]
                voff += n
            else:  # audio
                h[a:b] = audio_embed[aoff:aoff + n]
                aoff += n
        return h

    def forward(self, x, timestep, context, transformer_options={}, **kwargs):
        """Forge's call: x is the packed latent [B, 1, 1, N] of the current generation (engine.prepare)."""
        if isinstance(x, (list, tuple)):
            return self.forward_streams(x, timestep, context, transformer_options, **kwargs)
        generation = getattr(self, "generation", None)
        if generation is None:
            raise_pending_error()
            raise RuntimeError("[MiniMax H3] the H3 settings were not applied to this generation; "
                               "check the console for an earlier MiniMax H3 message")
        shapes = generation.shapes
        text_len = context.shape[1]
        # as ComfyUI's model_base.MiniMaxH3.extra_conds: the reference latents ride as condition rows, never denoised
        payload = {"seed": generation.seed, "layout": self._layout(text_len, shapes, generation.refs)}
        if generation.refs:
            payload["refs"] = generation.refs
            payload["cond_video_latents"] = [r["latent"] for r in generation.refs]
        tags = text_token_tags(text_len, generation.vision_spans)
        if tags is not None:
            payload["text_token_tags"] = tags
        # a PDD head bank picks its heads from the sampler's schedule (Forge keeps it under sampling_sigmas)
        transformer_options = {**transformer_options, "sample_sigmas": transformer_options.get("sampling_sigmas")}
        outputs = []
        # the first step of a generation reports its working memory, to size engine.sampling_memory from real runs
        measure = x.is_cuda and not generation.memory_reported
        if measure:
            torch.cuda.synchronize(x.device)
            torch.cuda.reset_peak_memory_stats(x.device)
            resident = torch.cuda.memory_allocated(x.device)
        # on-the-fly LoRAs keep the INT8 matmuls, with the LoRA as a low-rank term (native/lora.py)
        with lora.low_rank(self):
            for i in range(x.shape[0]):
                video, audio = shapes.unpack(x[i:i + 1])
                v, a = self.forward_streams([video, audio], timestep[i:i + 1], context[i:i + 1],
                                            {**transformer_options, "minimax_h3_item": i}, minimax_payload=payload)
                outputs.append(shapes.pack(v, a))
        if measure:
            torch.cuda.synchronize(x.device)
            generation.memory_reported = True
            working = torch.cuda.max_memory_allocated(x.device) - resident
            tokens = payload["layout"].seq_len
            print(f"[MiniMax H3] sampling step: {working / 2**30:.2f} GiB of working memory for {tokens} tokens "
                  f"({working / tokens / 1024:.0f} KiB per token), {resident / 2**30:.1f} GiB resident")
        return torch.cat(outputs)

    def _layout(self, text_len, shapes, refs=()):
        # one layout per text length (prompt and negative prompt differ), rebuilt when the shapes or references change
        _, _, latent_t, lat_h, lat_w = shapes.video
        signature = (text_len, latent_t, lat_h + lat_h % 2, lat_w + lat_w % 2, shapes.audio[-1])
        key = signature + tuple((r["kind"], tuple(r["latent"].shape)) for r in refs)
        if key not in self._layouts:
            if len(self._layouts) >= LAYOUT_CACHE_SIZE:
                self._layouts.clear()
            self._layouts[key] = PackedLayout(*signature, refs=list(refs) or None)
        return self._layouts[key]

    def forward_streams(self, x, timestep, context, transformer_options={}, minimax_payload=None, **kwargs):
        """x: [video [1, 24, T, H, W], audio [1, 32, 2, T40]]; timestep is sigma * 1000. Returns both velocities."""
        return self._forward(x, timestep, context, transformer_options, minimax_payload=minimax_payload)

    def _time_embedding(self, t_vals, dtype):
        if not self.use_adaln_curves:
            return self.time_embedder(t_vals).to(dtype)
        # adaln projections consume interpolated coordinates of the time-embedding curve
        table = kernels.cast_to(self.adaln_t_table, device=t_vals.device)
        pos = t_vals.clamp(0.0, 1.0) * (table.shape[0] - 1)     # t in [0,1] -> fractional grid index
        i0 = pos.floor().long().clamp(max=table.shape[0] - 2)   # keeps t=1.0 on the last interval
        return torch.lerp(table[i0], table[i0 + 1], (pos - i0).unsqueeze(1))

    def _forward(self, x, timestep, context, transformer_options={}, minimax_payload=None):
        video_x, audio_x = x[0], x[1]
        orig_t, orig_h, orig_w = video_x.shape[2], video_x.shape[3], video_x.shape[4]
        video_x = pad_to_patch_size(video_x, self.patch_size)
        if video_x.shape[0] != 1:
            raise ValueError("MiniMax H3 supports batch size 1")
        payload = minimax_payload or {}
        device = video_x.device
        dtype = context.dtype  # compute dtype

        latent_t, lat_h, lat_w = video_x.shape[2], video_x.shape[3], video_x.shape[4]
        has_audio = self.audio_latents_dim is not None
        signature = (context.shape[1], latent_t, lat_h, lat_w, audio_x.shape[-1] if has_audio else 0)
        # the engine builds the layout once per generation
        layout = payload.get("layout")
        if layout is None or layout.signature != signature:
            layout = PackedLayout(*signature, refs=payload.get("refs"))
        transformer_options["minimax_h3_layout"] = layout

        sigma = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
        t = float(1.0 - sigma)
        # distinct timesteps are known analytically: text and the targets follow the sigma, reference rows pin near 1
        vis_aug = float(payload.get("visual_cond_noise_aug", VISUAL_COND_TIMESTEP))
        seg_t = {"text": t, "video": t, "audio": t, "ref_img": max(t, vis_aug)}
        unique_t = sorted({t} | {seg_t[k] for _, _, k in layout.segments})
        t_row = {value: i for i, value in enumerate(unique_t)}

        mod_segments = self._mod_segments(layout, seg_t, t_row, payload.get("text_token_tags"))

        h = self._embed_and_pack(video_x, audio_x, context, layout, payload)
        t_emb = self._time_embedding(torch.tensor(unique_t, dtype=torch.float32, device=device), dtype)
        # rotation table computed once per forward, consumed by the kitchen split-half rope
        rope_freqs = rope_rotation_table(self.rope_freqs(layout.position_ids, device), dtype)

        for i, block in enumerate(self.blocks):
            transformer_options["block_index"] = i
            h = block(h, t_emb, mod_segments, rope_freqs, transformer_options)

        # target streams are single contiguous segments (audio then video, last two; the image model has no audio)
        va, vb, _ = next(s for s in layout.segments if s[2] == "video")
        audio_seg = None
        if has_audio:
            aa, ab, _ = next(s for s in layout.segments if s[2] == "audio")
            audio_seg = (aa, ab, t_row[seg_t["audio"]])
        v, a = self.final_layer(h, t_emb, (va, vb, t_row[seg_t["video"]]), audio_seg, sigma=float(sigma),
                                sample_sigmas=transformer_options.get("sample_sigmas"), grid=self.pdd_sigmas)

        video_out = unpatchify_video(v, latent_t, lat_h // 2, lat_w // 2, self.latents_dim, self.patch_size)
        video_out = video_out[:, :, :orig_t, :orig_h, :orig_w]
        # the image model hands its (empty) audio back untouched, so the packed latent keeps its form
        audio_out = -unpack_audio(a).to(audio_x.dtype) if has_audio else audio_x
        return [-video_out.to(video_x.dtype), audio_out]

    @staticmethod
    def _mod_segments(layout, seg_t, t_row, text_tags):
        mod_segments = []
        for a, b, kind in layout.segments:
            row_base = t_row[seg_t[kind]] * 3
            if kind == "text" and text_tags is not None:
                # the presentation text span mixes tags (vision pads carry the video modality): split into tag runs
                tags = text_tags.view(-1).tolist()
                run_start = 0
                for i in range(1, b - a + 1):
                    if i == b - a or tags[i] != tags[run_start]:
                        mod_segments.append((a + run_start, a + i, row_base + int(tags[run_start])))
                        run_start = i
            else:
                mod_segments.append((a, b, row_base + SEGMENT_TAG[kind]))
        return mod_segments
