"""The H3 image VAEs: the 3D causal CNN encoder of MiniMax H3 with a single-frame ViT decoder.

Ported from ComfyUI comfy/ldm/minimax/vae.py (see native.COMFYUI_COMMIT): H3ImageVAE and its two subclasses, one for
P-H3-Image and one for MiniMax-H3-Image. ComfyUI's fused kitchen paths (NDHWC pad, fp16-accumulate conv, INT8
activation folding, INT8 attention) are left out; the math is the same. One picture is one latent frame, 16x spatial.
The video decoder of MiniMax H3 (clips of 17 frames, temporal chunking) is not here: the extension makes stills only.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import kernels

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

LATENTS_MEAN = [
    0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075,
    -0.2727581858634949, -1.3675414323806763, -0.2553254961967468, -0.26907554268836975,
    -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,
    -0.5460608005523682, -0.4035342037677765, -0.23683024942874908, 0.25928452610969543,
    -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279,
    -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264,
]

LATENTS_STD = [
    1.2223774194717407, 1.2767263650894165, 1.68317747116088865, 1.7549455165863037,
    1.5636216402053833, 2.194143533706665, 0.96531379222869875, 1.05698859691619875,
    0.841948926448822, 0.7729952931404114, 1.8955937623977661, 0.946841835975647,
    0.7996809482574463, 0.44988900423049925, 0.7197399735450745, 0.69362932443618775,
    2.961095094680786, 2.7694199085235595, 3.0496184825897215, 2.1088054180145265,
    3.276226282119751, 3.1627357006073, 2.28168129920959475, 2.6127843856811525,
]

def _rms_norm(x, weight, eps):
    return F.rms_norm(x, (x.shape[-1],), weight=weight, eps=eps)


# 3D causal CNN encoder

def _offloadable(layer):
    """Forge's offload attributes on a layer that subclasses torch's own (defined before using_forge_operations swaps
    them): with these Forge may keep its weights in system RAM when VRAM is short, and the layer then casts them to
    the input's device for each call, as Forge's layers do."""
    layer.parameters_manual_cast = False
    layer.weight_function, layer.bias_function = [], []


class CausalConv3d(nn.Conv3d):
    # reflect spatial padding, causal (zeros, front-only) temporal padding
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        _offloadable(self)
        self.causal_padding = (padding,) * 3 if isinstance(padding, int) else tuple(padding)

    def forward(self, x, pre_norm=None, spatial_pad=None, residual=None):
        pad_t, pad_h, pad_w = self.causal_padding
        if spatial_pad is None:
            spatial_pad = (pad_w, pad_w, pad_h, pad_h)
        if pre_norm is not None:
            x = F.silu(pre_norm(x))
        if any(spatial_pad):
            x = F.pad(x, (*spatial_pad, 0, 0), mode="reflect")
        if pad_t:
            # zero front frames; for a single frame ComfyUI truncates the taps instead, which is the same sum
            x = F.pad(x, (0, 0, 0, 0, pad_t * 2, 0), mode="constant")
        if self.parameters_manual_cast:
            with kernels.weight_and_bias(self, x) as (weight, bias):
                out = self._conv_forward(x, weight, bias)
        else:
            out = super().forward(x)
        if residual is not None:
            out = out + residual
        return out


class TemporalIsolatedGroupNorm(nn.GroupNorm):
    # GroupNorm with statistics computed per frame (time merged into batch)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _offloadable(self)

    def _norm(self, x):
        if self.parameters_manual_cast and self.affine:
            with kernels.weight_and_bias(self, x) as (weight, bias):
                return F.group_norm(x, self.num_groups, weight, bias, self.eps)
        return super().forward(x)

    def forward(self, x):
        if x.dim() == 5:
            b, c, t, h, w = x.shape
            x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
            x = self._norm(x)
            return x.view(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()
        return self._norm(x)


def group_norm_3d(num_channels):
    return TemporalIsolatedGroupNorm(num_groups=32, num_channels=num_channels, eps=1e-6, affine=True)


class Downsample3D(nn.Module):
    def __init__(self, in_channels, out_channels, time_stride=1, space_stride=2):
        super().__init__()
        self.space_stride = space_stride
        self.conv = CausalConv3d(in_channels, out_channels, kernel_size=3, padding=(1, 0, 0),
                                 stride=(time_stride, space_stride, space_stride))

    def forward(self, x):
        if self.space_stride == 2:
            return self.conv(x, spatial_pad=(0, 1, 0, 1))
        return self.conv(x)


class ResnetBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.norm1 = group_norm_3d(in_channels)
        self.norm2 = group_norm_3d(out_channels)
        self.conv1 = CausalConv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = CausalConv3d(out_channels, out_channels, kernel_size=3, padding=1)
        if in_channels != out_channels:
            self.nin_shortcut = CausalConv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        h = self.conv1(x, pre_norm=self.norm1)
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return self.conv2(h, pre_norm=self.norm2, residual=x)


class EncoderFCN3D(nn.Module):
    def __init__(self, ch, ch_mult, space_down, time_down, num_res_blocks, in_channels, z_channels, double_z=True):
        super().__init__()
        self.num_levels = len(ch_mult)
        if isinstance(num_res_blocks, int):
            num_res_blocks = [num_res_blocks] * self.num_levels
        self.num_res_blocks = num_res_blocks

        block_mid = [ch * ch_mult[i] for i in range(self.num_levels)]
        block_in = [block_mid[0]] + block_mid[:-1]
        block_out = block_mid

        self.conv_in = CausalConv3d(in_channels, block_in[0], kernel_size=3, padding=1)
        self.down = nn.ModuleList()
        for i_level in range(self.num_levels):
            down = nn.Module()
            down.block = nn.ModuleList([
                ResnetBlock3D(in_channels=block_in[i_level] if i == 0 else block_mid[i_level], out_channels=block_mid[i_level])
                for i in range(self.num_res_blocks[i_level])])
            if space_down[i_level] * time_down[i_level] > 1:
                down.downsample = Downsample3D(block_mid[i_level], block_out[i_level],
                                               time_stride=time_down[i_level], space_stride=space_down[i_level])
            self.down.append(down)

        self.norm_out = group_norm_3d(block_out[-1])
        self.conv_out = CausalConv3d(block_out[-1], 2 * z_channels if double_z else z_channels, kernel_size=3, padding=1)

    def forward(self, x):
        h = self.conv_in(x)
        for i_level in range(self.num_levels):
            for i_block in range(self.num_res_blocks[i_level]):
                h = self.down[i_level].block[i_block](h)
            if hasattr(self.down[i_level], "downsample"):
                h = self.down[i_level].downsample(h)
        return self.conv_out(h, pre_norm=self.norm_out)


# ViT3D decoder

def create_token_ids(patch_dims, device, dtype):
    coords_list = []
    for dim_size in patch_dims:
        coords = torch.arange(0.5, dim_size, dtype=dtype, device=device) / dim_size
        coords_list.append(2.0 * coords - 1.0)
    coords = torch.stack(torch.meshgrid(*coords_list, indexing="ij"), dim=-1)
    return coords.flatten(0, len(patch_dims) - 1).unsqueeze(0)


class RotaryEmbeddingND(nn.Module):
    def __init__(self, dim, rotary_base=100.0, n_dim=3):
        super().__init__()
        self.n_dim = n_dim
        self.angle_scale = 2.0 * math.pi
        inv_freq = 1 / rotary_base ** torch.arange(0, 1, 2 * n_dim / dim, dtype=torch.float32)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, img_ids):
        # [B, S, n_dim] -> [B, S, 1, pairs, 2, 2] rotation table for the kitchen split-half rope
        angles = self.angle_scale * img_ids[:, :, :, None].float() * self.inv_freq.to(img_ids.device)[None, None, None, :]
        angles = angles.flatten(2, 3)
        c, s = torch.cos(angles), torch.sin(angles)
        table = torch.stack([c, -s, s, c], dim=-1).reshape(*angles.shape[:2], 1, angles.shape[-1], 2, 2)
        return table.to(img_ids.dtype)


class FeedForward(nn.Module):
    # gated SiLU FFN
    def __init__(self, dim, mult=4, bias=True):
        super().__init__()
        inner_dim = dim * mult
        self.w1 = nn.Linear(dim, inner_dim * 2, bias=bias)
        self.w2 = nn.Linear(inner_dim, dim, bias=bias)

    def forward(self, x, pre_norm, residual, residual_scale):
        weight = kernels.cast_to(pre_norm.weight, dtype=x.dtype, device=x.device)
        gate, up = self.w1(_rms_norm(x, weight, pre_norm.eps)).chunk(2, dim=-1)
        return torch.addcmul(residual, self.w2(F.silu(gate).mul_(up)), residual_scale)


class Attention(nn.Module):
    def __init__(self, heads, dim_head, bias=True, eps=1e-5):
        super().__init__()
        self.dim_head = dim_head
        self.heads = heads
        self.eps = eps
        inner_dim = dim_head * heads
        # unit scale for the fused rms+rope kernel; the qk norms are not affine
        self.register_buffer("qk_norm_scale", torch.ones(dim_head), persistent=False)
        self.to_qkv = nn.Linear(inner_dim, inner_dim * 3, bias=bias)
        self.to_out = nn.Linear(inner_dim, inner_dim, bias=bias)

    def forward(self, x, rotary_pos_emb, pre_norm, residual, residual_scale):
        batch_size, seq_len, _ = x.shape
        weight = kernels.cast_to(pre_norm.weight, dtype=x.dtype, device=x.device)
        qkv = self.to_qkv(_rms_norm(x, weight, pre_norm.eps))
        qkv = qkv.view(batch_size, seq_len, -1, 3 * self.dim_head)
        query, key, value = torch.chunk(qkv, 3, dim=-1)
        query, key = query.contiguous(), key.contiguous()

        if rotary_pos_emb is not None:
            # fused per-head RMSNorm + partial split-half rope, in place
            kernels.ck.rms_rope_split_half_(query, key, rotary_pos_emb, self.qk_norm_scale.to(query),
                                            epsilon=self.eps, rot_dim=rotary_pos_emb.shape[-3] * 2)
        else:
            query = _rms_norm(query, None, self.eps)
            key = _rms_norm(key, None, self.eps)

        out = kernels.attention(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), self.heads)
        return torch.addcmul(residual, self.to_out(torch.nan_to_num(out)), residual_scale)


class TransformerBlock(nn.Module):
    def __init__(self, heads, dim_head, bias=True, eps=1e-5):
        super().__init__()
        dim = heads * dim_head
        self.norm1 = nn.RMSNorm(dim, elementwise_affine=True, eps=eps)
        self.attn = Attention(heads=heads, dim_head=dim_head, bias=bias, eps=eps)
        self.scale1 = nn.Parameter(torch.empty(dim))
        self.norm2 = nn.RMSNorm(dim, elementwise_affine=True, eps=eps)
        self.ff = FeedForward(dim=dim, bias=bias)
        self.scale2 = nn.Parameter(torch.empty(dim))

    def forward(self, x, rotary_pos_emb=None):
        x = self.attn(x, rotary_pos_emb, pre_norm=self.norm1, residual=x,
                      residual_scale=kernels.cast_to(self.scale1, dtype=x.dtype, device=x.device))
        return self.ff(x, pre_norm=self.norm2, residual=x,
                       residual_scale=kernels.cast_to(self.scale2, dtype=x.dtype, device=x.device))


class ViT3DDecoder(nn.Module):
    def __init__(self, patch_size=16, patch_size_t=4, in_channels=24, out_channels=3, num_layers=36, heads=32,
                 dim_head=64, rope_theta=100.0, rope_dim_ratio=0.75, rope_dims=3, bias=True, eps=1e-5,
                 num_register_tokens=4):
        super().__init__()
        dim = heads * dim_head
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.out_channels = out_channels
        self.num_register_tokens = num_register_tokens

        # rope over the trailing rope_dims latent axes of (t, h, w)
        self.pos_embed = RotaryEmbeddingND(int(dim_head * rope_dim_ratio), rope_theta, n_dim=rope_dims)
        self.x_embedder = nn.Linear(in_channels, dim)
        self.register_tokens = nn.Parameter(torch.empty(1, num_register_tokens, dim))
        # unused at inference; kept so the checkpoint loads without leftover keys
        self.register_buffer("mask_token", torch.empty(1, 1, dim))
        self.transformer_blocks = nn.ModuleList([TransformerBlock(heads=heads, dim_head=dim_head, bias=bias, eps=eps)
                                                 for _ in range(num_layers)])
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=True, eps=eps)
        self.proj_out = nn.Linear(dim, out_channels * patch_size_t * patch_size * patch_size)

    def _token_ids(self, latent_T, latent_H, latent_W, device, dtype):
        return create_token_ids((latent_T, latent_H, latent_W)[-self.pos_embed.n_dim:], device, dtype)

    def forward(self, x):
        B, _, latent_T, latent_H, latent_W = x.shape
        h = self.x_embedder(x.flatten(2).transpose(1, 2))  # [B, T*H*W, C]
        num_patches = h.shape[1]
        num_suffix = 1 + self.num_register_tokens
        registers = kernels.cast_to(self.register_tokens, dtype=h.dtype, device=h.device).expand(B, -1, -1)
        h = torch.cat([h, registers, torch.zeros_like(h[:, 0:1, :])], dim=1)

        img_ids = self._token_ids(latent_T, latent_H, latent_W, x.device, x.dtype).expand(B, -1, -1)
        img_ids = torch.cat([img_ids, torch.zeros((B, num_suffix, self.pos_embed.n_dim), device=x.device, dtype=img_ids.dtype)], dim=1)
        rotary_pos_emb = self.pos_embed(img_ids)

        for block in self.transformer_blocks:
            h = block(h, rotary_pos_emb)

        output = self.proj_out(self.norm_out(h))[:, :num_patches, :]
        output = output.view(B, latent_T, latent_H, latent_W, self.out_channels, self.patch_size_t, self.patch_size, self.patch_size)
        output = output.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        return output.reshape(B, self.out_channels, latent_T * self.patch_size_t, latent_H * self.patch_size, latent_W * self.patch_size)


# The image VAEs (ComfyUI H3ImageVAE and its two subclasses)

# P-H3-Image: rope time coordinate of the decoded frame
PH3_IMAGE_FRAME_POSITION = 2.0 * (0.5 / 7) - 1.0
# the image VAEs condition on a posterior sample; the H3 keyframe recipe's seed
IMAGE_ENCODE_SEED = 42
# the decoders' output projection: 3 channels x one frame x 16 x 16
IMAGE_PROJ_OUT = 3 * 16 * 16
# MiniMax-H3-Image's decoder is 16 heads of 64 wide, P-H3-Image's 32
H3_IMAGE_DECODER_DIM = 16 * 64


class PH3ImageDecoder(ViT3DDecoder):
    """One frame per latent, at a fixed rope time coordinate."""

    def __init__(self, **kwargs):
        super().__init__(patch_size_t=1, **kwargs)
        del self.mask_token

    def _token_ids(self, latent_T, latent_H, latent_W, device, dtype):
        ids = create_token_ids((latent_H, latent_W), device, dtype)
        return torch.cat([torch.full_like(ids[..., :1], PH3_IMAGE_FRAME_POSITION), ids], dim=-1)


class H3ImageVAE(nn.Module):
    """The H3 encoder with a single-frame decoder; subclasses attach the decoder."""

    def __init__(
        self,
        in_channels=3,
        ch=128,
        embed_dim=24,
        z_channels=24,
        ch_mult=(1, 2, 2, 4, 4, 8),
        num_res_blocks=2,
        space_down=(2, 2, 2, 2, 1, 1),
        time_down=(1, 2, 2, 1, 1, 1),
        tile_size=256,
        tile_overlap_min=64,
        tiling=True,
    ):
        super().__init__()
        self.vae_ratio = int(math.prod(space_down))

        # spatial tiling parameters
        self.tiling = tiling
        self.tile_size = tile_size
        self.tile_overlap_min = tile_overlap_min

        self.encoder = EncoderFCN3D(
            ch=ch,
            ch_mult=list(ch_mult),
            space_down=list(space_down),
            time_down=list(time_down),
            num_res_blocks=num_res_blocks,
            in_channels=in_channels,
            z_channels=z_channels,
            double_z=True,
        )
        self.quant_conv = nn.Conv3d(z_channels * 2, 2 * embed_dim, 1)
        self.post_quant_conv = nn.Conv3d(embed_dim, z_channels, 1)
        self.decoder = None

        self.register_buffer("latents_mean", torch.tensor(LATENTS_MEAN))
        self.register_buffer("latents_std", torch.tensor(LATENTS_STD))
        self.register_buffer("pixel_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1, 1), persistent=False)
        self.register_buffer("pixel_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1, 1), persistent=False)

    # single-shot forward

    def _encode_moments(self, x):
        return self.quant_conv(self.encoder(x))

    def _decode_pixels(self, z):
        return self.decoder(self.post_quant_conv(z))

    def _normalize_pixels(self, x):
        return x.add(1.0).mul_(0.5).sub_(self.pixel_mean.to(x)).div_(self.pixel_std.to(x))

    def _finalize_pixels(self, part):
        # raw decoder output -> float32 pixels in [0, 1] (the VAE wrapper's process_output is identity)
        part = part * self.pixel_std.to(device=part.device, dtype=torch.float32)
        return part.add_(self.pixel_mean.to(device=part.device, dtype=torch.float32)).clamp_(0.0, 1.0)

    def _adaptive_encode(self, x):
        if self.tiling:
            return self.tiled_encode(x)
        return self._encode_moments(x)

    def _adaptive_decode(self, z):
        if self.tiling:
            return self.tiled_decode(z)
        return self._decode_pixels(z)

    # spatial tiling

    def split_tiles(self, input_len):
        tile_size = self.tile_size
        if tile_size >= input_len:
            return [0], [input_len], []

        N = math.ceil(input_len / tile_size)
        while True:
            overlaps = [self.tile_overlap_min] * (N - 1)
            remaining = tile_size * N - sum(overlaps) - input_len
            if remaining < 0:
                N += 1
            else:
                break

        remaining_units = remaining // self.vae_ratio
        for i in range(remaining_units):
            overlaps[i % (N - 1)] += self.vae_ratio

        tile_start_idx = [0]
        for i in range(N - 1):
            tile_start_idx.append(tile_start_idx[-1] + tile_size - overlaps[i])

        return tile_start_idx, [tile_size] * N, overlaps

    def blend(self, a, b, blend_extent, dim):
        blend_extent = min(a.shape[dim], b.shape[dim], blend_extent)

        positions = torch.arange(blend_extent, device=b.device, dtype=b.dtype)
        weight_a = 1 - positions / blend_extent
        weight_b = positions / blend_extent

        shape = [1] * a.ndim
        shape[dim] = blend_extent
        weight_a = weight_a.view(shape)
        weight_b = weight_b.view(shape)

        slice_a = [slice(None)] * a.ndim
        slice_a[dim] = slice(-blend_extent, None)
        slice_b = [slice(None)] * b.ndim
        slice_b[dim] = slice(0, blend_extent)

        blended = a[tuple(slice_a)] * weight_a + b[tuple(slice_b)] * weight_b

        if blend_extent < b.shape[dim]:
            slice_b_rest = [slice(None)] * b.ndim
            slice_b_rest[dim] = slice(blend_extent, None)
            return torch.cat([blended, b[tuple(slice_b_rest)]], dim=dim)
        return blended

    def tiled_encode(self, x):
        height, width = x.shape[-2], x.shape[-1]
        y_idx, y_len, y_overlap = self.split_tiles(height)
        x_idx, x_len, x_overlap = self.split_tiles(width)

        rows = []
        for i_pos, i_len in zip(y_idx, y_len):
            row = []
            for j_pos, j_len in zip(x_idx, x_len):
                tile = x[..., i_pos:i_pos + i_len, j_pos:j_pos + j_len]
                row.append(self._encode_moments(tile))
            rows.append(row)

        latent_y_overlap = [o // self.vae_ratio for o in y_overlap]
        latent_x_overlap = [o // self.vae_ratio for o in x_overlap]

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.blend(rows[i - 1][j], tile, latent_y_overlap[i - 1], dim=-2)
                if j > 0:
                    tile = self.blend(row[j - 1], tile, latent_x_overlap[j - 1], dim=-1)
                if i < len(rows) - 1:
                    tile = tile[..., :-latent_y_overlap[i], :]
                if j < len(row) - 1:
                    tile = tile[..., :, :-latent_x_overlap[j]]
                result_row.append(tile)
            result_rows.append(torch.cat(result_row, dim=-1))
        return torch.cat(result_rows, dim=-2)

    def _decode_tile_row(self, z_row, x_idx, x_len):
        # a few tiles per decoder call (~7%); each extra tile holds ~100 MB of activations
        free = kernels.free_memory(z_row.device)
        batch = int(max(1, min(4, free // (128 * 2**20 * z_row.shape[0]))))
        slices = [z_row[..., j_pos // self.vae_ratio:(j_pos + j_len) // self.vae_ratio] for j_pos, j_len in zip(x_idx, x_len)]
        for k in range(0, len(slices), batch):
            group = slices[k:k + batch]
            yield from self._decode_pixels(torch.cat(group)).chunk(len(group))

    def tiled_decode(self, z):
        height, width = z.shape[-2] * self.vae_ratio, z.shape[-1] * self.vae_ratio
        y_idx, y_len, y_overlap = self.split_tiles(height)
        x_idx, x_len, x_overlap = self.split_tiles(width)

        # Blended tiles are written straight into a pre-allocated canvas. Tiles blend against
        # their neighbours as already blended, so seams stay continuous where overlaps cross.
        canvas = None
        strip = None
        out_y = 0
        for i, (i_pos, i_len) in enumerate(zip(y_idx, y_len)):
            zi, zl = i_pos // self.vae_ratio, i_len // self.vae_ratio
            tiles = self._decode_tile_row(z[..., zi:zi + zl, :], x_idx, x_len)
            new_strip = None
            left_tail = None
            out_x = 0
            # enumerate would retain the previous tile while the next batch decodes.
            for j in range(len(x_idx)):
                tile = next(tiles)
                if i > 0:
                    tile = self.blend(strip[..., :, x_idx[j]:x_idx[j] + x_len[j]], tile, y_overlap[i - 1], dim=-2)
                if j > 0:
                    tile = self.blend(left_tail, tile, x_overlap[j - 1], dim=-1)
                left_tail = tile[..., :, -x_overlap[j]:].clone() if j < len(x_idx) - 1 else None
                if j < len(x_idx) - 1:
                    tile = tile[..., :, :-x_overlap[j]]
                if canvas is None:
                    canvas = torch.empty(*tile.shape[:-2], height, width, dtype=tile.dtype, device=tile.device)
                if i < len(y_idx) - 1:
                    if new_strip is None:
                        new_strip = torch.empty(*tile.shape[:-2], y_overlap[i], width, dtype=tile.dtype, device=tile.device)
                    new_strip[..., :, out_x:out_x + tile.shape[-1]] = tile[..., -y_overlap[i]:, :]
                    tile = tile[..., :-y_overlap[i], :]
                canvas[..., out_y:out_y + tile.shape[-2], out_x:out_x + tile.shape[-1]].copy_(tile)
                tile_height = tile.shape[-2]
                out_x += tile.shape[-1]
                del tile
            strip = new_strip
            out_y += tile_height
        return canvas

    def encode(self, x, device=None):
        # x: [B, 3, 1, H, W] in [-1, 1] -> a normalized posterior sample [B, 24, 1, H/16, W/16]; every picture draws
        # the same noise, so an edit is repeatable
        if x.ndim == 4:
            x = x.unsqueeze(2)
        if device is None:
            device = x.device
        if x.shape[2] != 1:
            raise ValueError(f"This image VAE encodes single pictures only, got {x.shape[2]} frames")
        moments = self._adaptive_encode(self._normalize_pixels(x.to(device)))
        mean, logvar = moments.float().chunk(2, dim=1)
        noise = torch.randn(mean.shape, generator=torch.Generator().manual_seed(IMAGE_ENCODE_SEED)).to(mean.device)
        latents = torch.addcmul(mean, torch.exp(0.5 * logvar.clamp(-30.0, 20.0)), noise)
        latents_mean = self.latents_mean.view(1, -1, 1, 1, 1).to(latents)
        latents_std = self.latents_std.view(1, -1, 1, 1, 1).to(latents)
        return (latents - latents_mean) / latents_std

    def decode(self, z, output_buffer=None):
        # z: [B, 24, 1, H_lat, W_lat] normalized latents -> float32 pixels [B, 3, 1, H, W] in [0, 1]
        if z.shape[2] != 1:
            raise ValueError(f"This image VAE decodes single-frame image latents only, got {z.shape[2]} latent frames")
        latents_mean = self.latents_mean.view(1, -1, 1, 1, 1).to(z)
        latents_std = self.latents_std.view(1, -1, 1, 1, 1).to(z)
        dec = self._finalize_pixels(self._adaptive_decode(z * latents_std + latents_mean))
        if output_buffer is None:
            return dec
        output_buffer.copy_(dec)
        return output_buffer


class PH3ImageVAE(H3ImageVAE):
    """P-H3-Image: a 3D-rope ViT decoder at a fixed time coordinate."""

    def __init__(self, num_layers=36, **kwargs):
        super().__init__(**kwargs)
        self.decoder = PH3ImageDecoder(patch_size=self.vae_ratio, in_channels=self.post_quant_conv.out_channels,
                                       num_layers=num_layers)


class MiniMaxH3ImageVAE(H3ImageVAE):
    """MiniMax-H3-Image: a 2D ViT decoder (16 heads, 2D rope over half the head) behind a 2D post-quant conv."""

    def __init__(self, num_layers=24, **kwargs):
        super().__init__(**kwargs)
        channels = self.post_quant_conv.out_channels
        self.post_quant_conv = nn.Conv2d(channels, channels, 1)
        self.decoder = ViT3DDecoder(patch_size=self.vae_ratio, patch_size_t=1, in_channels=channels, num_layers=num_layers,
                                    heads=16, rope_dim_ratio=0.5, rope_dims=2)

    def _decode_pixels(self, z):
        # the 2D post-quant conv takes the single frame; the decoder keeps the [B, C, T, H, W] layout with T = 1
        return self.decoder(self.post_quant_conv(z.squeeze(2)).unsqueeze(2))
