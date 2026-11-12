"""Packed-sequence geometry of the MiniMax H3 DiT: patching, positions and the segment table.

Ported from ComfyUI comfy/ldm/minimax/model.py. The packed sequence is [text | reference pictures | audio | video];
the targets are a still image's single latent frame and, for the audio-video checkpoints, its audio stub.
MiniMax-H3-Image has no audio stream: [text | reference pictures | image].
"""

import math

import torch

FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FRAME_RESCALE = 5.0 / 3.0
VISUAL_COND_TIMESTEP = 0.999
# the 32B text encoder (text_encoder.py Qwen3VL_32BConfig)
QWEN_HIDDEN = 5120
QWEN_INTERMEDIATE = 25600


def pad_to_patch_size(x, patch_size):
    # pad the trailing (T, H, W) axes up to multiples of the patch, as comfy.ldm.common_dit does
    pads = []
    for size, patch in zip(reversed(x.shape[-len(patch_size):]), reversed(patch_size)):
        pads += [0, (patch - size % patch) % patch]
    if not any(pads):
        return x
    return torch.nn.functional.pad(x, pads, mode="circular")


def patchify_video(latent, patch_size=(1, 2, 2)):
    # [B, C, T, H, W] -> [B*t*h*w, C*pt*ph*pw]
    b, c, t_full, h_full, w_full = latent.shape
    pt, ph, pw = patch_size
    t, h, w = t_full // pt, h_full // ph, w_full // pw
    x = latent.reshape(b, c, t, pt, h, ph, w, pw)
    x = torch.einsum("nctrhpwq->nthwcrpq", x)
    return x.reshape(b * t * h * w, c * pt * ph * pw)


def unpatchify_video(rows, t, h, w, c=24, patch_size=(1, 2, 2)):
    pt, ph, pw = patch_size
    x = rows.reshape(-1, t, h, w, c, pt, ph, pw)
    x = torch.einsum("nthwcrpq->nctrhpwq", x)
    return x.reshape(-1, c, t * pt, h * ph, w * pw)


def pack_audio(latent):
    # [B, C=32, ch=2, T] -> [ch*T, 32] channel-major (ch0 t0..T-1, ch1 t0..T-1)
    return latent[0].permute(1, 2, 0).reshape(latent.shape[2] * latent.shape[3], latent.shape[1])


def unpack_audio(rows, ch=2):
    t = rows.shape[0] // ch
    return rows.reshape(ch, t, rows.shape[-1]).permute(2, 0, 1).unsqueeze(0)


def _axis_from_sqrt_area(dim, patch, sqrt_area):
    # linspace((1 - ratio) / 2, (1 + ratio) / 2, dim // patch, endpoint=False) * 32
    ratio = dim / sqrt_area
    n = dim // patch
    return (torch.arange(n, dtype=torch.float64) * (ratio / n) + (1.0 - ratio) / 2.0) * 32.0


def _frame_grid(h, w):
    # area-normalized (h, w) coordinates of one latent frame's 2x2-patch rows
    area = math.sqrt(h * w)
    hh, ww = torch.meshgrid(_axis_from_sqrt_area(h, 2, area), _axis_from_sqrt_area(w, 2, area), indexing="ij")
    return torch.stack([hh.reshape(-1), ww.reshape(-1)], dim=-1), _axis_from_sqrt_area(w, 2, area)


def _video_t_grid(n, origin):
    # origin + exclusive cumsum of the frame spans
    spans = torch.tensor([FRAME_RESCALE * FRAME_PER_TOKEN[k % 5] for k in range(n)], dtype=torch.float64)
    return float(origin) + torch.cat([torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)])


def _audio_grid(cursor, t, w_low, w_high):
    # channel-major stereo rows: t advances per latent frame, w pinned to the grid extremes per stereo channel, h stays 0
    g = torch.zeros(t * 2, 3, dtype=torch.float64)
    g[:, 0] = (cursor + torch.arange(t, dtype=torch.float64)).repeat(2)
    g[:t, 2] = w_low
    g[t:, 2] = w_high
    return g


def _video_grid(vt, frame, cursor):
    g = torch.empty(vt, frame.shape[0], 3, dtype=torch.float64)
    g[:, :, 0] = _video_t_grid(vt, cursor)[:, None]
    g[:, :, 1:] = frame[None]
    return g.reshape(-1, 3)


class _Builder:
    """Accumulates segments, positions and the image/audio row bookkeeping."""

    def __init__(self, text_len):
        self.segments = [("text", text_len)]
        g = torch.zeros(text_len, 3, dtype=torch.float64)
        g[:, 0] = torch.arange(text_len, dtype=torch.float64)
        self.pos = [g]
        self.img_pos, self.img_update = [], []
        # never empty for torch.cat: the image model adds no audio rows
        self.audio_pos, self.audio_update = [torch.zeros(0, dtype=torch.long)], [torch.zeros(0, dtype=torch.bool)]
        self.row = text_len

    def add(self, kind, positions, stream, update):
        n = positions.shape[0]
        self.segments.append((kind, n))
        self.pos.append(positions)
        rows = torch.arange(self.row, self.row + n)
        flags = torch.full((n,), update, dtype=torch.bool)
        if stream == "image":
            self.img_pos.append(rows)
            self.img_update.append(flags)
        else:
            self.audio_pos.append(rows)
            self.audio_update.append(flags)
        self.row += n


class PackedLayout:
    """Static packed-sequence structure for one shape/conditioning signature."""

    def __init__(self, text_len, latent_t, latent_h, latent_w, audio_t, refs=None):
        frame, w_grid = _frame_grid(latent_h, latent_w)
        target_audio_w = (float(w_grid[0]), float(w_grid[-1]))
        b = _Builder(text_len)

        # reference pictures pack between the text and the targets, each on its own grid, one time step apart
        # (ComfyUI MiniMaxH3ReferenceToVideo's ref_blocks); the target timeline starts after them
        cursor = float(text_len)
        for blk in refs or ():
            r_frame, _ = _frame_grid(blk["latent_h"], blk["latent_w"])
            g = torch.empty(r_frame.shape[0], 3, dtype=torch.float64)
            g[:, 0] = cursor
            g[:, 1:] = r_frame
            b.add("ref_img", g, "image", False)
            cursor += 1.0

        # target audio (none for the image model) then target video, always last
        if audio_t:
            b.add("audio", _audio_grid(cursor, audio_t, *target_audio_w), "audio", True)
        b.add("video", _video_grid(latent_t, frame, cursor), "image", True)

        self.seq_len = b.row
        self.position_ids = torch.cat(b.pos)  # [S, 3] float64
        self.img_pos = torch.cat(b.img_pos)
        self.img_update = torch.cat(b.img_update)
        self.audio_pos = torch.cat(b.audio_pos)
        self.audio_update = torch.cat(b.audio_update)
        self.signature = (text_len, latent_t, latent_h, latent_w, audio_t)
        # contiguous segment table (start, stop, kind); kinds: text / ref_img / audio / video
        self.segments = []
        off = 0
        for kind, n in b.segments:
            self.segments.append((off, off + n, kind))
            off += n


def vision_size(height: int, width: int, factor=32, min_pixels=3136, max_pixels=12845056) -> tuple[int, int]:
    """Qwen's resize of a picture: multiples of patch x merge (32), within the pixel budget."""
    h_bar, w_bar = round(height / factor) * factor, round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar, w_bar = math.ceil(height * beta / factor) * factor, math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def prompt_memory(pictures: list[tuple[int, int]]) -> int:
    """Working memory to keep free for one prompt pass of the 32B encoder, given the (height, width) of each picture:
    one merged vision token per 32x32 pixels, and every token carries the bf16 MLP and the fp32 residual stream."""
    tokens = 1024  # the prompt text
    for height, width in pictures:
        h_bar, w_bar = vision_size(height, width)
        tokens += h_bar * w_bar // 1024
    return tokens * (QWEN_INTERMEDIATE * 8 + QWEN_HIDDEN * 32)
