"""Native backend checks on CPU: layouts against the real checkpoint headers, small forward passes and conversions.

The fixture holds only tensor names, dtypes and shapes of the published files (no weights). Needs torch and
comfy-kitchen; skipped in an environment without them.
"""

import gzip
import json
import unittest
from pathlib import Path

try:
    import comfy_kitchen  # noqa: F401
    import torch
except ImportError:
    torch = None

HEADERS = Path(__file__).with_name("fixtures") / "h3_headers.json.gz"
AUX = (".comfy_quant", ".weight_scale", ".input_scale", ".weight_s_channel", ".weight_s_rel", ".weight_codebook",
       ".comfy_attention.config")
DIT_FILES = ("minimax_h3_fl2va_pruned_int8_convrot", "minimax_h3_fl2va_int8_convrot",
             "minimax_h3_fl2va_pruned_w6a8", "minimax_h3_fl2va_pruned_fp8_scaled",
             "p-h3-image_bf16", "p-h3-image_w4a8")
H3_IMAGE_FILES = ("minimax_h3_image_pruned_bf16", "minimax_h3_image_pruned_int8_convrot", "minimax_h3_image_pruned_w4a8")


def headers():
    # stored compactly as name -> [dtype, shape]; returned in the safetensors header form
    with gzip.open(HEADERS, "rt", encoding="utf-8") as f:
        data = json.load(f)
    return {name: {k: v if k == "__metadata__" else {"dtype": v[0], "shape": v[1]} for k, v in header.items()}
            for name, header in data.items()}


def meta_state_dict(header):
    return {k: torch.empty(v["shape"], device="meta") for k, v in header.items() if k != "__metadata__"}


def layout_problems(module, header):
    """Missing, unexpected and mismatched keys of a module against a checkpoint header (quantization aside)."""
    file_keys = {k: v for k, v in header.items() if k != "__metadata__" and not k.endswith(AUX)}
    quantized = {k[: -len(".comfy_quant")] for k in header if k.endswith(".comfy_quant")}
    mine = {k: tuple(v.shape) for k, v in module.state_dict().items()}
    problems = [f"missing {k}" for k in sorted(set(mine) - set(file_keys))]
    problems += [f"unexpected {k}" for k in sorted(set(file_keys) - set(mine))]
    for k in sorted(set(mine) & set(file_keys)):
        shape = tuple(file_keys[k]["shape"])
        # quantized weights may be packed along the input axis; the output axis always matches
        same = shape[0] == mine[k][0] if k.rsplit(".", 1)[0] in quantized and k.endswith(".weight") else shape == mine[k]
        if not same:
            problems.append(f"shape {k}: {mine[k]} != {shape}")
    return problems


@unittest.skipIf(torch is None, "needs torch and comfy-kitchen")
class CheckpointLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.headers = headers()

    def test_dit_matches_every_published_build(self):
        from pi_h3_image.native.detect import detect
        from pi_h3_image.native.transformer import MiniMaxH3Model
        for name in DIT_FILES:
            with self.subTest(name):
                header = self.headers[name]
                config = detect(header, metadata=header.get("__metadata__"))
                with torch.device("meta"):
                    model = MiniMaxH3Model(**config)
                self.assertEqual(layout_problems(model, header), [])
                self.assertEqual(config["num_layers"], 50)
                self.assertEqual("adaln_curve_grid" in config, "pruned" in name or "p-h3" in name)
        # P-H3-Image's curve table is finer than Comfy-Org's pruned one; both are read from the shape
        from pi_h3_image.native.detect import detect
        self.assertEqual((detect(self.headers["p-h3-image_bf16"])["adaln_curve_grid"], detect(self.headers["p-h3-image_bf16"])["time_embed_dim"]), (4097, 16))
        self.assertEqual(detect(self.headers["minimax_h3_fl2va_pruned_int8_convrot"])["adaln_curve_grid"], 1025)

    def test_minimax_h3_image_matches_every_published_build(self):
        from pi_h3_image.native.detect import H3_IMAGE_PDD_SIGMAS, detect
        from pi_h3_image.native.transformer import MiniMaxH3Model
        for name in H3_IMAGE_FILES:
            with self.subTest(name):
                header = self.headers[name]
                config = detect(header, metadata=header.get("__metadata__"))
                with torch.device("meta"):
                    model = MiniMaxH3Model(**config)
                self.assertEqual(layout_problems(model, header), [])
                # no audio stream, six output heads on the checkpoint's own sigma grid, the pruned adaln curve
                self.assertEqual((config["audio_latents_dim"], config["pdd_steps"], config["adaln_curve_grid"]), (None, 6, 1025))
                self.assertEqual(config["pdd_sigmas"], json.loads(header["__metadata__"]["config"])["transformer"]["pdd_sigmas"])
                self.assertEqual(tuple(model.final_layer.video_out.weight.shape), (6 * 96, 5376))
                self.assertIsNone(model.final_layer.audio_out)
                self.assertFalse(hasattr(model, "audio_patch_proj"))
                # without the file's metadata (Forge detects from the tensors) the published grid stands in
                self.assertEqual(detect(header)["pdd_sigmas"], list(H3_IMAGE_PDD_SIGMAS))
        self.assertEqual(self.headers[H3_IMAGE_FILES[0]]["__metadata__"]["config"].count("pdd_sigmas"), 1)
        # the audio-video checkpoints have one head and no grid
        config = detect(self.headers["p-h3-image_bf16"])
        self.assertEqual((config["pdd_steps"], config["audio_latents_dim"], "pdd_sigmas" in config), (1, 32, False))

    def test_minimax_h3_image_vae_matches_the_published_file(self):
        from pi_h3_image.models import H3_IMAGE, _role, _variant
        from pi_h3_image.native import vae
        from pi_h3_image.native.image_vae import MiniMaxH3ImageVAE
        header = self.headers["minimax_h3_image_vae_fp16"]
        self.assertTrue(vae.is_h3_vae(header) and vae.is_image_vae(header) and vae.is_h3_image_vae(header))
        self.assertFalse(vae.is_ph3_vae(header))
        self.assertTrue(vae.is_ph3_vae(self.headers["p-h3-image_vae_bf16"]))
        self.assertFalse(vae.is_h3_image_vae(self.headers["p-h3-image_vae_bf16"]))
        self.assertEqual(vae.decoder_layers(header), 24)
        # the release file has no latent statistics: the loader adds H3's
        converted = vae.with_latent_stats(meta_state_dict(header))
        with torch.device("meta"):
            model = MiniMaxH3ImageVAE(num_layers=24)
        self.assertEqual({k: tuple(v.shape) for k, v in model.state_dict().items()},
                         {k: tuple(v.shape) for k, v in converted.items()})
        keys = set(header) - {"__metadata__"}
        self.assertEqual((_role(keys), _variant("vae", header, keys)), ("vae", H3_IMAGE))
        dit = self.headers[H3_IMAGE_FILES[1]]
        keys = set(dit) - {"__metadata__"}
        self.assertEqual((_role(keys), _variant("dit", dit, keys)), ("dit", H3_IMAGE))

    def test_detection_ignores_other_models(self):
        from pi_h3_image.native.detect import detect
        self.assertIsNone(detect(self.headers["minimax_h3_video_vae_fp16"]))

    def test_the_video_vae_is_told_apart_from_the_image_vaes(self):
        # the same encoder, a decoder that makes four frames per latent: recognized, and refused by the loader
        from pi_h3_image.native import vae
        for name in ("minimax_h3_video_vae_fp16", "minimax_original_video_vae"):
            header = self.headers[name]
            self.assertTrue(vae.is_h3_vae(header))
            self.assertFalse(vae.is_image_vae(header) or vae.is_ph3_vae(header) or vae.is_h3_image_vae(header))
            self.assertEqual(vae.decoder_layers(header), 36)
        self.assertFalse(vae.is_h3_vae({"pre_block.attn.zero_k_bias": None}))
        self.assertIn("video VAE", vae.VIDEO_VAE)

    def test_ph3_image_vae_matches_both_published_builds(self):
        from pi_h3_image.native import vae
        from pi_h3_image.native.image_vae import PH3ImageVAE
        for name in ("p-h3-image_vae_bf16", "p-h3-image_vae_int8_convrot"):
            with self.subTest(name), torch.device("meta"):
                header = self.headers[name]
                self.assertTrue(vae.is_h3_vae(header) and vae.is_ph3_vae(header))
                self.assertEqual(layout_problems(PH3ImageVAE(num_layers=vae.decoder_layers(header)), header), [])
        self.assertFalse(vae.is_ph3_vae(self.headers["minimax_h3_video_vae_fp16"]))

    def test_text_encoders_are_told_apart(self):
        from pi_h3_image.models import H3, PH3_IMAGE, _role, _variant
        for name, variant in (("qwen3vl_32b_minimax_h3_int8_convrot", H3), ("p-h3-image_text_encoder_bf16", PH3_IMAGE)):
            keys = set(self.headers[name]) - {"__metadata__"}
            self.assertEqual((_role(keys), _variant("text_encoder", self.headers[name], keys)), ("text_encoder", variant))
        for name, variant in (("minimax_h3_video_vae_fp16", H3), ("p-h3-image_vae_bf16", PH3_IMAGE)):
            keys = set(self.headers[name]) - {"__metadata__"}
            self.assertEqual((_role(keys), _variant("vae", self.headers[name], keys)), ("vae", variant))
        # the 4B encoder: 36 layers of 2560, the vision tower of Forge Neo's own 4B model, and the connector
        te = self.headers["p-h3-image_text_encoder_bf16"]
        self.assertEqual(te["model.language_model.embed_tokens.weight"]["shape"], [151936, 2560])
        self.assertEqual(max(int(k.split(".")[3]) for k in te if k.startswith("model.language_model.layers.")) + 1, 36)
        self.assertEqual(te["text_connector.proj.weight"], {"dtype": "F32", "shape": [5120, 5 * 2560]})


@unittest.skipIf(torch is None, "needs torch and comfy-kitchen")
class ConversionTests(unittest.TestCase):
    def test_quantized_vae_keeps_its_quantization(self):
        from pi_h3_image.native import vae
        sd = {"decoder.blocks.0.attn.to_qkv.comfy_quant": torch.zeros(1),
              "decoder.blocks.0.attn.to_qkv.weight_scale": torch.ones(1)}
        self.assertTrue(vae.is_quantized(sd))
        self.assertFalse(vae.is_quantized({"decoder.blocks.0.attn.to_qkv.weight": torch.ones(1)}))
        converted = vae.with_latent_stats({"decoder.proj_out.comfy_quant": torch.zeros(1)})
        self.assertIn("decoder.proj_out.comfy_quant", converted)
        self.assertEqual(sorted(converted)[-2:], ["latents_mean", "latents_std"])


def tiny_dit(curve, image=False):
    """A toy DiT; image=True makes MiniMax-H3-Image's: no audio stream, six output heads on its sigma grid."""
    from pi_h3_image.native.detect import H3_IMAGE_PDD_SIGMAS
    from pi_h3_image.native.transformer import MiniMaxH3Model
    torch.manual_seed(0)
    extra = dict(audio_latents_dim=None, pdd_steps=6, pdd_sigmas=list(H3_IMAGE_PDD_SIGMAS)) if image else {}
    # 128-dim heads: the rope rotates 96 of them, as in the full model
    model = MiniMaxH3Model(hidden_size=256, num_layers=2, token_refiner_num_layers=1, num_attention_heads=2,
                           attention_head_dim=128, ffn_hidden_size=96, text_dim=48, time_embed_hidden_size=256,
                           time_embed_dim=16, adaln_curve_grid=curve, **extra)
    with torch.no_grad():
        for p in model.parameters():
            torch.nn.init.normal_(p, std=0.02)
        model.rope.inv_freq.copy_(1.0 / (10000 ** (torch.arange(16) / 16)))
        if curve:
            torch.nn.init.normal_(model.adaln_t_table)
    return model.requires_grad_(False)


@unittest.skipIf(torch is None, "needs torch and comfy-kitchen")
class ForwardTests(unittest.TestCase):
    def test_streams_round_trip_through_the_packed_latent(self):
        from pi_h3_image.native.streams import still_shapes
        shapes = still_shapes(width=96, height=64)
        self.assertEqual(shapes.video, (1, 24, 1, 4, 6))
        self.assertEqual(shapes.audio, (1, 32, 2, 2))
        video, audio = torch.randn(shapes.video), torch.randn(shapes.audio)
        packed = shapes.pack(video, audio)
        self.assertEqual(packed.shape, (1, 1, 1, video.numel() + audio.numel()))
        self.assertEqual(shapes.size, video.numel() + audio.numel())
        v, a = shapes.unpack(packed)
        self.assertTrue(torch.equal(v, video) and torch.equal(a, audio))

    def test_both_adaln_forms_run_and_refined_text_is_reused(self):
        from pi_h3_image.native.layout import PackedLayout
        for curve in (None, 17):
            with self.subTest(curve=curve), torch.inference_mode():
                model = tiny_dit(curve)
                video, audio, context = torch.randn(1, 24, 1, 6, 10), torch.randn(1, 32, 2, 2), torch.randn(1, 7, 48)
                t = torch.tensor([700.0])
                payload = {"layout": PackedLayout(7, 1, 6, 10, 2)}
                raw = model.forward_streams([video, audio], t, context, minimax_payload=payload)
                refined = model.forward_streams([video, audio], t, model.preprocess_text_embeds(context), minimax_payload=payload)
                self.assertEqual([tuple(o.shape) for o in raw], [(1, 24, 1, 6, 10), (1, 32, 2, 2)])
                self.assertTrue(all(torch.isfinite(o).all() for o in raw))
                self.assertTrue(all(torch.allclose(a, b, atol=1e-5) for a, b in zip(raw, refined)))

    def test_packed_forward_matches_the_stream_forward(self):
        from pi_h3_image.native.streams import Generation, StreamShapes
        with torch.inference_mode():
            model = tiny_dit(17)
            shapes = StreamShapes(video=(1, 24, 1, 6, 10), audio=(1, 32, 2, 2))
            model.generation = Generation(shapes=shapes, seed=1)
            video, audio, context = torch.randn(shapes.video), torch.randn(shapes.audio), torch.randn(2, 7, 48)
            t = torch.tensor([700.0, 700.0])
            packed = model(torch.cat([shapes.pack(video, audio)] * 2), t, context)
            v, a = model.forward_streams([video, audio], t[:1], context[:1])
            self.assertTrue(torch.allclose(packed[:1], shapes.pack(v, a), atol=1e-5))
            layout = next(iter(model._layouts.values()))
            self.assertEqual([k for _, _, k in layout.segments], ["text", "audio", "video"])
            # the targets start right after the text; the audio rows sit on the video frame's time step
            (ta, _, _), (tv, _, _) = layout.segments[1:]
            self.assertEqual((layout.position_ids[ta, 0].item(), layout.position_ids[tv, 0].item()), (7.0, 7.0))

    def test_live_preview_takes_the_video_frame(self):
        from pi_h3_image.native.streams import StreamShapes, preview_frame
        shapes = StreamShapes(video=(1, 24, 1, 4, 6), audio=(1, 32, 2, 2))
        video = torch.arange(24 * 4 * 6, dtype=torch.float32).reshape(shapes.video)
        packed = shapes.pack(video, torch.zeros(shapes.audio))
        frame = preview_frame(packed, shapes)
        self.assertEqual(tuple(frame.shape), (1, 24, 4, 6))
        self.assertTrue(torch.equal(frame, video[:, :, 0]))
        # any other latent, or no generation, is left for Forge as it is
        other = torch.zeros(1, 4, 8, 8)
        self.assertIs(preview_frame(other, shapes), other)
        self.assertIs(preview_frame(packed, None), packed)

    def test_text_token_tags_mark_the_vision_blocks(self):
        from pi_h3_image.native.streams import text_token_tags
        self.assertIsNone(text_token_tags(7, []))
        self.assertEqual(text_token_tags(9, [(1, 4), (6, 12)]).tolist(), [1, 0, 0, 0, 1, 1, 0, 0, 0])

    def test_packed_forward_with_references_matches_the_stream_forward(self):
        from pi_h3_image.native.layout import PackedLayout
        from pi_h3_image.native.streams import Generation, StreamShapes, text_token_tags
        with torch.inference_mode():
            model = tiny_dit(17)
            shapes = StreamShapes(video=(1, 24, 1, 6, 10), audio=(1, 32, 2, 2))
            # reference pictures keep their own size: a portrait and a landscape reference for a landscape picture
            refs = [{"kind": "image", "latent_h": 8, "latent_w": 4, "latent": torch.randn(1, 24, 1, 8, 4)},
                    {"kind": "image", "latent_h": 6, "latent_w": 10, "latent": torch.randn(1, 24, 1, 6, 10)}]
            spans = [(0, 3), (4, 6)]
            video, audio, context = torch.randn(shapes.video), torch.randn(shapes.audio), torch.randn(1, 9, 48)
            t = torch.tensor([700.0])
            model.generation = Generation(shapes=shapes, seed=1, refs=refs, vision_spans=spans)
            packed = model(shapes.pack(video, audio), t, context)
            layout = model._layout(9, shapes, refs=refs)
            self.assertEqual([k for _, _, k in layout.segments], ["text", "ref_img", "ref_img", "audio", "video"])
            # each picture packs on its own grid, one time step apart, and the targets start after them
            (a1, b1, _), (a2, b2, _) = [s for s in layout.segments if s[2] == "ref_img"]
            self.assertEqual((b1 - a1, b2 - a2), (8 * 4 // 4, 6 * 10 // 4))
            self.assertEqual(layout.position_ids[a1, 0].item(), 9.0)
            self.assertEqual(layout.position_ids[a2, 0].item(), 10.0)
            tv, _, _ = next(s for s in layout.segments if s[2] == "video")
            self.assertEqual(layout.position_ids[tv, 0].item(), 11.0)
            payload = {"seed": 1, "refs": refs, "cond_video_latents": [r["latent"] for r in refs],
                       "text_token_tags": text_token_tags(9, spans), "layout": PackedLayout(9, 1, 6, 10, 2, refs=refs)}
            v, a = model.forward_streams([video, audio], t, context, minimax_payload=payload)
            self.assertTrue(torch.allclose(packed, shapes.pack(v, a), atol=1e-5))
            # the references change the prediction (the toy DiT couples weakly: a reference moves it by about 1e-4),
            # and other references get their own layout
            model.generation = Generation(shapes=shapes, seed=1)
            plain = model(shapes.pack(video, audio), t, context)
            self.assertGreater(float((packed - plain).abs().max()), 1e-6)
            self.assertIsNot(model._layout(9, shapes, refs=refs[:1]), layout)

    def test_the_image_model_packs_text_references_and_the_picture_only(self):
        from pi_h3_image.native.layout import PackedLayout
        from pi_h3_image.native.streams import Generation, still_shapes
        with torch.inference_mode():
            model = tiny_dit(17, image=True)
            shapes = still_shapes(width=160, height=96, audio=False)
            self.assertEqual((shapes.video, shapes.audio, shapes.size), ((1, 24, 1, 6, 10), (1, 32, 2, 0), 24 * 6 * 10))
            video, context = torch.randn(shapes.video), torch.randn(1, 7, 48)
            packed = shapes.pack(video, torch.zeros(shapes.audio))
            self.assertEqual(tuple(packed.shape), (1, 1, 1, shapes.size))
            sigmas = torch.tensor(model.pdd_sigmas)
            refs = [{"kind": "image", "latent_h": 8, "latent_w": 4, "latent": torch.randn(1, 24, 1, 8, 4)}]
            model.generation = Generation(shapes=shapes, seed=1, refs=refs, vision_spans=[(0, 3)])
            out = model(packed, sigmas[2:3] * 1000, context, {"sampling_sigmas": sigmas})
            self.assertEqual(out.shape, packed.shape)
            layout = model._layout(7, shapes, refs=refs)
            self.assertEqual([k for _, _, k in layout.segments], ["text", "ref_img", "video"])
            self.assertEqual((layout.audio_pos.numel(), layout.signature[-1]), (0, 0))
            # the picture starts one time step after its reference, as with the audio-video checkpoints
            (ra, _, _), (va, _, _) = layout.segments[1:]
            self.assertEqual((layout.position_ids[ra, 0].item(), layout.position_ids[va, 0].item()), (7.0, 8.0))
            self.assertEqual(PackedLayout(7, 1, 6, 10, 0).seq_len, 7 + 15)

    def test_a_step_takes_the_heads_of_the_grid_intervals_it_spans(self):
        from pi_h3_image.native import dit
        from pi_h3_image.native.detect import H3_IMAGE_PDD_SIGMAS
        grid = list(H3_IMAGE_PDD_SIGMAS)
        six = torch.tensor(grid)
        # the checkpoint's own schedule: one head per step
        self.assertEqual([dit.pdd_span(grid, float(six[i]), six) for i in range(6)], [(i, i + 1) for i in range(6)])
        # three steps span two heads each; an off-grid sigma takes the nearest grid point, one interval at least
        three = six[::2]
        self.assertEqual([dit.pdd_span(grid, float(three[i]), three) for i in range(3)], [(0, 2), (2, 4), (4, 6)])
        twelve = torch.tensor([grid[0], 0.99, *grid[1:]])
        self.assertEqual(dit.pdd_span(grid, 0.99, twelve), (0, 1))
        self.assertEqual(dit.pdd_span(grid, 0.0, six), (5, 6))
        with self.assertRaisesRegex(ValueError, "sigma schedule"):
            dit.pdd_span(grid, 1.0, None)
        # row block 0 is a full head, later blocks are offsets from it; a step blends its heads by their dt
        torch.manual_seed(0)
        heads, hidden, out = 6, 8, 4
        full = torch.randn(heads, out, hidden, dtype=torch.float64)
        bias = torch.randn(heads, out, dtype=torch.float64)
        bank = torch.nn.Linear(hidden, heads * out).double()
        with torch.no_grad():
            bank.weight.copy_(torch.cat([full[:1], full[1:] - full[:1]]).reshape(heads * out, hidden))
            bank.bias.copy_(torch.cat([bias[:1], bias[1:] - bias[:1]]).reshape(-1))
        h = torch.randn(3, hidden, dtype=torch.float64)
        for start, stop in ((0, 1), (3, 4), (2, 4), (0, 6)):
            dt = torch.tensor([grid[k] - grid[k + 1] for k in range(start, stop)], dtype=torch.float64)
            w = dt / dt.sum()
            expected = torch.nn.functional.linear(h, torch.einsum("n,noi->oi", w, full[start:stop]),
                                                  torch.einsum("n,no->o", w, bias[start:stop]))
            torch.testing.assert_close(dit._pdd_head(bank, h, grid, start, stop), expected)

    def test_the_layout_cache_is_bounded(self):
        from pi_h3_image.native.streams import StreamShapes
        from pi_h3_image.native.transformer import LAYOUT_CACHE_SIZE
        model = tiny_dit(None)
        shapes = StreamShapes(video=(1, 24, 1, 6, 10), audio=(1, 32, 2, 2))
        for text_len in range(LAYOUT_CACHE_SIZE + 2):
            model._layout(text_len + 1, shapes)
        self.assertLessEqual(len(model._layouts), LAYOUT_CACHE_SIZE)


@unittest.skipIf(torch is None, "needs torch and comfy-kitchen")
class VaeTests(unittest.TestCase):
    def test_the_ph3_image_vae_samples_its_posterior_and_decodes_one_frame(self):
        from pi_h3_image.native.image_vae import PH3_IMAGE_FRAME_POSITION, PH3ImageVAE
        torch.manual_seed(0)
        model = PH3ImageVAE(ch=32, num_layers=1)
        with torch.no_grad():
            for p in model.parameters():
                torch.nn.init.normal_(p, std=0.02)
        model.requires_grad_(False)
        self.assertNotIn("decoder.mask_token", model.state_dict())
        self.assertEqual(tuple(model.decoder.proj_out.weight.shape), (3 * 16 * 16, 2048))
        with torch.inference_mode():
            picture = torch.rand(1, 3, 1, 64, 96) * 2 - 1
            z = model.encode(picture)
            self.assertEqual(tuple(z.shape), (1, 24, 1, 4, 6))
            # the posterior sample is seeded: the same picture gives the same latent, and not the mean
            self.assertTrue(torch.equal(z, model.encode(picture)))
            mean = (model.quant_conv(model.encoder(model._normalize_pixels(picture))).float().chunk(2, dim=1)[0]
                    - model.latents_mean.view(1, -1, 1, 1, 1)) / model.latents_std.view(1, -1, 1, 1, 1)
            self.assertFalse(torch.allclose(z, mean))
            out = model.decode(z)
            self.assertEqual(tuple(out.shape), (1, 3, 1, 64, 96))
            self.assertTrue(0.0 <= float(out.min()) and float(out.max()) <= 1.0)
            ids = model.decoder._token_ids(1, 4, 6, "cpu", torch.float32)
            self.assertEqual(tuple(ids.shape), (1, 24, 3))
            self.assertTrue(bool((ids[..., 0] == PH3_IMAGE_FRAME_POSITION).all()))
            with self.assertRaisesRegex(ValueError, "single-frame"):
                model.decode(torch.zeros(1, 24, 2, 4, 6))
            with self.assertRaisesRegex(ValueError, "single pictures"):
                model.encode(torch.zeros(1, 3, 5, 64, 96))
            # a picture without the frame axis takes the same path
            self.assertTrue(torch.equal(model.encode(torch.zeros(1, 3, 64, 96)), model.encode(torch.zeros(1, 3, 1, 64, 96))))

    def test_the_minimax_h3_image_vae_decodes_one_picture_through_its_2d_decoder(self):
        from pi_h3_image.native.image_vae import MiniMaxH3ImageVAE
        torch.manual_seed(0)
        model = MiniMaxH3ImageVAE(ch=32, num_layers=1)
        with torch.no_grad():
            for p in model.parameters():
                torch.nn.init.normal_(p, std=0.02)
        model.requires_grad_(False)
        self.assertEqual(tuple(model.post_quant_conv.weight.shape), (24, 24, 1, 1))
        self.assertEqual(tuple(model.decoder.proj_out.weight.shape), (3 * 16 * 16, 1024))
        self.assertIn("decoder.mask_token", model.state_dict())
        with torch.inference_mode():
            picture = torch.rand(1, 3, 1, 64, 96) * 2 - 1
            z = model.encode(picture)
            self.assertEqual(tuple(z.shape), (1, 24, 1, 4, 6))
            self.assertTrue(torch.equal(z, model.encode(picture)))  # the seeded posterior sample
            out = model.decode(z)
            self.assertEqual(tuple(out.shape), (1, 3, 1, 64, 96))
            self.assertTrue(0.0 <= float(out.min()) and float(out.max()) <= 1.0)
            # 2D rope: (h, w) coordinates only, 8 frequencies an axis over half of the 64-wide head
            ids = model.decoder._token_ids(1, 4, 6, "cpu", torch.float32)
            self.assertEqual(tuple(ids.shape), (1, 24, 2))
            self.assertEqual(tuple(model.decoder.pos_embed(ids).shape), (1, 24, 1, 16, 2, 2))
            with self.assertRaisesRegex(ValueError, "single-frame"):
                model.decode(torch.zeros(1, 24, 2, 4, 6))

    def test_encoder_layers_work_with_offloaded_weights(self):
        # issue #4: when VRAM is short Forge keeps these weights in fp32 system RAM and turns on manual cast; a
        # float64 input stands in for the GPU one (both make plain torch layers fail on the stored weights)
        from pi_h3_image.native.image_vae import CausalConv3d, TemporalIsolatedGroupNorm
        torch.manual_seed(0)
        conv, norm = CausalConv3d(32, 32, kernel_size=3, padding=1), TemporalIsolatedGroupNorm(8, 32)
        reference_conv, reference_norm = CausalConv3d(32, 32, kernel_size=3, padding=1), TemporalIsolatedGroupNorm(8, 32)
        reference_conv.load_state_dict(conv.state_dict())
        reference_norm.load_state_dict(norm.state_dict())
        reference_conv.double()
        reference_norm.double()
        for layer in (conv, norm):
            self.assertEqual((layer.weight_function, layer.bias_function), ([], []))
            layer.parameters_manual_cast = True
        x = torch.randn(1, 32, 2, 8, 8, dtype=torch.float64)
        with torch.inference_mode():
            torch.testing.assert_close(norm(x), reference_norm(x))
            torch.testing.assert_close(conv(x, pre_norm=norm, residual=x),
                                       reference_conv(x, pre_norm=reference_norm, residual=x))
        self.assertEqual({conv.weight.dtype, norm.weight.dtype}, {torch.float32})


if __name__ == "__main__":
    unittest.main()
