"""Forge Neo's generation order replayed on CPU, with the real engine, VAE and DiT at toy sizes.

Forge's own order (modules/processing.py): scripts.before_process, model load, scripts.process, p.setup_conds
(get_learned_conditioning), then p.sample with process_before_every_sampling, decode_first_stage and
scripts.postprocess. Forge itself is replaced by forge_stubs and the fakes below.
"""

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import forge_stubs
from PIL import Image
from test_contracts import (
    DIT,
    H3_IMAGE_DIT,
    H3_IMAGE_VAE,
    PH3_TE,
    PH3_VAE,
    TE,
    checkpoint,
)

try:
    import comfy_kitchen  # noqa: F401
    import torch
except ImportError:
    torch = None

from pi_h3_image import integration, pictures
from pi_h3_image.contracts import H3Error, raise_pending_error, set_pending_error

WIDTH, HEIGHT = 96, 64


class Script:
    def __init__(self, title, args_from, args_to):
        self._title, self.args_from, self.args_to = title, args_from, args_to

    def title(self):
        return self._title


class Txt2Img:
    def __init__(self, gallery=()):
        self.prompt, self.negative_prompt = "a bird takes off", ""
        self.n_iter, self.batch_size = 1, 1
        self.width, self.height = WIDTH, HEIGHT
        self.enable_hr = self.restore_faces = False
        self.image_mask = None
        self.subseed_strength, self.seed_resize_from_w, self.seed_resize_from_h = 0, -1, -1
        self.seeds, self.subseeds = [123], [0]
        self.distilled_cfg_scale, self.is_api = 3.5, False
        self.extra_generation_params, self.override_settings = {}, {}
        self.clear_prompt_cache = Mock()
        # Script: None; the H3 panel has no inputs of its own; ImageStitch (enable, gallery, maximum side)
        self.script_args = [0, bool(gallery), [(image, None) for image in gallery] or None, 1024]
        self.scripts = types.SimpleNamespace(alwayson_scripts=[Script("MiniMax H3", 1, 1),
                                                               Script(pictures.IMAGE_STITCH, 1, 4)])


class Img2Img(Txt2Img):
    def __init__(self, gallery=(), init=True):
        super().__init__(gallery)
        self.init_images = [Image.new("RGB", (300, 100), "red")] if init else []
        self.resize_mode, self.denoising_strength = 0, 0.75


class FakeUnetPatcher:
    """Forge's UnetPatcher, as far as a script reserves sampling memory: each clone keeps the memory asked so far."""

    def __init__(self, model, memory=0):
        self.model, self.extra_preserved_memory_during_sampling = model, memory

    def clone(self):
        return FakeUnetPatcher(self.model, self.extra_preserved_memory_during_sampling)

    def add_extra_preserved_memory_during_sampling(self, size):
        self.extra_preserved_memory_during_sampling += size


class FakeTextEngine:
    """Records the pictures handed with the prompt; its vision block spans the first three tokens."""

    def __init__(self):
        self.images, self.vision_spans, self.picture_label = [], [], None

    def __call__(self, texts, images=(), picture_label=None):
        self.images, self.picture_label = list(images), picture_label
        self.vision_spans = [(0, 3)] if images else []
        return [torch.randn(7, 48) for _ in texts]


def forge_modules(files):
    info = types.SimpleNamespace(filename=str(files[0]))
    opts = types.SimpleNamespace(sd_model_checkpoint="model", forge_additional_modules=[str(f) for f in files[1:]],
                                 forge_preset="sd")

    class ImageRNG:
        def __init__(self, shape, seeds, **kwargs):
            self.shape, self.seeds = shape, seeds

        def next(self):
            return torch.randn((len(self.seeds), *self.shape))

    modules = forge_stubs.module("modules")
    modules.processing = forge_stubs.module("modules.processing", StableDiffusionProcessingImg2Img=Img2Img)
    modules.shared = forge_stubs.module("modules.shared", opts=opts,
                                        state=types.SimpleNamespace(interrupted=False, skipped=False))
    modules.sd_models = forge_stubs.module("modules.sd_models", get_closet_checkpoint_match=lambda value: info)
    modules.rng = forge_stubs.module("modules.rng", ImageRNG=ImageRNG)
    presets = forge_stubs.module("modules_forge.presets")
    forge = forge_stubs.module("modules_forge", presets=presets,
                               main_entry=forge_stubs.module("modules_forge.main_entry", module_list={}))
    patches = forge_stubs.module("pi_h3_image.native.patches", begin_sampling=Mock(), end_sampling=Mock())
    return {"modules": modules, "modules.processing": modules.processing, "modules.shared": modules.shared,
            "modules.sd_models": modules.sd_models, "modules.rng": modules.rng,
            "modules_forge": forge,
            "modules_forge.presets": presets, "modules_forge.main_entry": forge.main_entry,
            "pi_h3_image.native.patches": patches}


@unittest.skipIf(torch is None, "needs torch and comfy-kitchen")
class FlowTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = self.root = Path(tmp.name)
        files = [checkpoint(root / f"{name}.safetensors", tensors) for name, tensors in
                 (("p-h3-image_w4a8", DIT), ("ph3_te", PH3_TE), ("ph3_vae", PH3_VAE))]
        stubs = {**forge_stubs.engine_modules(), **forge_modules(files)}
        self.enterContext(patch.dict(sys.modules, stubs))
        for name in ("pi_h3_image.native.engine", "pi_h3_image.native.model", "pi_h3_image.native.text_engine",
                     "pi_h3_image.native.presets"):
            sys.modules.pop(name, None)
        self.addCleanup(set_pending_error, None)
        self.engine = self.make_engine()

    def make_engine(self):
        from test_native import tiny_dit

        from pi_h3_image.native.engine import MiniMaxH3Engine
        from pi_h3_image.native.image_vae import PH3ImageVAE
        torch.manual_seed(0)
        vae = PH3ImageVAE(ch=32, num_layers=1)
        with torch.no_grad():
            for p in vae.parameters():
                torch.nn.init.normal_(p, std=0.02)
        engine = object.__new__(MiniMaxH3Engine)
        dit = tiny_dit(17)
        engine.forge_objects = types.SimpleNamespace(
            vae=types.SimpleNamespace(patcher=None, device="cpu", vae_dtype=torch.float32,
                                      first_stage_model=vae.requires_grad_(False)),
            unet=FakeUnetPatcher(types.SimpleNamespace(diffusion_model=dit)),
            clip=types.SimpleNamespace(patcher=None))
        engine.text_processing_engine_h3 = FakeTextEngine()
        engine.is_h3_image, engine.shift, engine.generation = True, 12.0, None
        engine.references = []
        return engine

    def use_ph3(self, preset="sd"):
        """Select P-H3-Image's text encoder and VAE next to the checkpoint (the set the tests start with)."""
        files = [checkpoint(self.root / f"{name}.safetensors", tensors) for name, tensors in
                 (("ph3_te", PH3_TE), ("ph3_vae", PH3_VAE))]
        sys.modules["modules.shared"].opts.forge_additional_modules = [str(f) for f in files]
        sys.modules["modules.shared"].opts.forge_preset = preset

    def run_until_sampling(self, p):
        """before_process .. process_before_every_sampling, as Forge calls them; returns the conditioning."""
        integration.before_process(p)
        p.sd_model = self.engine
        integration.process(p)
        if isinstance(p, Img2Img):
            # images_tensor_to_samples: the input image, already resized by Forge, in [-1, 1]
            p.init_latent = self.engine.encode_first_stage(torch.full((1, 3, p.height, p.width), 0.25) * 2 - 1)
        cond = self.engine.get_learned_conditioning([p.prompt])
        integration.before_sampling(p, torch.zeros(1, 4, HEIGHT // 8, WIDTH // 8))
        return cond

    def test_img2img_edits_picture_one_with_the_gallery_as_references(self):
        self.use_ph3()
        p = Img2Img(gallery=[Image.new("RGB", (40, 80), "green")])
        p.prompt = "put the hat from <Picture 2> on the person"
        cond = self.run_until_sampling(p)
        self.assertEqual((p.pi_h3_image_request.references, p.denoising_strength), (2, 1.0))
        self.assertEqual(p.prompt, "Edit <Picture 1>. put the hat from <Picture 2> on the person")
        self.assertEqual(p.extra_generation_params["H3 Edit pictures"], 2)
        p.clear_prompt_cache.assert_called()
        # the original pictures, not Forge's resized copy, each within the 96x64 output area on the 32 grid for the
        # VAE and within 1 MP for the text encoder
        self.assertEqual([tuple(r.shape) for r in self.engine.references], [(1, 32, 128, 3), (1, 64, 32, 3)])
        self.assertTrue(torch.allclose(self.engine.references[0][0, 0, 0], torch.tensor([1.0, 0.0, 0.0])))
        self.assertEqual([tuple(i.shape) for i in self.engine.text_processing_engine_h3.images], [(1, 96, 288, 3), (1, 64, 32, 3)])
        self.assertEqual(self.engine.text_processing_engine_h3.picture_label, "<Picture %d>: ")
        refs = self.engine.generation.refs
        self.assertEqual([(r["kind"], r["latent_h"], r["latent_w"], tuple(r["latent"].shape)) for r in refs],
                         [("image", 2, 8, (1, 24, 1, 2, 8)), ("image", 4, 2, (1, 24, 1, 4, 2))])
        # img2img samples from init_latent at full denoise: the packed start replaces the placeholder
        self.assertEqual(p.init_latent.shape, p.modified_noise.shape)
        self.assertEqual(float(p.init_latent.abs().sum()), 0.0)
        dit = self.engine.forge_objects.unet.model.diffusion_model
        out = dit(p.modified_noise, torch.tensor([900.0]), cond[0].unsqueeze(0))
        self.assertEqual(out.shape, p.modified_noise.shape)
        self.assertTrue(torch.isfinite(out).all())
        layout = next(iter(dit._layouts.values()))
        self.assertEqual([k for _, _, k in layout.segments], ["text", "ref_img", "ref_img", "audio", "video"])
        # the pictures' rows join the sequence the reservation is sized from
        from pi_h3_image.native import engine as engine_module
        self.assertEqual(engine_module.sampling_tokens(self.engine.generation),
                         6 + 4 + (2 * 8 // 4) + (4 * 2 // 4) + engine_module.TEXT_TOKENS_RESERVED)
        # the next txt2img request drops the pictures
        p = Txt2Img()
        self.run_until_sampling(p)
        self.assertEqual((self.engine.references, self.engine.generation.refs), ([], []))

    def use_h3_image(self):
        """Select MiniMax-H3-Image: its checkpoint (no audio stream, six output heads), the H3 text encoder and its
        VAE; the engine gets the matching toy DiT."""
        from test_native import tiny_dit
        files = [checkpoint(self.root / f"{name}.safetensors", tensors) for name, tensors in
                 (("h3image", H3_IMAGE_DIT), ("encoder", TE), ("h3image_vae", H3_IMAGE_VAE))]
        sys.modules["modules.sd_models"].get_closet_checkpoint_match = lambda value: types.SimpleNamespace(filename=str(files[0]))
        sys.modules["modules.shared"].opts.forge_additional_modules = [str(f) for f in files[1:]]
        self.engine.forge_objects.unet = FakeUnetPatcher(types.SimpleNamespace(diffusion_model=tiny_dit(17, image=True)))

    def test_minimax_h3_image_samples_the_picture_alone_on_its_six_heads(self):
        self.use_h3_image()
        p = Txt2Img()
        cond = self.run_until_sampling(p)
        self.assertEqual((p.pi_h3_image_components.profile.name, p.extra_generation_params["H3 Model"]), ("h3image", "MiniMax-H3-Image"))
        shapes = self.engine.generation.shapes
        # no audio stub: the packed latent is the picture's latent
        self.assertEqual((shapes.video, shapes.audio), ((1, 24, 1, HEIGHT // 16, WIDTH // 16), (1, 32, 2, 0)))
        self.assertEqual(tuple(p.modified_noise.shape), (1, 1, 1, 24 * (HEIGHT // 16) * (WIDTH // 16)))
        from pi_h3_image.native import engine as engine_module
        self.assertEqual(engine_module.sampling_tokens(self.engine.generation), 6 + engine_module.TEXT_TOKENS_RESERVED)
        # each of the six steps takes its own head: the schedule Forge hands the model is the checkpoint's grid
        dit = self.engine.forge_objects.unet.model.diffusion_model
        sigmas = torch.tensor(dit.pdd_sigmas)
        outs = [dit(p.modified_noise, sigmas[i:i + 1] * 1000, cond[0].unsqueeze(0), {"sampling_sigmas": sigmas}) for i in range(6)]
        self.assertTrue(all(o.shape == p.modified_noise.shape and torch.isfinite(o).all() for o in outs))
        layout = next(iter(dit._layouts.values()))
        self.assertEqual([k for _, _, k in layout.segments], ["text", "video"])
        with self.assertRaisesRegex(ValueError, "sampler's sigma schedule"):
            dit(p.modified_noise, sigmas[:1] * 1000, cond[0].unsqueeze(0))
        image = self.engine.decode_first_stage(p.modified_noise)
        self.assertEqual(tuple(image.shape), (1, 3, HEIGHT, WIDTH))

    def test_minimax_h3_image_takes_the_p_h3_image_text_encoder_as_an_experiment(self):
        self.use_h3_image()
        modules = sys.modules["modules.shared"].opts.forge_additional_modules
        modules[0] = str(checkpoint(self.root / "ph3_te.safetensors", PH3_TE))
        p = Txt2Img()
        with patch("builtins.print") as printed:
            self.run_until_sampling(p)
        self.assertTrue(any("MiniMax-H3-Image (experimental: P-H3-Image text encoder)" in str(call.args[0])
                            for call in printed.call_args_list))
        # the checkpoint's recipe, not P-H3-Image's: six steps on its own grid, no audio stub
        self.assertEqual((p.pi_h3_image_components.profile.name, p.extra_generation_params["H3 Experimental"]),
                         ("h3image", "P-H3-Image text encoder"))
        self.assertEqual(self.engine.generation.shapes.audio, (1, 32, 2, 0))
        p = Txt2Img()
        modules[0] = str(self.root / "encoder.safetensors")
        self.run_until_sampling(p)
        self.assertNotIn("H3 Experimental", p.extra_generation_params)

    def test_minimax_h3_image_edits_with_its_own_labels_and_sizes(self):
        self.use_h3_image()
        p = Img2Img()
        p.init_images = [Image.new("RGB", (128, 64), "red")]
        p.prompt = "make Picture 1 a painting"
        with patch.object(pictures, "H3_IMAGE_VAE_SHORT_EDGE", (64, 128)), patch.object(pictures, "H3_IMAGE_VISION_SHORT_EDGE", 32):
            cond = self.run_until_sampling(p)
        # no edit prefix, "Picture 1: " before the vision block, and the model's own picture sizes
        self.assertEqual((p.prompt, p.denoising_strength, p.pi_h3_image_request.references), ("make Picture 1 a painting", 1.0, 1))
        text = self.engine.text_processing_engine_h3
        self.assertEqual(text.picture_label, "Picture %d: ")
        self.assertEqual([tuple(i.shape) for i in text.images], [(1, 32, 64, 3)])
        self.assertEqual([tuple(r.shape) for r in self.engine.references], [(1, 64, 128, 3)])
        dit = self.engine.forge_objects.unet.model.diffusion_model
        sigmas = torch.tensor(dit.pdd_sigmas)
        out = dit(p.modified_noise, sigmas[:1] * 1000, cond[0].unsqueeze(0), {"sampling_sigmas": sigmas})
        self.assertTrue(torch.isfinite(out).all())
        layout = next(iter(dit._layouts.values()))
        self.assertEqual([k for _, _, k in layout.segments], ["text", "ref_img", "video"])
        # standard img2img starts from the encoded picture, with no audio part
        sys.modules["modules.shared"].opts.h3_do_reference = False
        p = Img2Img()
        self.run_until_sampling(p)
        self.assertEqual(tuple(p.init_latent.shape), tuple(p.modified_noise.shape))
        self.assertGreater(float(p.init_latent.abs().sum()), 0.0)

    def test_an_edit_brief_is_encoded_as_it_is_and_large_pictures_shrink_for_the_encoder(self):
        self.use_ph3()
        p = Img2Img()
        p.init_images = [Image.new("RGB", (2560, 1440), "red")]
        p.width, p.height = 2048, 1152
        p.prompt = "Edit <Picture 1>. Replace the sky."
        self.run_until_sampling(p)
        self.assertEqual(p.prompt, "Edit <Picture 1>. Replace the sky.")
        # the VAE copy fits the 2048x1152 area, the text encoder's copy fits 1 MP
        self.assertEqual(tuple(self.engine.references[0].shape), (1, 1152, 2048, 3))
        self.assertEqual(tuple(self.engine.text_processing_engine_h3.images[0].shape), (1, 736, 1344, 3))

    def test_img2img_needs_an_input_image_and_a_plain_resize(self):
        self.use_ph3()
        with self.assertRaisesRegex(H3Error, "Add an input image"):
            integration.before_process(Img2Img(init=False))
        p = Img2Img()
        p.resize_mode = integration.LATENT_UPSCALE
        with self.assertRaisesRegex(H3Error, "latent upscale"):
            integration.before_process(p)
        with self.assertRaisesRegex(H3Error, "up to 8 pictures"):
            integration.before_process(Img2Img(gallery=[Image.new("RGB", (32, 32))] * 8))

    def test_with_reference_off_img2img_is_forges_own(self):
        sys.modules["modules.shared"].opts.h3_do_reference = False
        for repeat in (False, True):
            with self.subTest(repeat=repeat):
                p = Img2Img(gallery=[Image.new("RGB", (40, 80), "green")])  # the gallery is not an edit reference now
                p.prompt = "a bird takes off"
                cond = self.run_until_sampling(p)
                self.assertTrue(p.pi_h3_image_standard_img2img)
                self.assertEqual((p.pi_h3_image_request.references, p.denoising_strength, p.prompt), (0, 0.75, "a bird takes off"))
                self.assertNotIn("H3 Edit pictures", p.extra_generation_params)
                self.assertEqual((self.engine.references, self.engine.generation.refs), ([], []))
                # the input image is encoded into the packed latent, the audio stub at zero, and kept for Forge to noise
                self.assertEqual(tuple(p.init_latent.shape), tuple(p.modified_noise.shape))
                video, audio = self.engine.generation.shapes.unpack(p.init_latent)
                self.assertGreater(float(video.abs().sum()), 0.0)
                self.assertEqual(float(audio.abs().sum()), 0.0)
                dit = self.engine.forge_objects.unet.model.diffusion_model
                start = p.init_latent * 0.25 + p.modified_noise * 0.75
                self.assertTrue(torch.isfinite(dit(start, torch.tensor([750.0]), cond[0].unsqueeze(0))).all())
        # an input image that did not reach the VAE (another VAE for Encoding) is reported before sampling
        p = Img2Img()
        integration.before_process(p)
        p.sd_model = self.engine
        integration.process(p)
        p.init_latent = torch.zeros(1, 4, HEIGHT // 8, WIDTH // 8)
        with self.assertRaisesRegex(H3Error, "did not reach H3"):
            integration.before_sampling(p, torch.zeros(1, 4, HEIGHT // 8, WIDTH // 8))
        # txt2img afterwards starts from noise again
        p = Txt2Img()
        self.run_until_sampling(p)
        self.assertFalse(self.engine.img2img_input)

    def test_txt2img_samples_one_packed_still_image(self):
        p = Txt2Img()
        cond = self.run_until_sampling(p)
        self.assertEqual((p.pi_h3_image_request.width, p.pi_h3_image_request.height, p.pi_h3_image_request.references), (WIDTH, HEIGHT, 0))
        self.assertEqual([m.role for m in p.pi_h3_image_components.models], ["dit", "text_encoder", "vae"])
        shapes = self.engine.generation.shapes
        self.assertEqual((shapes.video, shapes.audio), ((1, 24, 1, HEIGHT // 16, WIDTH // 16), (1, 32, 2, 2)))
        self.assertEqual(tuple(p.modified_noise.shape), (1, 1, 1, shapes.size))
        self.assertEqual(p.rng.shape, (1, 1, shapes.size))
        self.assertEqual((self.engine.generation.refs, self.engine.generation.vision_spans), ([], []))
        # the DiT's working memory is reserved on this generation's copy of the UNet patcher: 24 video rows, 4 audio
        # rows and the prompt's share
        from pi_h3_image.native import engine as engine_module
        tokens = 6 * 4 // 4 + 4 + engine_module.TEXT_TOKENS_RESERVED
        self.assertEqual(self.engine.forge_objects.unet.extra_preserved_memory_during_sampling,
                         tokens * engine_module.SAMPLING_BYTES_PER_TOKEN + engine_module.SAMPLING_FIXED_BYTES)
        # outside the ph3 preset the Shift slider belongs to another model: P-H3-Image keeps its own
        self.assertEqual(p.distilled_cfg_scale, 1.0)
        # one sampler step through the real DiT, then the decode Forge saves
        dit = self.engine.forge_objects.unet.model.diffusion_model
        out = dit(p.modified_noise, torch.tensor([900.0]), cond[0].unsqueeze(0))
        self.assertEqual(out.shape, p.modified_noise.shape)
        self.assertTrue(torch.isfinite(out).all())
        image = self.engine.decode_first_stage(p.modified_noise)
        self.assertEqual(tuple(image.shape), (1, 3, HEIGHT, WIDTH))
        self.assertTrue(float(image.min()) >= -1.0 and float(image.max()) <= 1.0)
        integration.postprocess(p, None)
        self.assertIsNone(self.engine.generation)
        self.assertIsNone(dit.generation)

    def test_p_h3_image_takes_its_own_shift_outside_its_preset(self):
        self.use_ph3()
        p = Txt2Img()
        self.run_until_sampling(p)
        self.assertEqual((p.pi_h3_image_components.profile.name, p.distilled_cfg_scale), ("ph3", 1.0))
        self.assertEqual(p.extra_generation_params["H3 Model"], "P-H3-Image")
        sys.modules["modules.shared"].opts.forge_preset = "ph3"
        p = Txt2Img()
        p.distilled_cfg_scale = 2.0
        self.run_until_sampling(p)
        self.assertEqual(p.distilled_cfg_scale, 2.0)

    def test_the_ph3_preset_keeps_the_slider_and_fills_the_api_default(self):
        sys.modules["modules.shared"].opts.forge_preset = "ph3"
        p = Txt2Img()
        p.distilled_cfg_scale = 6.0
        self.run_until_sampling(p)
        self.assertEqual(p.distilled_cfg_scale, 6.0)
        p = Txt2Img()
        p.is_api = True
        sys.modules["modules.shared"].opts.ph3_t2i_dcfg = 10.0
        self.run_until_sampling(p)
        self.assertEqual(p.distilled_cfg_scale, 10.0)

    def test_the_clip_wrapper_asks_p_h3_image_for_its_five_hidden_states(self):
        from pi_h3_image.native import text_engine
        from pi_h3_image.profiles import PH3_IMAGE_HIDDEN_STATES
        self.assertEqual(text_engine.tap_layers(types.SimpleNamespace(text_connector=object())), PH3_IMAGE_HIDDEN_STATES)
        self.assertEqual(text_engine.tap_layers(types.SimpleNamespace()), "last")

    def test_set_shift_takes_the_size_newer_forge_passes(self):
        # Forge Neo after d70373e calls set_shift(shift=..., width=..., height=...)
        self.engine.set_shift(shift=10.0, width=WIDTH, height=HEIGHT)
        self.assertEqual((self.engine.predictor_shift, self.engine.shift), (10.0, 10.0))
        self.engine.set_shift(0)
        self.assertEqual(self.engine.shift, 12.0)  # the model definition's own shift

    def test_reference_pictures_ride_as_condition_rows_before_the_targets(self):
        # the engine's picture path, which img2img editing will feed: <Picture 1> at its own size
        p = Txt2Img()
        integration.before_process(p)
        p.sd_model = self.engine
        integration.process(p)
        self.engine.set_references([torch.rand(1, 32, 128, 3)])
        cond = self.engine.get_learned_conditioning([p.prompt])
        self.assertEqual(len(self.engine.text_processing_engine_h3.images), 1)
        integration.before_sampling(p, torch.zeros(1, 4, HEIGHT // 8, WIDTH // 8))
        generation = self.engine.generation
        self.assertEqual([(r["kind"], r["latent_h"], r["latent_w"], tuple(r["latent"].shape)) for r in generation.refs],
                         [("image", 2, 8, (1, 24, 1, 2, 8))])
        self.assertEqual(generation.vision_spans, [(0, 3)])
        dit = self.engine.forge_objects.unet.model.diffusion_model
        out = dit(p.modified_noise, torch.tensor([900.0]), cond[0].unsqueeze(0))
        self.assertTrue(torch.isfinite(out).all())
        layout = next(iter(dit._layouts.values()))
        self.assertEqual([k for _, _, k in layout.segments], ["text", "ref_img", "audio", "video"])
        # the next request without pictures drops them and the cached conditioning that carried them
        p = Txt2Img()
        self.run_until_sampling(p)
        p.clear_prompt_cache.assert_called()
        self.assertEqual((self.engine.references, self.engine.generation.refs), ([], []))

    def test_a_rejected_img2img_stops_at_the_model(self):
        with self.assertRaisesRegex(H3Error, "Add an input image"):
            integration.before_process(Img2Img(init=False))
        # the pending error reaches the model and the encoder, which Forge calls directly
        dit = self.engine.forge_objects.unet.model.diffusion_model
        with self.assertRaisesRegex(H3Error, "Add an input image"):
            dit(torch.zeros(1, 1, 1, 8), torch.tensor([900.0]), torch.zeros(1, 7, 48))
        with self.assertRaisesRegex(H3Error, "Add an input image"):
            self.engine.encode_first_stage(torch.zeros(1, 3, HEIGHT, WIDTH))

    def test_output_scale_shrinks_the_decoded_picture_with_an_area_filter(self):
        sys.modules["modules.shared"].opts.h3_output_scale = "1/2"
        p = Txt2Img()
        self.run_until_sampling(p)
        from fractions import Fraction
        self.assertEqual((self.engine.output_scale, p.extra_generation_params["H3 Output scale"]), (Fraction(1, 2), "1/2"))
        image = self.engine.decode_first_stage(p.modified_noise)
        self.assertEqual(tuple(image.shape), (1, 3, HEIGHT // 2, WIDTH // 2))
        # each output pixel is the mean of a 2x2 block of the full decode
        self.engine.set_output_scale(1)
        full = self.engine.decode_first_stage(p.modified_noise)
        torch.testing.assert_close(image, torch.nn.functional.avg_pool2d(full, 2), rtol=0, atol=1e-5)
        # 3/4 keeps three quarters of each side
        self.engine.set_output_scale(Fraction(3, 4))
        quarter = self.engine.decode_first_stage(p.modified_noise)
        self.assertEqual(tuple(quarter.shape), (1, 3, round(HEIGHT * 3 / 4), round(WIDTH * 3 / 4)))
        integration.postprocess(p, None)
        # an unknown value falls back to 1, and 1 leaves no infotext
        sys.modules["modules.shared"].opts.h3_output_scale = "5/6"
        p = Txt2Img()
        self.run_until_sampling(p)
        self.assertEqual(self.engine.output_scale, 1)
        self.assertNotIn("H3 Output scale", p.extra_generation_params)
        integration.postprocess(p, None)

    def test_resolution_shift_moves_the_grid_of_the_schedule_and_the_heads(self):
        model = self.engine.forge_objects.unet.model
        moved = []
        model.predictor = types.SimpleNamespace(set_odds_factor=moved.append, grid=[1.0, 0.5, 0.0])
        sys.modules["modules.shared"].opts.h3_resolution_shift = 0.5
        p = Txt2Img()
        self.run_until_sampling(p)
        factor = ((WIDTH // 32) * (HEIGHT // 32) / 16384) ** 0.5
        self.assertAlmostEqual(moved[-1], factor)
        self.assertEqual(model.diffusion_model.pdd_sigmas, [1.0, 0.5, 0.0])
        self.assertEqual(p.extra_generation_params["H3 Resolution shift"], f"0.5 (x{factor:.2f})")
        integration.postprocess(p, None)
        # 0 puts the stock grid back and leaves no infotext; a model without the grid gets neither
        sys.modules["modules.shared"].opts.h3_resolution_shift = 0
        p = Txt2Img()
        self.run_until_sampling(p)
        self.assertEqual(moved[-1], 1.0)
        self.assertNotIn("H3 Resolution shift", p.extra_generation_params)
        integration.postprocess(p, None)
        del model.predictor
        sys.modules["modules.shared"].opts.h3_resolution_shift = 0.5
        p = Txt2Img()
        self.run_until_sampling(p)
        self.assertNotIn("H3 Resolution shift", p.extra_generation_params)
        integration.postprocess(p, None)

    def test_a_batch_of_pictures_needs_batch_count(self):
        p = Txt2Img()
        p.batch_size = 2
        with self.assertRaisesRegex(H3Error, "Batch Size to 1"):
            integration.before_process(p)
        p = Txt2Img()
        p.n_iter = 4
        integration.before_process(p)
        self.assertIsNotNone(p.pi_h3_image_request)

    def test_a_selected_script_runs_like_a_plain_generation(self):
        # X/Y/Z plot, Prompts from file: Forge's script dropdown sends the script's index first
        p = Txt2Img()
        p.script_args = [3] + p.script_args[1:]
        integration.before_process(p)
        self.assertIsNotNone(p.pi_h3_image_request)

    def test_other_checkpoints_are_left_alone(self):
        sys.modules["modules.sd_models"].get_closet_checkpoint_match = lambda value: types.SimpleNamespace(
            filename=str(checkpoint(self.root / "sd.safetensors", {"unet.weight": ("F16", [4, 4])})))
        p = Txt2Img()
        p.batch_size = 4
        integration.before_process(p)
        self.assertIsNone(p.pi_h3_image_request)
        integration.process(p)
        integration.before_sampling(p, torch.zeros(1, 4, 8, 8))
        self.assertFalse(hasattr(p, "modified_noise"))

    def test_script_rejection_prints_one_line_and_stops_at_the_model(self):
        p = Txt2Img()
        p.width = 100
        with patch("builtins.print") as printed:
            integration.script_before_process(p)
        self.assertRegex(printed.call_args.args[0], r"^\[MiniMax H3\] H3 width and height must be multiples of 32")
        with self.assertRaisesRegex(H3Error, "multiples of 32"):
            raise_pending_error()
        # the model Forge loaded is checked too
        p = Txt2Img()
        integration.before_process(p)
        p.sd_model = types.SimpleNamespace()
        with self.assertRaisesRegex(H3Error, "did not load the H3 model"):
            integration.process(p)

    def test_nan_after_sampling_or_decoding_is_reported_instead_of_a_black_picture(self):
        self.run_until_sampling(Txt2Img())
        shapes = self.engine.generation.shapes
        latent = torch.zeros(1, 1, 1, shapes.size)
        latent[..., 3] = float("nan")
        with self.assertRaisesRegex(H3Error, "sampling produced NaN"):
            self.engine.decode_first_stage(latent)
        with self.assertRaisesRegex(RuntimeError, "does not belong"):
            self.engine.decode_first_stage(torch.zeros(1, 1, 1, 8))
        broken = types.SimpleNamespace(decode=lambda z: torch.full((1, 3, 1, 4, 4), float("inf"), dtype=z.dtype))
        self.engine.forge_objects.vae = types.SimpleNamespace(patcher=None, device="cpu", vae_dtype=torch.float16,
                                                              first_stage_model=broken)
        with self.assertRaisesRegex(H3Error, "VAE produced NaN.*--fp32-vae"):
            self.engine.decode_first_stage(torch.zeros_like(latent))


if __name__ == "__main__":
    unittest.main()
