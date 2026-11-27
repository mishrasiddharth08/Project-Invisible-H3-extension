"""Public Ostris tensor-header regression; optional real-weight CPU decode check.

Set H3_VAE_CPU_SMOKE_DIR to a folder containing stock and Ostris ComfyUI VAEs
to run the heavier test. No downloads and no CUDA calls are made by this module.
"""

import gc
import json
import os
import tempfile
import unittest
from pathlib import Path

from test_contracts import H3_IMAGE_DIT, TE, checkpoint

try:
    import comfy_kitchen  # noqa: F401
    import torch
except ImportError:
    torch = None

FIXTURE = Path(__file__).with_name("fixtures") / "ostris_h3_image_vae_header.json"
FILES = ("minimax_h3_image_vae_fp16.safetensors", "ostris_h3_image_vae_fp16.safetensors")


class OstrisHeaderTests(unittest.TestCase):
    def test_community_vae_resolves_by_tensor_layout_without_filename_dependency(self):
        from pi_h3_image.models import H3_IMAGE, inspect_model, resolve_components
        header = json.loads(FIXTURE.read_text())["header"]
        tensors = {k: (v["dtype"], v["shape"]) for k, v in header.items() if k != "__metadata__"}
        self.assertEqual(len(tensors), 416)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            vae = checkpoint(root / "renamed-community-vae.safetensors", tensors)
            self.assertEqual((inspect_model(vae).role, inspect_model(vae).variant), ("vae", H3_IMAGE))
            dit = checkpoint(root / "dit.safetensors", H3_IMAGE_DIT)
            te = checkpoint(root / "te.safetensors", TE)
            self.assertEqual(resolve_components(dit, [te, vae]).profile.name, "h3image")

    @unittest.skipIf(torch is None, "needs torch and comfy-kitchen")
    def test_all_published_tensor_shapes_match_the_existing_native_vae(self):
        import torch
        from pi_h3_image.native import vae
        from pi_h3_image.native.image_vae import MiniMaxH3ImageVAE
        header = json.loads(FIXTURE.read_text())["header"]
        self.assertTrue(vae.is_h3_image_vae(header))
        self.assertEqual(vae.decoder_layers(header), 24)
        self.assertNotIn("latents_mean", header)
        with torch.device("meta"):
            model = MiniMaxH3ImageVAE(num_layers=24)
        expected = {k: tuple(v.shape) for k, v in model.state_dict().items()}
        actual = {k: tuple(v["shape"]) for k, v in header.items() if k != "__metadata__"}
        actual.update(latents_mean=(24,), latents_std=(24,))
        self.assertEqual(actual, expected)

    @unittest.skipUnless(torch is not None and os.environ.get("H3_VAE_CPU_SMOKE_DIR"), "optional real-weight CPU smoke")
    def test_real_stock_and_ostris_weights_strict_load_and_decode_shared_latent(self):
        import torch
        from safetensors.torch import load_file
        from pi_h3_image.native import vae
        from pi_h3_image.native.image_vae import IMAGENET_MEAN, IMAGENET_STD, MiniMaxH3ImageVAE, RotaryEmbeddingND
        root = Path(os.environ["H3_VAE_CPU_SMOKE_DIR"])
        latent = torch.randn(1, 24, 1, 4, 4, generator=torch.Generator().manual_seed(42))
        outputs = []
        for name in FILES:
            with self.subTest(name=name):
                state = vae.with_latent_stats(load_file(str(root / name), device="cpu"))
                self.assertTrue(vae.is_h3_image_vae(state))
                with torch.device("meta"):
                    model = MiniMaxH3ImageVAE(num_layers=vae.decoder_layers(state), tiling=True)
                # Strict verifies every loaded key and shape, including the untouched encoder.
                model.load_state_dict(state, strict=True, assign=True)
                del state
                # Decode on CPU in fp32. The unused encoder remains file-backed fp16.
                model.decoder.float()
                model.post_quant_conv.float()
                # These computed, non-persistent buffers are absent from the file;
                # constructing on meta requires recreating them on CPU.
                model.decoder.pos_embed = RotaryEmbeddingND(32, 100.0, n_dim=2)
                for block in model.decoder.transformer_blocks:
                    block.attn.qk_norm_scale = torch.ones(block.attn.dim_head)
                model.pixel_mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1, 1)
                model.pixel_std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1, 1)
                with torch.inference_mode():
                    picture = model.decode(latent)
                self.assertEqual(tuple(picture.shape), (1, 3, 1, 64, 64))
                self.assertTrue(bool(torch.isfinite(picture).all()))
                self.assertGreaterEqual(float(picture.min()), 0)
                self.assertLessEqual(float(picture.max()), 1)
                outputs.append(picture)
                print(f"CPU VAE {name}: strict load OK; finite 64x64 output")
                del model
                gc.collect()
        if len(outputs) == 2:
            self.assertFalse(torch.equal(*outputs), "fine-tuned decoder should change pixels")


if __name__ == "__main__":
    unittest.main()
