import json
import struct
import tempfile
import unittest
from pathlib import Path

from pi_h3_image.contracts import GenerationRequest, H3Error
from pi_h3_image.models import inspect_model, read_header, resolve_components


def checkpoint(path, tensors, metadata=None):
    header = {name: {"dtype": dtype, "shape": shape, "data_offsets": [0, 0]}
              for name, (dtype, shape) in tensors.items()}
    if metadata:
        header["__metadata__"] = metadata
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw)
    return path


DIT = {"video_patch_proj.weight": ("BF16", [5376, 96]),
       "audio_patch_proj.weight": ("BF16", [5376, 64]),
       "condition_proj.weight": ("BF16", [5376, 5120]),
       "final_layer.video_out.weight": ("BF16", [96, 5376]),
       "final_layer.audio_out.weight": ("BF16", [64, 5376])}
# MiniMax-H3-Image: no audio stream, six stacked output heads, INT8 attention declared for its blocks
H3_IMAGE_DIT = {"video_patch_proj.weight": ("F32", [5376, 96]),
                "condition_proj.weight": ("BF16", [5376, 5120]),
                "final_layer.video_out.weight": ("F32", [576, 5376]),
                "blocks.0.attn.comfy_attention.config": ("U8", [37])}
TE = {"model.language_model.embed_tokens.weight": ("BF16", [151936, 5120]),
      "model.language_model.layers.49.self_attn.q_proj.weight": ("BF16", [5120, 5120]),
      "model.visual.patch_embed.proj.weight": ("BF16", [1280, 3, 2, 14, 14])}
# the video VAE of MiniMax H3, which the extension refuses
VAE = {"decoder.x_embedder.proj.weight": ("F16", [1536, 24, 1, 2, 2]),
       "decoder.register_tokens": ("F16", [1, 4, 1536]),
       "encoder.conv_in.weight": ("F16", [128, 3, 3, 3, 3])}
AUDIO_VAE = {"pre_block.attn.q_bias": ("F32", [768]),
             "pre_block.attn.v_bias": ("F32", [768]),
             "pre_block.attn.zero_k_bias": ("F32", [768])}
# P-H3-Image: the 4B encoder with Pruna's text connector, and the VAE whose decoder makes one frame per latent
PH3_TE = {"model.language_model.embed_tokens.weight": ("BF16", [151936, 2560]),
          "model.language_model.layers.35.self_attn.q_proj.weight": ("BF16", [4096, 2560]),
          "model.visual.patch_embed.proj.weight": ("BF16", [1024, 3, 2, 16, 16]),
          "text_connector.proj.weight": ("F32", [5120, 12800])}
PH3_VAE = dict(VAE, **{"decoder.proj_out.weight": ("BF16", [768, 2048]), "decoder.x_embedder.weight": ("BF16", [2048, 24])})
# MiniMax-H3-Image's VAE: the 2D decoder, 16 heads of 64
H3_IMAGE_VAE = dict(VAE, **{"decoder.proj_out.weight": ("F16", [768, 1024]), "decoder.x_embedder.weight": ("F16", [1024, 24])})


class HeaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_recognizes_renamed_community_model_from_tensors(self):
        item = inspect_model(checkpoint(self.root / "any-name.safetensors", DIT))
        self.assertEqual(item.role, "dit")
        self.assertEqual(item.quantization, "plain")

    def test_filename_cannot_turn_sd_into_h3(self):
        path = checkpoint(self.root / "minimax_h3.safetensors", {"unet.weight": ("F16", [4, 4])})
        self.assertIsNone(inspect_model(path))

    def test_header_size_is_bounded_before_reading(self):
        path = self.root / "bad.safetensors"
        path.write_bytes(struct.pack("<Q", 100_000_000))
        with self.assertRaisesRegex(H3Error, "header"):
            read_header(path)

    def test_duplicate_tensor_keys_are_rejected(self):
        path = self.root / "bad.safetensors"
        raw = b'{"w":{},"w":{}}'
        path.write_bytes(struct.pack("<Q", len(raw)) + raw)
        with self.assertRaisesRegex(H3Error, "Duplicate"):
            read_header(path)

    def test_component_resolution_requires_each_role(self):
        dit = checkpoint(self.root / "model.safetensors", DIT)
        te = checkpoint(self.root / "encoder.safetensors", TE)
        with self.assertRaisesRegex(H3Error, "H3 VAE"):
            resolve_components(dit, [te])
        vae = checkpoint(self.root / "vae.safetensors", PH3_VAE)
        with self.assertRaisesRegex(H3Error, "text encoder"):
            resolve_components(dit, [vae])

    def test_duplicate_role_is_not_silently_selected(self):
        dit = checkpoint(self.root / "model.safetensors", DIT)
        a = checkpoint(self.root / "a.safetensors", TE)
        b = checkpoint(self.root / "b.safetensors", TE)
        with self.assertRaisesRegex(H3Error, "More than one"):
            resolve_components(dit, [a, b])

    def test_the_audio_vae_is_recognized_and_refused(self):
        audio = checkpoint(self.root / "audio.safetensors", AUDIO_VAE)
        self.assertEqual(inspect_model(audio).role, "audio_vae")
        dit = checkpoint(self.root / "model.safetensors", DIT)
        modules = [checkpoint(self.root / f"{name}.safetensors", tensors) for name, tensors in (("te", PH3_TE), ("vae", PH3_VAE))]
        with self.assertRaisesRegex(H3Error, "audio VAE: still images do not use it"):
            resolve_components(dit, modules + [audio])

    def test_nvfp4_text_encoder_is_rejected(self):
        dit = checkpoint(self.root / "model.safetensors", DIT)
        te = checkpoint(self.root / "encoder.safetensors", TE, {"quantization": "nvfp4_awq"})
        vae = checkpoint(self.root / "vae.safetensors", PH3_VAE)
        with self.assertRaisesRegex(H3Error, "NVFP4"):
            resolve_components(dit, [te, vae])

    def test_int8_vae_is_accepted(self):
        dit = checkpoint(self.root / "model.safetensors", DIT)
        te = checkpoint(self.root / "encoder.safetensors", PH3_TE)
        vae = quantized_checkpoint(self.root / "vae.safetensors", ["int8_tensorwise"], base=PH3_VAE,
                                   layer="decoder.transformer_blocks.0.attn.to_qkv")
        components = resolve_components(dit, [te, vae])
        self.assertEqual((components.vae.role, components.vae.quantization), ("vae", "int8"))

    def test_complete_selection_resolves(self):
        paths = [checkpoint(self.root / f"{n}.safetensors", t) for n, t in (("te", PH3_TE), ("v", PH3_VAE))]
        components = resolve_components(checkpoint(self.root / "model.safetensors", DIT), paths)
        self.assertEqual([m.role for m in components.models], ["dit", "text_encoder", "vae"])
        self.assertEqual([m.variant for m in components.models], ["h3", "ph3-image", "ph3-image"])
        self.assertEqual(components.profile.name, "ph3")

    def test_the_video_model_set_is_refused(self):
        # MiniMax H3 as MiniMax ships it (32B text encoder, video VAE) is a video model: no still-image profile
        dit = checkpoint(self.root / "model.safetensors", DIT)
        te, video = checkpoint(self.root / "te.safetensors", TE), checkpoint(self.root / "video.safetensors", VAE)
        self.assertEqual((inspect_model(video).role, inspect_model(video).variant), ("vae", "h3"))
        with self.assertRaisesRegex(H3Error, "video.safetensors is the H3 video VAE.*p-h3-image_vae.*minimax_h3_image_vae"):
            resolve_components(dit, [te, video])
        with self.assertRaisesRegex(H3Error, "audio-video checkpoint makes still images with the P-H3-Image text encoder"):
            resolve_components(dit, [te, checkpoint(self.root / "v.safetensors", PH3_VAE)])

    def test_the_text_encoder_selects_the_p_h3_image_profile_and_its_vae(self):
        dit = checkpoint(self.root / "model.safetensors", DIT)
        te, vae = checkpoint(self.root / "te.safetensors", PH3_TE), checkpoint(self.root / "v.safetensors", PH3_VAE)
        self.assertEqual((inspect_model(te).role, inspect_model(te).variant), ("text_encoder", "ph3-image"))
        self.assertEqual((inspect_model(vae).role, inspect_model(vae).variant), ("vae", "ph3-image"))
        components = resolve_components(dit, [te, vae])
        self.assertEqual((components.profile.name, components.profile.shift, components.profile.steps), ("ph3", 1.0, 8))
        # the two P-H3-Image parts go together: the DiT file cannot tell, the mismatch is refused
        with self.assertRaisesRegex(H3Error, "is the H3 video VAE"):
            resolve_components(dit, [te, checkpoint(self.root / "video.safetensors", VAE)])
        with self.assertRaisesRegex(H3Error, "P-H3-Image text encoder .p-h3-image_text_encoder."):
            resolve_components(dit, [checkpoint(self.root / "encoder.safetensors", TE), vae])

    def test_minimax_h3_image_is_told_by_its_checkpoint(self):
        dit = checkpoint(self.root / "any-name.safetensors", H3_IMAGE_DIT)
        te, vae = checkpoint(self.root / "te.safetensors", TE), checkpoint(self.root / "v.safetensors", H3_IMAGE_VAE)
        self.assertEqual((inspect_model(dit).role, inspect_model(dit).variant), ("dit", "h3-image"))
        self.assertEqual((inspect_model(vae).role, inspect_model(vae).variant), ("vae", "h3-image"))
        components = resolve_components(dit, [te, vae])
        profile = components.profile
        self.assertEqual((profile.name, profile.steps, profile.cfg, profile.scheduler, profile.max_pictures),
                         ("h3image", 6, 1.0, "Simple", 16))
        self.assertTrue(components.wants_int8_attention)
        ph3 = [checkpoint(self.root / "ph3_te.safetensors", PH3_TE), checkpoint(self.root / "ph3_vae.safetensors", PH3_VAE)]
        self.assertFalse(resolve_components(checkpoint(self.root / "model.safetensors", DIT), ph3).wants_int8_attention)
        self.assertEqual(components.experimental, "")
        # experimental: P-H3-Image's 4B text encoder in front of this checkpoint keeps the checkpoint's own recipe
        mixed = resolve_components(dit, [checkpoint(self.root / "ph3_te.safetensors", PH3_TE), vae])
        self.assertEqual((mixed.profile.name, mixed.text_encoder.variant, mixed.experimental),
                         ("h3image", "ph3-image", "P-H3-Image text encoder"))
        # experimental too: P-H3-Image's VAE (the same H3 encoder and latents, another single-frame decoder), alone or
        # with that text encoder; the recipe stays the checkpoint's
        ph3_vae = checkpoint(self.root / "other.safetensors", PH3_VAE)
        with_vae = resolve_components(dit, [te, ph3_vae])
        self.assertEqual((with_vae.profile.name, with_vae.vae.variant, with_vae.experimental),
                         ("h3image", "ph3-image", "P-H3-Image VAE"))
        both = resolve_components(dit, [ph3[0], ph3_vae])
        self.assertEqual((both.profile.name, both.experimental), ("h3image", "P-H3-Image text encoder + P-H3-Image VAE"))
        # MiniMax-H3-Image's own VAE still goes with no other checkpoint
        with self.assertRaisesRegex(H3Error, "is the H3 video VAE"):
            resolve_components(dit, [te, checkpoint(self.root / "video.safetensors", VAE)])
        with self.assertRaisesRegex(H3Error, "goes with the MiniMax-H3-Image checkpoint"):
            resolve_components(checkpoint(self.root / "model.safetensors", DIT), [ph3[0], vae])


def quantized_checkpoint(path, formats, metadata=None, base=DIT, layer="blocks.{i}.mlp.fc1"):
    """A checkpoint plus one comfy_quant JSON tensor per format, with its bytes, as ComfyUI's quantized files store
    them."""
    header, blobs, offset = {}, [], 0
    for name, (dtype, shape) in base.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset]}
    for i, fmt in enumerate(formats):
        blob = json.dumps({"format": fmt, "convrot_groupsize": 256}).encode()
        header[f"{layer.format(i=i)}.comfy_quant"] = {"dtype": "U8", "shape": [len(blob)],
                                                      "data_offsets": [offset, offset + len(blob)]}
        blobs.append(blob)
        offset += len(blob)
    if metadata:
        header["__metadata__"] = metadata
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"".join(blobs))
    return path


class QuantizationLabelTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_w4a8_metadata_is_not_labelled_w6a8(self):
        layers = {"blocks.0.attn.qkv_proj": {"format": "asym_w4a8_int8", "group_size": 16}}
        path = checkpoint(self.root / "kijai.safetensors", dict(DIT, **{"blocks.0.attn.qkv_proj.weight_s_rel": ("F32", [1])}),
                          {"_quantization_metadata": json.dumps({"layers": layers})})
        self.assertEqual(inspect_model(path).quantization, "w4a8")

    def test_comfy_quant_tensors_give_the_format(self):
        self.assertEqual(inspect_model(quantized_checkpoint(self.root / "int4.safetensors", ["convrot_w4a4"])).quantization,
                         "int4")
        mixed = quantized_checkpoint(self.root / "mixed.safetensors", ["convrot_w4a4", "int8_tensorwise", "convrot_w4a4"])
        self.assertEqual(inspect_model(mixed).quantization, "int4 + int8")


GGML = {"F32": 0, "F16": 1, "Q8_0": 8, "Q2_K": 10, "Q4_K": 12, "Q5_K": 13, "IQ1_S": 19, "BF16": 30}


def gguf_string(text):
    raw = text.encode()
    return struct.pack("<Q", len(raw)) + raw


def gguf_checkpoint(path, tensors, metadata=None):
    """A GGUF v3 header without tensor data: string metadata, then name, ggml dims (innermost first), type, offset."""
    metadata = metadata or {}
    out = [b"GGUF", struct.pack("<IQQ", 3, len(tensors), len(metadata))]
    for key, value in metadata.items():
        out += [gguf_string(key), struct.pack("<I", 8), gguf_string(value)]
    for name, (dtype, shape) in tensors.items():
        out += [gguf_string(name), struct.pack("<I", len(shape)), struct.pack(f"<{len(shape)}Q", *reversed(shape)),
                struct.pack("<IQ", GGML[dtype], 0)]
    path.write_bytes(b"".join(out))
    return path


GGUF_DIT = dict(DIT, **{"blocks.0.attn.qkv_proj.weight": ("Q4_K", [21504, 5376]),
                        "blocks.0.mlp.fc1.weight": ("Q4_K", [28672, 5376]),
                        "blocks.0.norm1.weight": ("BF16", [5376]),
                        "video_patch_proj.weight": ("F32", [5376, 96])})


GGUF_H3_IMAGE_DIT = {"video_patch_proj.weight": ("F32", [5376, 96]),
                     "condition_proj.weight": ("BF16", [5376, 5120]),
                     "final_layer.video_out.weight": ("F32", [576, 5376]),
                     "blocks.0.mlp.fc1.weight": ("Q4_K", [28672, 5376])}


# unsloth's qwen3vl_32b_minimax_h3-Q2_K_M.gguf in miniature: Hugging Face names, the vision tower inside, no metadata
GGUF_TE = {"model.embed_tokens.weight": ("BF16", [151936, 5120]),
           "model.layers.49.self_attn.q_proj.weight": ("Q2_K", [8192, 5120]),
           "model.layers.49.mlp.gate_proj.weight": ("Q2_K", [25600, 5120]),
           "model.layers.49.mlp.down_proj.weight": ("Q5_K", [5120, 25600]),
           "visual.deepstack_merger_list.0.norm.weight": ("BF16", [4608]),
           "visual.patch_embed.proj.weight": ("BF16", [3456, 2, 16, 16])}


class GGUFTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_gguf_dit_is_recognized_from_its_tensor_table(self):
        item = inspect_model(gguf_checkpoint(self.root / "any-name.gguf", GGUF_DIT, {"general.name": "x"}))
        self.assertEqual((item.role, item.quantization), ("dit", "gguf Q4_K"))

    def test_gguf_shapes_are_read_in_torch_order(self):
        header = read_header(gguf_checkpoint(self.root / "model.gguf", GGUF_DIT))
        self.assertEqual(header["blocks.0.attn.qkv_proj.weight"], {"dtype": "Q4_K", "shape": [21504, 5376]})
        self.assertEqual(header["__metadata__"], {})

    def test_gguf_dit_resolves_with_safetensors_modules(self):
        dit = gguf_checkpoint(self.root / "model.gguf", GGUF_DIT)
        paths = [checkpoint(self.root / f"{n}.safetensors", t) for n, t in (("te", PH3_TE), ("v", PH3_VAE))]
        self.assertEqual(resolve_components(dit, paths).dit.quantization, "gguf Q4_K")

    def test_gguf_types_forge_cannot_dequantize_are_refused(self):
        path = gguf_checkpoint(self.root / "tiny.gguf", dict(DIT, **{"blocks.0.mlp.fc1.weight": ("IQ1_S", [28672, 5376])}))
        with self.assertRaisesRegex(H3Error, "IQ1_S"):
            inspect_model(path)

    def test_llama_cpp_text_encoder_is_refused_with_a_clear_message(self):
        path = gguf_checkpoint(self.root / "qwen.gguf", {"blk.0.attn_q.weight": ("Q4_K", [5120, 5120]),
                                                         "token_embd.weight": ("Q8_0", [151936, 5120])})
        with self.assertRaisesRegex(H3Error, "separate file.*unsloth"):
            inspect_model(path)

    def test_unsloth_text_encoder_is_recognized(self):
        item = inspect_model(gguf_checkpoint(self.root / "qwen3vl_32b_minimax_h3-Q2_K_M.gguf", GGUF_TE))
        self.assertEqual((item.role, item.quantization), ("text_encoder", "gguf Q2_K"))

    def test_all_gguf_set_resolves(self):
        # the 32B text encoder goes with MiniMax-H3-Image
        dit = gguf_checkpoint(self.root / "model.gguf", GGUF_H3_IMAGE_DIT)
        te = gguf_checkpoint(self.root / "te.gguf", GGUF_TE)
        paths = [te, checkpoint(self.root / "v.safetensors", H3_IMAGE_VAE)]
        components = resolve_components(dit, paths)
        self.assertEqual((components.dit.quantization, components.text_encoder.quantization), ("gguf Q4_K", "gguf Q2_K"))
        self.assertEqual(components.profile.name, "h3image")

    def test_gguf_vae_is_refused(self):
        with self.assertRaisesRegex(H3Error, "diffusion model and text encoder only"):
            inspect_model(gguf_checkpoint(self.root / "vae.gguf", VAE))

    def test_gguf_header_is_bounded_and_checked(self):
        path = self.root / "bad.gguf"
        path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 10**9, 0))
        with self.assertRaisesRegex(H3Error, "GGUF"):
            read_header(path)
        path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 1, 0) + gguf_string("w"))
        with self.assertRaisesRegex(H3Error, "truncated"):
            read_header(path)
        path.write_bytes(b"NOPE")
        with self.assertRaisesRegex(H3Error, "magic"):
            read_header(path)


class RequestTests(unittest.TestCase):
    def test_sides_are_multiples_of_32(self):
        request = GenerationRequest(width="1024", height=768)
        self.assertEqual((request.width, request.height), (1024, 768))
        for width, height in ((833, 480), (32, 64), (96.5, 64)):
            with self.assertRaisesRegex(H3Error, "32|integer"):
                GenerationRequest(width=width, height=height)

    def test_pictures_are_counted_and_bounded(self):
        self.assertEqual(GenerationRequest(references=16).references, 16)
        with self.assertRaisesRegex(H3Error, "up to 16 pictures; 17 were given"):
            GenerationRequest(references=17)
        with self.assertRaisesRegex(H3Error, "References must be an integer"):
            GenerationRequest(references=1.5)


if __name__ == "__main__":
    unittest.main()
