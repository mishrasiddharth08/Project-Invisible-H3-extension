"""The text encoder module without Forge Neo: Pruna's text connector and the image path's hidden-state taps."""

import importlib
import sys
import types
import unittest
from dataclasses import dataclass
from unittest.mock import patch

import forge_stubs

try:
    import torch
except ImportError:
    torch = None


@dataclass
class _Config:
    hidden_size: int = 8


class _Layer(torch.nn.Module if torch is not None else object):
    """A decoder layer that adds its index, so every tap is told apart."""

    def __init__(self, index):
        super().__init__()
        self.index = index

    def forward(self, x, **kwargs):
        return x + self.index, None


@unittest.skipIf(torch is None, "needs torch")
class TextEncoderTests(unittest.TestCase):
    def setUp(self):
        llama = forge_stubs.module("backend.nn.llm.llama", Llama2_=object, Qwen3VL=torch.nn.Module,
                                   Qwen3VL_4BConfig=_Config, attention_function=None)
        qwen35 = forge_stubs.module("backend.nn.llm.qwen35", QWEN3VL_VISION={}, Qwen3VLVisionModel=object)
        llm = forge_stubs.module("backend.nn.llm", llama=llama, qwen35=qwen35)
        modules = {"backend": forge_stubs.module("backend"), "backend.nn": forge_stubs.module("backend.nn", llm=llm),
                   "backend.nn.llm": llm, "backend.nn.llm.llama": llama, "backend.nn.llm.qwen35": qwen35}
        self.enterContext(patch.dict(sys.modules, modules))
        sys.modules.pop("pi_h3_image.native.text_encoder", None)
        self.addCleanup(sys.modules.pop, "pi_h3_image.native.text_encoder", None)
        # import_module: the package may keep an earlier test's module as an attribute
        self.module = importlib.import_module("pi_h3_image.native.text_encoder")

    def test_the_connector_standardizes_projects_and_puts_the_sink_first(self):
        torch.manual_seed(0)
        connector = self.module.TextConnector(2 * 4, 6, dtype=torch.float32)
        with torch.no_grad():
            for p in connector.parameters():
                p.normal_()
            connector.input_std.abs_().add_(0.5)
        hidden = torch.randn(1, 2, 5, 4)  # [B, layers, T, C]
        out = connector(hidden)
        self.assertEqual(tuple(out.shape), (1, 5, 6))
        self.assertTrue(torch.equal(out[0, 0], connector.sink_embedding))
        x = hidden.movedim(1, 2).flatten(2)
        expected = (x - connector.input_mean) / connector.input_std @ connector.proj.weight.T
        expected = connector.output_mean + expected * connector.output_std
        torch.testing.assert_close(out[:, 1:], expected[:, 1:])
        # the connector runs in fp32 whatever the hidden states' dtype
        self.assertEqual(connector(hidden.to(torch.bfloat16)).dtype, torch.float32)
        self.assertEqual(connector.input_mean.dtype, torch.float32)

    def test_the_image_path_taps_the_same_states_as_forges_llama(self):
        # Llama2_.forward: a tap i is the state entering layer i, the layer count is the normed output
        model = types.SimpleNamespace(config=types.SimpleNamespace(head_dim=8, rope_theta=10000.0, rope_dims=[2, 1, 1]),
                                      layers=[_Layer(1), _Layer(10), _Layer(100)], norm=lambda x: x * 2)
        x = torch.zeros(1, 3, 4)
        position_ids = torch.zeros(3, 3)  # the three M-RoPE axes of three tokens
        visual_mask = torch.tensor([False, True, False])
        out, taps = self.module.forward_with_images(model, x, None, [1, 2, 3], False, position_ids, visual_mask, deepstack=[])
        self.assertEqual(float(out[0, 0, 0]), 2 * 111)
        self.assertEqual([float(t) for t in taps[0, :, 0, 0]], [1.0, 11.0, 222.0])
        out, single = self.module.forward_with_images(model, x, None, 1, True, position_ids, visual_mask, deepstack=[])
        self.assertEqual(float(single[0, 0, 0]), 2 * 11)  # after layer 1, normed on request
        # DeepStack features reach the image positions after the first layers
        deep = [torch.full((1, 4), 1000.0)]
        out, _ = self.module.forward_with_images(model, x, None, None, True, position_ids, visual_mask, deepstack=deep)
        self.assertEqual([float(v) for v in out[0, :, 0]], [222.0, 2222.0, 222.0])

    def test_vision_settings_read_both_forge_layouts(self):
        qwen35 = sys.modules["backend.nn.llm.qwen35"]
        qwen35.QWEN3VL_VISION = dict(num_heads=16, patch_size=16, hidden_size=1024, depth=24)
        self.assertEqual(self.module.vision_defaults(), qwen35.QWEN3VL_VISION)
        qwen35.QWEN3VL_VISION_COMMON = dict(num_heads=16, patch_size=16)
        qwen35.QWEN3VL_VISION = {"qwen3vl_4b": dict(hidden_size=1024, depth=24), "qwen3vl_8b": dict(hidden_size=1152, depth=27)}
        self.assertEqual(self.module.vision_defaults(), dict(num_heads=16, patch_size=16, hidden_size=1152, depth=27))


if __name__ == "__main__":
    unittest.main()
