"""Legacy video encoder tolerates old and current Forge loader interfaces."""
import importlib
import sys
import unittest
from dataclasses import dataclass
from unittest.mock import patch
import torch
import forge_stubs

@dataclass
class Config:
    hidden_size: int = 8

class TextEncoderCompatibilityTests(unittest.TestCase):
    def setUp(self):
        llama = forge_stubs.module("backend.nn.llm.llama", Llama2_=lambda config: config,
            Qwen3VL=torch.nn.Module, Qwen3VL_4BConfig=Config)
        self.vision = forge_stubs.module("backend.nn.llm.qwen35",
            QWEN3VL_VISION={"patch_size": 16}, Qwen3VLVisionModel=lambda config: config)
        llm = forge_stubs.module("backend.nn.llm", llama=llama, qwen35=self.vision)
        self.enterContext(patch.dict(sys.modules, {"backend": forge_stubs.module("backend"),
            "backend.nn": forge_stubs.module("backend.nn", llm=llm), "backend.nn.llm": llm,
            "backend.nn.llm.llama": llama, "backend.nn.llm.qwen35": self.vision}))
        sys.modules.pop("forge_h3.native.text_encoder", None)
        self.addCleanup(sys.modules.pop, "forge_h3.native.text_encoder", None)
        self.encoder = importlib.import_module("forge_h3.native.text_encoder")

    def test_old_loader_flat_vision(self):
        model = self.encoder.Qwen3VL32B({"text_config": {"hidden_size": 5120}})
        self.assertEqual(model.model.hidden_size, 5120)
        self.assertEqual(model.num_layers, 50)
        self.assertEqual(model.visual["patch_size"], 16)
        self.assertEqual(model.visual["out_hidden_size"], 5120)

    def test_current_loader_keeps_32b_config_and_merges_vision(self):
        self.vision.QWEN3VL_VISION_COMMON = {"patch_size": 16, "num_heads": 16}
        self.vision.QWEN3VL_VISION = {"qwen3vl_8b": {"hidden_size": 1152, "depth": 27}}
        model = self.encoder.Qwen3VL32B({"text_config": {"hidden_size": 5120},
            "vision_config": {"depth": 32}}, model_type="qwen3vl_4b")
        self.assertEqual(model.model.hidden_size, 5120)
        self.assertEqual(model.visual["patch_size"], 16)
        self.assertEqual(model.visual["hidden_size"], 1152)
        self.assertEqual(model.visual["depth"], 32)
        self.assertEqual(model.visual["out_hidden_size"], 5120)
        with self.assertRaises(ValueError):
            self.encoder.Qwen3VL32B({"text_config": {"hidden_size": 2560}}, model_type="qwen3vl_4b")

if __name__ == "__main__": unittest.main()
