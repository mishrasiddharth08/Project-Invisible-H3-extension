"""Exercise the real detection hook without importing heavyweight Forge backends."""
import ast
from pathlib import Path
import types
import unittest
from unittest.mock import Mock

class DetectionChainTests(unittest.TestCase):
    def hook(self, prior, legacy):
        path = Path(__file__).resolve().parents[1] / "forge_h3/native/patches.py"
        tree = ast.parse(path.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_hook_detection")
        detection = types.SimpleNamespace(detect_unet_config=prior)
        model = types.SimpleNamespace(detect=legacy)
        namespace = {"detection": detection, "model": model}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
        namespace["_hook_detection"]()
        return detection.detect_unet_config

    def test_pruna_image_detection_wins_over_legacy_audio_layout(self):
        image = {"image_model": "minimax_h3_image"}
        prior, legacy = Mock(return_value=image), Mock(return_value={"image_model": "minimax_h3"})
        self.assertIs(self.hook(prior, legacy)({}, "", extra=True), image)
        prior.assert_called_once_with({}, "", extra=True)
        legacy.assert_not_called()

    def test_original_h3_still_uses_legacy_detector(self):
        result = {"image_model": "minimax_h3"}
        prior, legacy = Mock(return_value=None), Mock(return_value=result)
        self.assertIs(self.hook(prior, legacy)({}, ""), result)
        legacy.assert_called_once_with({}, "")

    def test_unrelated_forge_models_remain_on_their_route(self):
        result = {"image_model": "qwen"}
        legacy = Mock()
        self.assertIs(self.hook(Mock(return_value=result), legacy)({}, ""), result)
        legacy.assert_not_called()

    def test_unknown_model_remains_unknown(self):
        self.assertIsNone(self.hook(Mock(return_value=None), Mock(return_value=None))({}, ""))
