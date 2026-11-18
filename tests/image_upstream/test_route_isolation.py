import unittest
from unittest.mock import patch

from pi_h3_image.models import H3, H3_IMAGE, PH3_IMAGE, inspect_model, is_image_checkpoint
from pi_h3_image.native.detect import AUDIO_KEY, IDENTITY_KEYS


@patch("pi_h3_image.models._quantization", return_value="plain")
class ImageRouteIsolationTests(unittest.TestCase):
    def test_official_image_is_recognized_by_tensor_keys(self, _quantization):
        with patch("pi_h3_image.models.read_header", return_value=dict.fromkeys(IDENTITY_KEYS)):
            model = inspect_model("renamed-image.safetensors")
        self.assertEqual(model.variant, H3_IMAGE)
        self.assertTrue(is_image_checkpoint(model))

    def test_pruna_and_legacy_video_keep_distinct_routes(self, _quantization):
        header = dict.fromkeys((*IDENTITY_KEYS, AUDIO_KEY))
        with patch("pi_h3_image.models.read_header", return_value=header):
            pruna = inspect_model("p-h3-image_w4a8.safetensors")
            video = inspect_model("MiniMax-H3-FL2VA-Pruned-int8_convrot.safetensors")
        self.assertEqual(pruna.variant, PH3_IMAGE)
        self.assertTrue(is_image_checkpoint(pruna))
        self.assertEqual(video.variant, H3)
        self.assertFalse(is_image_checkpoint(video))
