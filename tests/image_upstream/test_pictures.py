"""The pictures of a P-H3-Image edit: gallery reading, sizing and the edit prefix."""

import types
import unittest

from PIL import Image

from pi_h3_image import pictures
from pi_h3_image.contracts import H3Error
from pi_h3_image.profiles import H3_IMAGE, PH3_IMAGE

try:
    import torch
except ImportError:
    torch = None


class Script:
    def __init__(self, title, args_from, args_to):
        self._title, self.args_from, self.args_to = title, args_from, args_to

    def title(self):
        return self._title


def processing(init=None, gallery=None):
    p = types.SimpleNamespace(init_images=[init] if init else [])
    p.script_args = [0, bool(gallery), [(image, None) for image in gallery] if gallery else None, 1024]
    p.scripts = types.SimpleNamespace(alwayson_scripts=[Script("MiniMax H3", 1, 1), Script(pictures.IMAGE_STITCH, 1, 4)])
    return p


class PictureTests(unittest.TestCase):
    def test_sizes_scale_down_to_the_area_on_the_grid_never_up(self):
        self.assertEqual(pictures.within_area(4000, 3000, 2048 * 2048), (2368, 1760))  # Pruna's 4:3 4 MP size
        self.assertEqual(pictures.within_area(1920, 1080, 1_000_000), (1344, 736))
        self.assertEqual(pictures.within_area(640, 480, 2048 * 2048), (640, 480))
        self.assertEqual(pictures.within_area(20, 4000, 640 * 384), (32, 4000))  # within the area: the grid only
        self.assertEqual(pictures.within_area(20, 40000, 640 * 384), (32, 22176))  # the short side never goes below 32

    def test_the_input_image_is_picture_one_then_the_gallery(self):
        first, gallery = Image.new("RGB", (64, 64), "red"), [Image.new("RGB", (64, 64), "green")]
        self.assertEqual(pictures.collect(processing(first, gallery), 8), [first, *gallery])
        self.assertEqual(pictures.collect(processing(first), 8), [first])
        self.assertEqual(pictures.collect(processing(None, gallery), 8), gallery)
        with self.assertRaisesRegex(H3Error, "P-H3-Image edits with up to 8 pictures, the input image included; 9 were given"):
            pictures.collect(processing(first, gallery * 8), PH3_IMAGE.max_pictures, PH3_IMAGE.label)
        self.assertEqual(len(pictures.collect(processing(first, gallery * 15), H3_IMAGE.max_pictures, H3_IMAGE.label)), 16)
        with self.assertRaisesRegex(H3Error, "MiniMax-H3-Image edits with up to 16 pictures"):
            pictures.collect(processing(first, gallery * 16), H3_IMAGE.max_pictures, H3_IMAGE.label)

    def test_short_edge_scales_into_its_range_both_ways(self):
        # ComfyUI _size_short_edge: MiniMax-H3-Image's references, 1024 for the text encoder, 1024-2048 for the VAE
        self.assertEqual(pictures.short_edge(512, 512, 1024, 1024), (1024, 1024))      # enlarged
        self.assertEqual(pictures.short_edge(4000, 3000, 1024, 1024), (1376, 1024))    # reduced
        self.assertEqual(pictures.short_edge(1536, 1536, 1024, 2048), (1536, 1536))    # inside the range: untouched
        self.assertEqual(pictures.short_edge(6000, 4000, 1024, 2048), (3072, 2048))
        self.assertEqual(pictures.short_edge(300, 100, 1024, 2048), (3072, 1024))

    def test_gallery_items_from_the_api_are_decoded(self):
        import base64
        import io
        buffer = io.BytesIO()
        Image.new("RGB", (8, 8), "blue").save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode()
        for item in (encoded, "data:image/png;base64," + encoded, (Image.new("RGB", (8, 8)), "caption"), [None]):
            loaded = pictures.load(item)
            self.assertTrue(loaded is None or loaded.size == (8, 8))
        self.assertIsNone(pictures.load(""))

    def test_edit_prefix_rule(self):
        self.assertEqual(pictures.edit_prompt("put a hat on"), "Edit <Picture 1>. put a hat on")
        self.assertEqual(pictures.edit_prompt("  Edit <Picture 2>. swap them "), "Edit <Picture 2>. swap them")
        self.assertEqual(pictures.edit_prompt("subject_definitions: a brief"), "subject_definitions: a brief")

    @unittest.skipIf(torch is None, "needs torch")
    def test_prepared_picture_is_rgb_in_zero_one_within_the_area(self):
        tensor = pictures.prepare(Image.new("RGBA", (1920, 1080), (255, 0, 0, 128)), 640 * 384)
        self.assertEqual(tuple(tensor.shape), (1, 384, 672, 3))
        self.assertEqual(tensor[0, 0, 0].tolist(), [1.0, 0.0, 0.0])
        small = pictures.prepare(Image.new("RGB", (64, 32)), 640 * 384)
        self.assertEqual(tuple(small.shape), (1, 32, 64, 3))

    @unittest.skipIf(torch is None, "needs torch")
    def test_each_model_sizes_its_edit_pictures_its_own_way(self):
        image = Image.new("RGB", (3000, 1500), "red")
        vae, vision = pictures.edit_inputs(PH3_IMAGE, [image], 2048, 1024)
        self.assertEqual([tuple(t.shape) for t in (*vae, *vision)], [(1, 1024, 2048, 3), (1, 704, 1408, 3)])
        vae, vision = pictures.edit_inputs(H3_IMAGE, [image], 2048, 1024)
        self.assertEqual([tuple(t.shape) for t in (*vae, *vision)], [(1, 1504, 3008, 3), (1, 1024, 2048, 3)])
        # a small picture is enlarged to the 1024 short edge for MiniMax-H3-Image, never for P-H3-Image
        small = Image.new("RGB", (256, 256))
        self.assertEqual(tuple(pictures.edit_inputs(H3_IMAGE, [small], 512, 512)[0][0].shape), (1, 1024, 1024, 3))
        self.assertEqual(tuple(pictures.edit_inputs(PH3_IMAGE, [small], 512, 512)[0][0].shape), (1, 256, 256, 3))


if __name__ == "__main__":
    unittest.main()
