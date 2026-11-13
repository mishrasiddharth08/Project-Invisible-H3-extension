import unittest
from pi_h3_image.dimensions import dimension_updates, fit_dimensions, validate_dimensions, MAX_SIDE, MAX_PIXELS
from pi_h3_image.contracts import H3Error

class DimensionsTests(unittest.TestCase):
    def setUp(self):
        self.defaults = [dict(minimum=64, maximum=2048, step=8)] * 2

    def test_enter_and_restore(self):
        state, updates = dimension_updates(True, {}, (1024, 1536), self.defaults)
        self.assertEqual([u["maximum"] for u in updates], [8192, 8192])
        self.assertEqual([u["value"] for u in updates], [1024, 1536])
        state2, repeated = dimension_updates(True, state, (4000, 4000), self.defaults)
        self.assertEqual(repeated, [{}, {}])
        _, restored = dimension_updates(False, state2, (4000, 4000), self.defaults)
        self.assertEqual(restored, [dict(self.defaults[0], value=1024), dict(self.defaults[1], value=1536)])

    def test_other_models_unchanged(self):
        _, updates = dimension_updates(False, {}, (2048, 1024), self.defaults)
        self.assertEqual(updates, [{}, {}])

    def test_grid_area_and_out_of_range_entry(self):
        for pair in ((4000, 4000), (5312, 3008), (3008, 5312)):
            self.assertEqual(fit_dimensions(*pair), pair)
            validate_dimensions("h3image", *pair)
        for pair in ((8192, 8192), (5472, 3072), (4096, 4096), (64, 8192)):
            for axis in (None, 0, 1):
                w, h = fit_dimensions(*pair, axis)
                self.assertLessEqual(w * h, MAX_PIXELS)
                self.assertEqual((w % 32, h % 32), (0, 0))
        with self.assertRaises(H3Error):
            validate_dimensions("h3image", 4096, 4096)
        for other in ("ph3", "h3", "other"):
            validate_dimensions(other, 8192, 8192)

if __name__ == "__main__":
    unittest.main()
