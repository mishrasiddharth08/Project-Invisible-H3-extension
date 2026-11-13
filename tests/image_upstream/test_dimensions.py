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

    def test_exact_cap_and_one_grid_step_over(self):
        self.assertEqual(MAX_PIXELS, 4096 * 4096)
        self.assertEqual(fit_dimensions(4096, 4128, changed=1), (4064, 4128))
        self.assertEqual(fit_dimensions(4128, 4096, changed=0), (4128, 4064))
        self.assertEqual(fit_dimensions(8192, 8192, changed=0), (8192, 2048))
        self.assertEqual(fit_dimensions(8192, 8192), (4096, 4096))

    def test_grid_area_and_out_of_range_entry(self):
        for pair in ((4000, 4000), (4096, 4096), (5472, 3040), (3040, 5472), (8192, 2048), (2048, 8192)):
            self.assertEqual(fit_dimensions(*pair), pair)
            validate_dimensions("h3image", *pair)
        for pair in ((8192, 8192), (5472, 3072), (4096, 4096), (64, 8192)):
            for axis in (None, 0, 1):
                w, h = fit_dimensions(*pair, axis)
                self.assertLessEqual(w * h, MAX_PIXELS)
                self.assertEqual((w % 32, h % 32), (0, 0))
        with self.assertRaises(H3Error):
            validate_dimensions("h3image", 4096, 4128)
        for other in ("ph3", "h3", "other"):
            validate_dimensions(other, 8192, 8192)

if __name__ == "__main__":
    unittest.main()
