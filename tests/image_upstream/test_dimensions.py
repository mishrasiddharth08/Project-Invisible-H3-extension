import unittest
from pi_h3_image.dimensions import dimension_updates, validate_dimensions, MAX_PIXELS
from pi_h3_image.contracts import H3Error

class DimensionsTests(unittest.TestCase):
    def setUp(self):
        self.defaults = [dict(minimum=64, maximum=2048, step=8)] * 2

    def test_enter_and_restore(self):
        state, updates = dimension_updates(True, {}, (1024, 1536), self.defaults)
        self.assertEqual([u["maximum"] for u in updates], [4096, 4096])
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
        validate_dimensions("h3image", 4096, 4096)
        for pair in ((4096, 4128), (4128, 4096), (8192, 8192)):
            with self.assertRaises(H3Error):
                validate_dimensions("h3image", *pair)

    def test_grid_area_and_out_of_range_entry(self):
        for pair in ((4000, 4000), (4096, 4096), (3072, 4096), (4096, 3072)):
            validate_dimensions("h3image", *pair)
        validate_dimensions("h3image", 4128, 1024)
        _, updates = dimension_updates(True, {}, (4096, 4128), self.defaults)
        self.assertEqual([update["value"] for update in updates], [4096, 4096])
        for other in ("ph3", "h3", "other"):
            validate_dimensions(other, 8192, 8192)

if __name__ == "__main__":
    unittest.main()
