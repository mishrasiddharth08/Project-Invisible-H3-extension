"""The [H3] Resolution shift: the odds factor for a picture size, and the schedules that follow it."""

import math
import types
import unittest

try:
    import torch
except ImportError:
    torch = None

from pi_h3_image import profiles
from pi_h3_image.resolution_shift import REFERENCE_TOKENS, odds_factor, shift_sigma, tokens

GRID = (1.0, 0.9675236940383911, 0.8875485062599182, 0.75, 0.5327739119529724, 0.23200887441635132, 0.0)


class OddsFactorTests(unittest.TestCase):
    def test_sixteen_megapixels_is_the_reference(self):
        self.assertEqual((tokens(4096, 4096), REFERENCE_TOKENS), (16384, 16384))
        self.assertEqual(odds_factor(4096, 4096, 0.5), 1.0)
        # strength 0 is MiniMax's fixed grid at every size
        self.assertEqual(odds_factor(2048, 2048, 0.0), 1.0)

    def test_the_factor_follows_the_token_count_to_the_strength(self):
        # SD3's rule at strength 0.5: a quarter of the pixels halves the shift
        self.assertAlmostEqual(odds_factor(2048, 2048, 0.5), 0.5)
        self.assertAlmostEqual(odds_factor(2048, 2048, 0.25), 0.5 ** 0.5)
        self.assertGreater(odds_factor(5472, 3072, 0.5), 1.0)
        # a side that is not a whole number of tokens counts its whole tokens
        self.assertEqual(tokens(2050, 2047), 64 * 63)

    def test_a_shift_multiplies_the_odds_and_keeps_the_ends(self):
        for sigma in GRID[1:-1]:
            moved = shift_sigma(sigma, 0.5)
            self.assertAlmostEqual(moved / (1 - moved), 0.5 * sigma / (1 - sigma))
            self.assertLess(moved, sigma)
        self.assertEqual((shift_sigma(1.0, 0.5), shift_sigma(0.0, 0.5)), (1.0, 0.0))
        self.assertEqual(shift_sigma(0.75, 1.0), 0.75)

    def test_the_grid_is_a_shift_three_curve(self):
        # the property the setting rests on: logit(sigma) = ln 3 + 1.42 logit(t) on the grid, within 0.02
        for k, sigma in enumerate(GRID[1:-1], start=1):
            t = 1 - k / 6
            fitted = math.log(3) + 1.4215 * math.log(t / (1 - t))
            self.assertAlmostEqual(math.log(sigma / (1 - sigma)), fitted, delta=0.02)


@unittest.skipIf(torch is None, "needs torch")
class FollowGridTests(unittest.TestCase):
    def test_a_schedule_moves_with_the_predictors_factor(self):
        schedule = dict((name, function) for name, _, function in profiles.SCHEDULERS)["h3_image_8"]
        stock = profiles.h3_image_8_step_sigmas(8)
        # no model, or a predictor without a grid (P-H3-Image's flow predictor): the list as it is
        torch.testing.assert_close(schedule(8), stock)
        flow = types.SimpleNamespace(predictor=types.SimpleNamespace(shift=12.0))
        torch.testing.assert_close(schedule(8, inner_model=flow), stock)
        linker = types.SimpleNamespace(predictor=types.SimpleNamespace(odds_factor=0.5))
        moved = schedule(8, sigma_min=0.0, sigma_max=1.0, inner_model=linker)
        self.assertEqual((float(moved[0]), float(moved[-1])), (1.0, 0.0))
        torch.testing.assert_close(moved[1:-1] / (1 - moved[1:-1]), 0.5 * stock[1:-1] / (1 - stock[1:-1]))

    def test_every_schedule_asks_forge_for_the_model(self):
        for name, _, function in profiles.SCHEDULERS:
            self.assertTrue(hasattr(function, "__wrapped__"), name)
