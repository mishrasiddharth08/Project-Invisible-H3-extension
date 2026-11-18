"""The sigma grid predictor of MiniMax-H3-Image, with a stand-in for Forge Neo's flow predictor."""

import importlib
import sys
import unittest
from unittest.mock import patch

import forge_stubs

try:
    import torch
except ImportError:
    torch = None

GRID = (1.0, 0.9675236940383911, 0.8875485062599182, 0.75, 0.5327739119529724, 0.23200887441635132, 0.0)


class _Flow(torch.nn.Module if torch is not None else object):
    """Forge Neo's PredictionDiscreteFlow, as far as the subclass builds on it."""

    def __init__(self, model_config):
        super().__init__()
        self.shift, self.multiplier = 1.0, 1000
        self.set_parameters(shift=model_config.sampling_settings.get("shift", 1.0))

    def set_parameters(self, *, shift=None, multiplier=None, timesteps=1000):
        self.shift = shift or self.shift
        self.multiplier = multiplier or self.multiplier
        self.register_buffer("sigmas", self.sigma((torch.arange(1, timesteps + 1, 1) / timesteps) * self.multiplier))


def simple(n, sigmas):
    # Forge Neo's simple_scheduler
    step = len(sigmas) / n
    return [float(sigmas[-(1 + int(x * step))]) for x in range(n)] + [0.0]


@unittest.skipIf(torch is None, "needs torch")
class GridPredictorTests(unittest.TestCase):
    def setUp(self):
        prediction = forge_stubs.module("backend.modules.k_prediction", PredictionDiscreteFlow=_Flow)
        modules = {"backend": forge_stubs.module("backend"), "backend.modules": forge_stubs.module("backend.modules"),
                   "backend.modules.k_prediction": prediction}
        self.enterContext(patch.dict(sys.modules, modules))
        sys.modules.pop("pi_h3_image.native.sampling", None)
        self.addCleanup(sys.modules.pop, "pi_h3_image.native.sampling", None)
        module = importlib.import_module("pi_h3_image.native.sampling")
        config = forge_stubs.module("config", sampling_settings={"shift": 12.0})
        self.predictor = module.PredictionH3Image(config, GRID)

    def test_simple_at_six_steps_is_the_checkpoints_grid_exactly(self):
        torch.testing.assert_close(torch.tensor(simple(6, self.predictor.sigmas)), torch.tensor(GRID), rtol=0, atol=1e-7)
        torch.testing.assert_close(torch.tensor(simple(3, self.predictor.sigmas)), torch.tensor(GRID[::2]), rtol=0, atol=1e-7)

    def test_sigma_is_linear_in_the_grid_index_and_ignores_the_shift(self):
        sigmas = self.predictor.sigmas
        self.assertEqual((float(sigmas[-1]), sigmas.shape[0]), (1.0, 1000))
        self.assertTrue(bool((sigmas[1:] >= sigmas[:-1]).all()))
        # halfway between two grid points
        self.assertAlmostEqual(float(self.predictor.sigma(torch.tensor(750.0))), (GRID[1] + GRID[2]) / 2, places=6)
        before = sigmas.clone()
        self.predictor.set_parameters(shift=3.0)
        self.assertTrue(torch.equal(self.predictor.sigmas, before))
        self.assertEqual((self.predictor.percent_to_sigma(0.0), self.predictor.percent_to_sigma(1.0)), (1.0, 0.0))
        self.assertAlmostEqual(self.predictor.percent_to_sigma(0.5), 0.75, places=6)

    def test_the_resolution_shift_moves_the_grid_and_simple_follows_it(self):
        self.predictor.set_odds_factor(0.5)
        moved = [s if s in (0.0, 1.0) else 0.5 * s / (1.0 - 0.5 * s) for s in GRID]
        self.assertEqual(self.predictor.odds_factor, 0.5)
        torch.testing.assert_close(torch.tensor(self.predictor.grid), torch.tensor(moved), rtol=0, atol=1e-7)
        torch.testing.assert_close(torch.tensor(simple(6, self.predictor.sigmas)), torch.tensor(moved), rtol=0, atol=1e-6)
        # factor 1 is the checkpoint's grid again
        self.predictor.set_odds_factor(1.0)
        torch.testing.assert_close(torch.tensor(simple(6, self.predictor.sigmas)), torch.tensor(GRID), rtol=0, atol=1e-7)
        self.assertEqual(self.predictor.grid, list(GRID))

    def test_a_point_just_inside_an_end_stays_inside(self):
        # ER SDE replaces a first sigma of 1 with percent_to_sigma(1e-4): it must come back below 1, or its log-SNR is
        # infinite and the sampler makes NaN
        near_top = self.predictor.percent_to_sigma(1e-4)
        self.assertLess(near_top, 1.0)
        self.assertGreater(near_top, GRID[1])
        self.assertTrue(bool(torch.isfinite(torch.tensor(near_top).logit())))
        self.assertGreater(self.predictor.percent_to_sigma(1.0 - 1e-4), 0.0)

    def test_the_smallest_sigma_is_above_zero(self):
        # Normal steps down to sigma_min before its closing 0, and a zero there divides by zero in the sampler
        sigmas = self.predictor.sigmas
        self.assertAlmostEqual(float(sigmas[0]), GRID[-2] * 6 / 1000, places=7)
        self.assertGreater(float(sigmas[1]), float(sigmas[0]))


if __name__ == "__main__":
    unittest.main()
