"""The H3-Image LCM sampler: LCM with a falling noise scale."""

import unittest

try:
    import torch
except ImportError:
    torch = None

from pi_h3_image.sampler import S_NOISE_END, S_NOISE_START, SAMPLER, noise_scale


class NoiseScaleTests(unittest.TestCase):
    def test_the_scale_falls_linearly_from_the_first_step_to_the_last(self):
        self.assertEqual(SAMPLER, "H3-Image LCM")
        self.assertEqual((S_NOISE_START, S_NOISE_END), (1.10, 0.90))
        # Simple at 6 steps: the community workflow's numbers, the last one unused (the step to 0 adds no noise)
        self.assertEqual([round(noise_scale(i, 6), 4) for i in range(6)], [1.10, 1.06, 1.02, 0.98, 0.94, 0.90])
        self.assertEqual(noise_scale(0, 1), S_NOISE_START)


class FlowPredictor:
    """Forge's flow noise scaling: sigma * noise + (1 - sigma) * latent."""

    @staticmethod
    def noise_scaling(sigma, noise, latent):
        return sigma * noise + (1.0 - sigma) * latent


class FakeModel:
    """A denoiser that always predicts the same clean latent and records what it was given."""

    def __init__(self, clean):
        self.clean = clean
        self.calls = []
        self.inner_model = type("Inner", (), {"predictor": FlowPredictor()})()

    def __call__(self, x, sigma, **kwargs):
        self.calls.append((x.clone(), float(sigma[0])))
        return self.clean.clone()


@unittest.skipIf(torch is None, "needs torch")
class SamplerTests(unittest.TestCase):
    def test_each_step_noises_the_clean_estimate_afresh_with_the_scaled_noise(self):
        from pi_h3_image.sampler import sample_h3_image_lcm

        clean = torch.full((1, 2, 3, 3), 0.5)
        noise = torch.ones_like(clean)
        sigmas = torch.tensor([1.0, 0.75, 0.25, 0.0])
        model = FakeModel(clean)
        drawn = []
        out = sample_h3_image_lcm(model, torch.randn_like(clean), sigmas,
                                  noise_sampler=lambda s, s_next: drawn.append((float(s), float(s_next))) or noise)
        # one model call per sigma before 0, fresh noise before every call but the first
        self.assertEqual([round(s, 4) for _, s in model.calls], [1.0, 0.75, 0.25])
        self.assertEqual(drawn, [(1.0, 0.75), (0.75, 0.25)])
        # three steps: the scale runs 1.10, 1.00, 0.90; call 2 sees the clean estimate noised to 0.75 with noise x 1.10,
        # call 3 to 0.25 with x 1.00
        torch.testing.assert_close(model.calls[1][0], 0.75 * 1.10 * noise + 0.25 * clean)
        torch.testing.assert_close(model.calls[2][0], 0.25 * 1.00 * noise + 0.75 * clean)
        # the last step to 0 returns the clean estimate as it is
        torch.testing.assert_close(out, clean)

    def test_the_callback_sees_every_step(self):
        from pi_h3_image.sampler import sample_h3_image_lcm

        clean = torch.zeros(1, 1, 2, 2)
        seen = []
        sample_h3_image_lcm(FakeModel(clean), clean, torch.tensor([1.0, 0.5, 0.0]),
                            callback=lambda d: seen.append(d["i"]), noise_sampler=lambda *_: torch.zeros_like(clean))
        self.assertEqual(seen, [0, 1])


if __name__ == "__main__":
    unittest.main()
