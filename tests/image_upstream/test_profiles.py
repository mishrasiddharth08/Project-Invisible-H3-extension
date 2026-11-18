"""The model profiles, Pruna's 8-step schedule and the MiniMax-H3-Image schedules."""

import unittest

from pi_h3_image.profiles import (
    ALL,
    H3_IMAGE,
    H3_IMAGE_6_STEP,
    H3_IMAGE_8_STEP,
    H3_IMAGE_8_STEP_SOFTER,
    H3_IMAGE_PDD_SIGMAS,
    H3_IMAGE_REFERENCE,
    H3_IMAGE_TAIL,
    PH3_IMAGE,
    PH3_IMAGE_SIGMAS,
    SCHEDULERS,
    for_components,
    h3_image_6_step_sigmas,
    h3_image_8_step_sigmas,
    h3_image_8_step_softer_sigmas,
    ph3_image_sigmas,
)

try:
    import torch
except ImportError:
    torch = None


class ProfileTests(unittest.TestCase):
    def test_the_text_encoder_picks_p_h3_image_on_an_audio_video_checkpoint(self):
        self.assertIs(for_components("h3", "ph3-image"), PH3_IMAGE)
        self.assertEqual((PH3_IMAGE.name, PH3_IMAGE.steps, PH3_IMAGE.cfg, PH3_IMAGE.shift), ("ph3", 8, 1.0, 1.0))
        self.assertEqual((PH3_IMAGE.picture_label, PH3_IMAGE.edit_prefix, PH3_IMAGE.max_pictures), ("<Picture %d>: ", True, 8))
        # MiniMax H3 with its own text encoder is the video model: no still-image profile
        self.assertIsNone(for_components("h3", "h3"))
        self.assertEqual([profile.name for profile in ALL], ["ph3", "h3image"])

    def test_the_image_checkpoint_picks_its_own_profile(self):
        self.assertIs(for_components("h3-image", "h3"), H3_IMAGE)
        self.assertIs(for_components("h3-image", "ph3-image"), H3_IMAGE)
        self.assertEqual((H3_IMAGE.steps, H3_IMAGE.cfg, H3_IMAGE.picture_label, H3_IMAGE.edit_prefix, H3_IMAGE.shift_used),
                         (6, 1.0, "Picture %d: ", False, False))
        # the default is MiniMax's own grid (Simple, 6 steps) through the LCM sampler; MiniMax's Euler recipe is kept
        # as the documented reference
        self.assertEqual((H3_IMAGE.sampler, H3_IMAGE.scheduler, H3_IMAGE_REFERENCE), ("H3-Image LCM", "Simple", ("Euler", "Simple", 6)))

    def test_every_default_schedule_is_registered(self):
        labels = [label for _, label, _ in SCHEDULERS]
        self.assertEqual(labels, ["P-H3-Image", "H3-Image 8-step", "H3-Image 8-step softer", "H3-Image 6-step"])
        self.assertEqual(len({name for name, _, _ in SCHEDULERS}), len(SCHEDULERS))
        # Pruna's grid is an added schedule; MiniMax's is Forge's Simple
        self.assertIn(PH3_IMAGE.scheduler, labels)
        self.assertEqual(H3_IMAGE.scheduler, "Simple")


@unittest.skipIf(torch is None, "needs torch")
class ScheduleTests(unittest.TestCase):
    def test_eight_steps_give_prunas_grid_as_it_is(self):
        sigmas = ph3_image_sigmas(8, sigma_min=0.001, sigma_max=1.0)
        self.assertEqual(sigmas.dtype, torch.float32)
        torch.testing.assert_close(sigmas, torch.tensor(PH3_IMAGE_SIGMAS, dtype=torch.float32))

    def test_other_step_counts_resample_the_same_curve(self):
        for n in (4, 12, 20):
            with self.subTest(n=n):
                sigmas = ph3_image_sigmas(n, sigma_min=0.001, sigma_max=1.0)
                self.assertEqual(sigmas.shape, (n + 1,))
                self.assertEqual((float(sigmas[0]), float(sigmas[-1])), (1.0, 0.0))
                self.assertTrue(bool((sigmas[1:] < sigmas[:-1]).all()))
        # the grid's own points stay on the curve: 16 steps pass through every 8-step sigma
        sixteen = ph3_image_sigmas(16)
        torch.testing.assert_close(sixteen[::2], torch.tensor(PH3_IMAGE_SIGMAS, dtype=torch.float32))


@unittest.skipIf(torch is None, "needs torch")
class H3ImageScheduleTests(unittest.TestCase):
    FUNCTIONS = ((H3_IMAGE_8_STEP, h3_image_8_step_sigmas, H3_IMAGE_TAIL),
                 (H3_IMAGE_8_STEP_SOFTER, h3_image_8_step_softer_sigmas, H3_IMAGE_TAIL),
                 (H3_IMAGE_6_STEP, h3_image_6_step_sigmas, H3_IMAGE_TAIL))

    def test_the_named_step_count_gives_the_tested_list(self):
        # the lists of the owner's same-seed tests, as typed into the sigma field
        eight = h3_image_8_step_sigmas(8, sigma_min=0.0014, sigma_max=1.0)
        self.assertEqual(eight.dtype, torch.float32)
        torch.testing.assert_close(eight, torch.tensor([1.0, 0.9726, 0.9383, 0.8942, 0.8353, 0.7526, 0.6283, 0.42, 0.0]),
                                   rtol=0, atol=1e-4)
        torch.testing.assert_close(h3_image_6_step_sigmas(6), torch.tensor(H3_IMAGE_PDD_SIGMAS[:5] + (0.42, 0.0)))
        # the 6-step schedule is the checkpoint's grid with its last point lifted out of the failing range
        self.assertGreater(H3_IMAGE_TAIL, H3_IMAGE_PDD_SIGMAS[5])

    def test_the_softer_start_is_halfway_between_the_curve_and_prunas_grid(self):
        softer = h3_image_8_step_softer_sigmas(8)
        torch.testing.assert_close(softer, torch.tensor([1.0, 0.9787, 0.9531, 0.923, 0.8803, 0.8172, 0.7192, 0.42, 0.0]))
        halfway = (h3_image_8_step_sigmas(8)[:7].double() + torch.tensor(PH3_IMAGE_SIGMAS[:7], dtype=torch.float64)) / 2
        torch.testing.assert_close(softer[:7].double(), halfway, rtol=0, atol=1e-4)
        # slower than the 8-step curve at every step before the tail, faster than Pruna's grid
        self.assertTrue(bool((softer[1:7] > h3_image_8_step_sigmas(8)[1:7]).all()))
        self.assertTrue(bool((softer[1:7] < torch.tensor(PH3_IMAGE_SIGMAS[1:7])).all()))

    def test_twelve_steps_of_the_curve_are_the_tested_list_too(self):
        tested = [1.0, 0.9832, 0.964, 0.9418, 0.9157, 0.8847, 0.8474, 0.8014, 0.7434, 0.6679, 0.5659, 0.42, 0.0]
        torch.testing.assert_close(h3_image_8_step_sigmas(12), torch.tensor(tested), rtol=0, atol=1e-4)

    def test_any_step_count_keeps_the_ends_and_never_goes_below_the_tail(self):
        for label, function, tail in self.FUNCTIONS:
            for n in (1, 2, 3, 4, 5, 6, 8, 10, 12, 20, 30):
                with self.subTest(schedule=label, n=n):
                    sigmas = function(n, sigma_min=0.0014, sigma_max=1.0, device="cpu")
                    self.assertEqual(sigmas.shape, (n + 1,))
                    self.assertEqual((float(sigmas[0]), float(sigmas[-1])), (1.0, 0.0))
                    self.assertTrue(bool((sigmas[1:] < sigmas[:-1]).all()))
                    if n > 1:
                        # the last model evaluation is at the tail, whatever the count
                        self.assertAlmostEqual(float(sigmas[-2]), tail, places=6)

    def test_a_grid_schedule_resampled_passes_through_its_own_points(self):
        # 11 steps of the 6-step schedule: every second sigma is one of the six
        torch.testing.assert_close(h3_image_6_step_sigmas(11)[:-1:2], h3_image_6_step_sigmas(6)[:-1])


if __name__ == "__main__":
    unittest.main()
