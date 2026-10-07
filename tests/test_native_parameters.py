from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from forge_h3.contracts import GenerationRequest, H3Error, align_frames


class NativeParameterTests(unittest.TestCase):
    def test_valid_numeric_boundaries(self):
        for frames in (5, 22, 362, "124"):
            with self.subTest(frames=frames):
                self.assertEqual(GenerationRequest(frames=frames).frames, int(frames))
        for shift in (0.01, 3, 100, "6"):
            with self.subTest(shift=shift):
                self.assertEqual(GenerationRequest(audio_shift=shift).audio_shift, float(shift))
        for references in (0, 9, "0", "9"):
            with self.subTest(references=references):
                self.assertEqual(
                    GenerationRequest(mode="ref2va", references=references).references,
                    int(references),
                )
        self.assertEqual((GenerationRequest(width=64, height=4096).width,
                          GenerationRequest(width=64, height=4096).height), (64, 4096))

    def test_invalid_frame_values(self):
        for value in (4, 6, 363, -12, 5.5, True, False, None, math.nan, math.inf, -math.inf):
            with self.subTest(value=value), self.assertRaises(H3Error):
                GenerationRequest(frames=value)

    def test_invalid_dimensions(self):
        for field in ("width", "height"):
            for value in (0, 32, 65, -64, 64.5, True, False, None, math.nan, math.inf):
                with self.subTest(field=field, value=value), self.assertRaises(H3Error):
                    GenerationRequest(**{field: value})

    def test_invalid_audio_shift(self):
        for value in (0, -1, 100.1, True, False, None, "loud", math.nan, math.inf, -math.inf):
            with self.subTest(value=value), self.assertRaises(H3Error):
                GenerationRequest(audio_shift=value)

    def test_boolean_fields_are_strict(self):
        for field in ("include_audio", "first_frame", "last_frame"):
            for value in (0, 1, "true", "false", None):
                with self.subTest(field=field, value=value), self.assertRaisesRegex(H3Error, "true or false"):
                    GenerationRequest(**{field: value})

    def test_output_and_mode_are_exact(self):
        for output in ("video", "Still", "", None, True):
            with self.subTest(output=output), self.assertRaisesRegex(H3Error, "output"):
                GenerationRequest(output=output)
        for mode in ("FL2VA", "video", "", None, True):
            with self.subTest(mode=mode), self.assertRaisesRegex(H3Error, "mode"):
                GenerationRequest(mode=mode)

    def test_reference_count_is_integer_and_bounded(self):
        for value in (-1, 10, 1.5, True, False, None, math.nan, math.inf):
            with self.subTest(value=value), self.assertRaises(H3Error):
                GenerationRequest(mode="ref2va", references=value)

    def test_modes_reject_incompatible_pictures(self):
        with self.assertRaisesRegex(H3Error, "Ref2VA checkpoint"):
            GenerationRequest(mode="fl2va", references=1)
        for field in ("first_frame", "last_frame"):
            with self.subTest(field=field), self.assertRaisesRegex(H3Error, "first or last frame"):
                GenerationRequest(mode="ref2va", **{field: True})
            with self.subTest(field=field, output="still"), self.assertRaisesRegex(H3Error, "Still image"):
                GenerationRequest(output="Still image", **{field: True})

    def test_still_output_forces_five_silent_frames(self):
        request = GenerationRequest(output="Still image", frames="ignored", include_audio=True)
        self.assertEqual(request.frames, 5)
        self.assertFalse(request.include_audio)

    def test_alignment_rejects_fractional_boolean_and_nonfinite_values(self):
        self.assertEqual([align_frames(value) for value in (1, 5, 6, 123, "124")], [5, 5, 22, 124, 124])
        for value in (5.5, True, False, None, math.nan, math.inf, -math.inf):
            with self.subTest(value=value), self.assertRaises(H3Error):
                align_frames(value)


if __name__ == "__main__":
    unittest.main()
