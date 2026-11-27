"""CPU pixel parity against the decoder tiling implementation before the fast paths."""

import unittest
from unittest.mock import patch

import torch

from pi_h3_image.native import image_vae


class ToyVAE(image_vae.H3ImageVAE):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.vae_ratio = 2
        self.tile_size = 12
        self.tile_overlap_min = 4
        self.calls = []

    def _decode_pixels(self, z):
        self.calls.append((z.shape[0], z.is_contiguous()))
        # Tile-dependent pixels expose changes to slicing, grouping and overlap blending.
        pixels = z.square() + z.mean(dim=(-2, -1), keepdim=True)
        return pixels.repeat_interleave(2, -2).repeat_interleave(2, -1)


class TiledDecodeRegression(unittest.TestCase):
    def test_single_tile_bypasses_tiling_allocations(self):
        vae = ToyVAE()
        z = torch.randn(1, 3, 1, 6, 12)[..., ::2]
        expected = vae._decode_pixels(z.contiguous())
        vae.calls.clear()
        with patch.object(vae, 'split_tiles', side_effect=AssertionError('unnecessary tiling')):
            actual = vae.tiled_decode(z)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(vae.calls, [(1, True)])

    def test_singleton_group_is_contiguous_without_cat(self):
        vae = ToyVAE()
        z = torch.randn(2, 3, 1, 6, 14)[..., ::2]
        with patch.object(image_vae.kernels, 'free_memory', return_value=0), \
                patch.object(torch, 'cat', side_effect=AssertionError('unnecessary copy')):
            result = list(vae._decode_tile_row(z, [0], [12]))
        self.assertEqual(len(result), 1)
        self.assertEqual(vae.calls, [(2, True)])


if __name__ == '__main__':
    unittest.main()
