import base64
import io
import json
import logging
from pathlib import Path
import queue
import struct
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pi_h3 import assets
from pi_h3.transport import Worker
from pi_h3.worker import lora_patches, preview


def safetensor(path, tensors, payload=b'1234'):
    raw = json.dumps(tensors).encode()
    path.write_bytes(struct.pack('<Q', len(raw)) + raw + payload)


def native_quant_file(path, quant_format):
    quant = json.dumps({'format': quant_format}).encode()
    tensors = {
        'video_patch_proj.weight': {'dtype': 'U8', 'shape': [1], 'data_offsets': [0, 1]},
        'audio_patch_proj.weight': {'dtype': 'U8', 'shape': [1], 'data_offsets': [1, 2]},
        'blocks.0.comfy_quant': {'dtype': 'U8', 'shape': [len(quant)],
                                 'data_offsets': [2, 2 + len(quant)]},
    }
    safetensor(path, tensors, b'12' + quant)


class HeaderReview(unittest.TestCase):
    def test_rejects_truncated_header(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'bad.safetensors'
            path.write_bytes(struct.pack('<Q', 20) + b'{}')
            with self.assertRaisesRegex(ValueError, 'Incomplete model header'):
                assets.header(path)

    def test_rejects_negative_or_gapped_offsets(self):
        cases = [([-1, 4],), ([1, 4],)]
        for offsets, in cases:
            with self.subTest(offsets=offsets), tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / 'bad.safetensors'
                safetensor(path, {'x': {'dtype': 'F32', 'shape': [1], 'data_offsets': offsets}})
                with self.assertRaisesRegex(ValueError, 'tensor'):
                    assets.header(path)

    def test_native_quant_metadata_names_pass_detection_only(self):
        formats = ('int8_tensorwise', 'float8_e4m3fn', 'mxfp8', 'nvfp4',
                   'convrot_w4a4', 'asym_w4a8_int8')
        for quant_format in formats:
            with self.subTest(quant_format=quant_format), tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / f'minimax_h3_fl2va_{quant_format}.safetensors'
                native_quant_file(path, quant_format)
                inventory = {'dit': [str(path.resolve())]}
                self.assertEqual(assets.resolve('Auto', 'dit', inventory), str(path.resolve()))

    def test_gguf_is_explicitly_unsupported(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'minimax_h3_fl2va_int4.gguf'
            path.write_bytes(b'GGUF')
            with self.assertRaisesRegex(ValueError, 'safetensors'):
                assets.header(path)


class LoraReview(unittest.TestCase):
    def test_partial_adapter_is_rejected(self):
        fake = types.SimpleNamespace(
            model_lora_keys_unet=lambda model, keys: {'good': 'layer.weight'},
            load_lora=lambda weights, keys, log_missing=True: (
                logging.warning('lora key not loaded: bad') or {'layer.weight': object()}))
        model = types.SimpleNamespace(model=object())
        with self.assertRaisesRegex(ValueError, 'unsupported tensor'):
            lora_patches(model, {'good': object(), 'bad': object()}, 'partial.safetensors', fake)

    def test_complete_adapter_is_accepted(self):
        patch = object()
        fake = types.SimpleNamespace(model_lora_keys_unet=lambda model, keys: {'good': 'layer.weight'},
                                     load_lora=lambda weights, keys, log_missing=True: {'layer.weight': patch})
        model = types.SimpleNamespace(model=object())
        self.assertEqual(lora_patches(model, {'good': object()}, 'good.safetensors', fake),
                         {'layer.weight': patch})


class PreviewReview(unittest.TestCase):
    def test_nested_preview_uses_video_stream(self):
        import torch
        from PIL import Image
        latent_format = types.ModuleType('comfy.latent_formats')
        latent_format.MiniMaxH3Video = types.SimpleNamespace(
            latent_rgb_factors=[[0.0, 0.0, 0.0]] * 24,
            latent_rgb_factors_bias=[0.0, 0.0, 0.0])
        comfy = types.ModuleType('comfy')
        comfy.__path__ = []
        comfy.utils = types.ModuleType('comfy.utils')
        nested = types.SimpleNamespace(is_nested=True,
            unbind=lambda: (torch.zeros(1, 24, 1, 2, 2), torch.zeros(1, 32, 2, 2)))
        with patch.dict(sys.modules, {'comfy': comfy, 'comfy.utils': comfy.utils,
                                      'comfy.latent_formats': latent_format}):
            encoded = preview(nested, 64, 32)
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
            self.assertEqual(image.size, (64, 32))


class TransportReview(unittest.TestCase):
    def worker(self, event):
        worker = object.__new__(Worker)
        worker.events = queue.Queue()
        worker.events.put(event)
        worker.errors = []
        worker.close = lambda: None
        return worker

    def test_protocol_error_fails_immediately(self):
        with self.assertRaisesRegex(RuntimeError, 'invalid response'):
            self.worker({'type': 'protocol_error'}).receive(lambda: False, timeout=1)

    def test_unknown_event_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, 'unknown response'):
            self.worker({'type': 'mystery'}).receive(lambda: False, timeout=1)

    def test_worker_exit_reports_code_and_stderr(self):
        worker = self.worker({'type': 'exit'})
        worker.process = types.SimpleNamespace(poll=lambda: 17)
        worker.errors = ['backend import failed']
        with self.assertRaisesRegex(RuntimeError, r'exit code 17[\s\S]*backend import failed'):
            worker.receive(lambda: False, timeout=1)

    def test_cancelled_request_is_never_written(self):
        worker = object.__new__(Worker)
        worker.process = types.SimpleNamespace(stdin=types.SimpleNamespace(write=lambda value: self.fail('write')))
        worker.close = lambda: None
        with self.assertRaises(InterruptedError):
            worker.generate({}, lambda: True, lambda event: None)


if __name__ == '__main__':
    unittest.main()
