from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pi_h3 import request, runtime


class Processing:
    class StableDiffusionProcessingImg2Img:
        pass

    @staticmethod
    def fix_seed(p):
        pass

    @staticmethod
    def Processed(p, images, **kwargs):
        return types.SimpleNamespace(images=images, **kwargs)


class StillParameterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.requests, self.workers, self.closed = [], [], []
        owner = self

        class Worker:
            def __init__(self, mode):
                self.mode = mode
                self.process = types.SimpleNamespace(poll=lambda: None)
                owner.workers.append(self)

            def wait_ready(self, cancelled):
                if cancelled():
                    raise InterruptedError()

            def generate(self, data, cancelled, update):
                if cancelled():
                    raise InterruptedError()
                owner.requests.append({**data, 'reference_pixels': [
                    Image.open(path).getpixel((0, 0)) for path in data['references']]})
                Image.new('RGB', (data['width'], data['height']), 'green').save(data['output'])
                return data['output']

            def close(self):
                owner.closed.append(self)

        self.state = types.SimpleNamespace(interrupted=False, skipped=False,
                                           assign_current_image=Mock())
        opts = types.SimpleNamespace(live_previews_enable=False, samples_save=False,
                                     samples_format='png')
        shared = types.SimpleNamespace(
            state=self.state, opts=opts, total_tqdm=Mock(),
            prompt_styles=types.SimpleNamespace(
                apply_styles_to_prompt=lambda text, styles: text,
                apply_negative_styles_to_prompt=lambda text, styles: text))
        modules = types.ModuleType('modules')
        modules.shared = shared
        modules.processing = Processing
        modules.images = types.SimpleNamespace(save_image=Mock())
        backend = types.ModuleType('backend')
        backend.memory_management = types.SimpleNamespace(unload_all_models=Mock())
        inventory = {'dit': ['dit'], 'clip': ['clip'], 'vae': ['vae'], 'lora': []}
        self.patches = [
            patch.dict(sys.modules, {'modules': modules, 'backend': backend}),
            patch.object(runtime, 'Worker', Worker),
            patch.object(runtime, 'scan', return_value=inventory),
            patch.object(runtime, 'resolve', side_effect=lambda value, role, found: role),
            patch.object(runtime, 'header'),
        ]
        self.inventory = inventory
        for item in self.patches:
            item.start()
        runtime._worker = None
        runtime._selection_cancel.clear()

    def tearDown(self):
        runtime.release()
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    def p(self, editing=False, **changes):
        value = Processing.StableDiffusionProcessingImg2Img() if editing else types.SimpleNamespace()
        fields = dict(prompt='prompt', negative_prompt='negative', styles=[], width=512,
                      height=512, steps=20, cfg_scale=1.0, sampler_name='ER SDE',
                      scheduler='Simple', seed=7, batch_size=1, n_iter=1,
                      init_images=[], denoising_strength=1.0, extra_generation_params={},
                      outpath_samples=str(self.folder))
        fields.update(changes)
        for key, item in fields.items():
            setattr(value, key, item)
        return value

    def test_every_sampler_scheduler_and_automatic(self):
        for label, internal in request.SAMPLERS.items():
            self.assertEqual(request.validate(512, 512, 20, 1, label, 'Automatic'),
                             (internal, 'simple'))
        for label, internal in request.SCHEDULERS.items():
            self.assertEqual(request.validate(512, 512, 20, 1, 'ER SDE', label),
                             ('er_sde', internal))

    def test_numeric_boundaries(self):
        for width, height, steps, cfg in ((64, 64, 1, 1), (4096, 4096, 150, 20)):
            self.assertEqual(request.validate(width, height, steps, cfg, 'ER SDE', 'Simple'),
                             ('er_sde', 'simple'))
        for changes in ({'steps': 0}, {'steps': 151}, {'cfg': 0.9}, {'cfg': 21},
                        {'width': 65}, {'height': 4128}):
            values = dict(width=512, height=512, steps=20, cfg=1,
                          sampler='ER SDE', scheduler='Simple')
            values.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                request.validate(**values)

    def test_edit_denoise_must_be_finite_and_exact(self):
        for denoise in (float('nan'), float('inf'), 'invalid', 0.999):
            with self.subTest(denoise=denoise), self.assertRaisesRegex(ValueError, 'Denoising strength to 1'):
                request.validate(512, 512, 20, 1, 'ER SDE', 'Simple',
                                 editing=True, image=object(), denoise=denoise)

    def test_core_controls_reach_worker_exactly(self):
        runtime.generate(self.p(width=640, height=384, steps=37, cfg_scale=2.5,
                                sampler_name='Heun', scheduler='Karras'), {})
        sent = self.requests[0]
        self.assertEqual((sent['width'], sent['height'], sent['steps'], sent['cfg']),
                         (640, 384, 37, 2.5))
        self.assertEqual((sent['sampler'], sent['scheduler']), ('heun', 'karras'))

    def test_prompt_lists_batch_seeds_and_limit(self):
        result = runtime.generate(self.p(prompt=['a', 'b'], negative_prompt=['x'],
                                         batch_size=2, n_iter=2), {})
        self.assertEqual([item['prompt'] for item in self.requests], ['a', 'b', 'a', 'b'])
        self.assertEqual(result.all_seeds, [7, 8, 9, 10])
        with self.assertRaisesRegex(ValueError, '1–64'):
            runtime.generate(self.p(batch_size=8, n_iter=9), {})

    def test_fractional_batch_controls_are_rejected(self):
        for values in ({'batch_size': 1.5}, {'n_iter': 2.5}):
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, 'whole numbers'):
                runtime.generate(self.p(**values), {})

    def test_reference_order_limit_and_type(self):
        red, blue = Image.new('RGB', (8, 8), 'red'), Image.new('RGB', (8, 8), 'blue')
        runtime.generate(self.p(True, init_images=[red]), {'refs': [blue]})
        self.assertEqual(self.requests[0]['reference_pixels'], [(255, 0, 0), (0, 0, 255)])
        with self.assertRaisesRegex(ValueError, 'at most nine'):
            runtime.generate(self.p(True, init_images=[red]), {'refs': [blue] * 9})
        before = len(self.workers)
        with self.assertRaisesRegex(ValueError, 'must be an image'):
            runtime.generate(self.p(True, init_images=['bad']), {})
        self.assertEqual(len(self.workers), before)

    def test_prompt_and_dropdown_loras_are_forwarded_and_bounded(self):
        first = str(self.folder / 'first.safetensors')
        second = str(self.folder / 'second.safetensors')
        self.inventory['lora'][:] = [first, second]
        runtime.generate(self.p(prompt='cat <lora:first:0.25>'),
                         {'lora': second, 'strength': -0.5})
        self.assertEqual(self.requests[0]['prompt'], 'cat')
        self.assertEqual(self.requests[0]['adapters'], [
            {'path': first, 'strength': 0.25}, {'path': second, 'strength': -0.5}])
        for strength in (3, float('nan'), 'bad'):
            with self.subTest(strength=strength), self.assertRaisesRegex(ValueError, 'LoRA strength'):
                runtime.generate(self.p(), {'lora': first, 'strength': strength})

    def test_worker_reuse_and_memory_switch(self):
        runtime.generate(self.p(), {'keep_loaded': True, 'memory': 'auto'})
        first = runtime._worker
        runtime.generate(self.p(), {'keep_loaded': True, 'memory': 'auto'})
        self.assertIs(runtime._worker, first)
        runtime.generate(self.p(), {'keep_loaded': True, 'memory': 'cpu'})
        self.assertIsNot(runtime._worker, first)
        self.assertIn(first, self.closed)
        with self.assertRaisesRegex(ValueError, 'memory mode'):
            runtime.generate(self.p(), {'memory': 'invalid'})

    def test_stop_before_start_creates_no_worker(self):
        self.state.interrupted = True
        result = runtime.generate(self.p(), {})
        self.assertEqual(result.images, [])
        self.assertEqual(self.workers, [])
        self.assertEqual(result.info, 'H3 stopped.')

    def test_stop_and_skip_during_generation(self):
        owner = self

        class StopWorker:
            mode = 'auto'
            process = types.SimpleNamespace(poll=lambda: None)

            def generate(self, data, cancelled, update):
                owner.state.interrupted = True
                raise InterruptedError()

            def close(self):
                owner.closed.append(self)

        runtime._worker = StopWorker()
        result = runtime.generate(self.p(), {'keep_loaded': True})
        self.assertEqual(result.images, [])
        self.assertIsNone(runtime._worker)

        self.state.interrupted = False

        class SkipWorker(StopWorker):
            def generate(self, data, cancelled, update):
                owner.state.skipped = True
                raise InterruptedError()

        skipped = SkipWorker()
        runtime._worker = skipped
        result = runtime.generate(self.p(batch_size=2), {})
        self.assertEqual(result.all_seeds, [8])
        self.assertIn(skipped, self.closed)


if __name__ == '__main__':
    unittest.main()
