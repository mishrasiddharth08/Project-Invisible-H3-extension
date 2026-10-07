from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch
import tempfile

from pi_h3 import forge, preset, runtime
from pi_h3.config import LABEL, OUTPUT_KEY, PRESET, STILL_OUTPUT, VIDEO_OUTPUT, VIDEO_PRESET
from pi_h3.worker import Engine


class NativeRoutingTests(TestCase):
    def modules(self, active=PRESET, output=STILL_OUTPUT):
        marker = SimpleNamespace(filename='C:/models/minimax_h3_fl2va.safetensors', _pi_h3=True)
        physical = SimpleNamespace(filename='C:/models/minimax_h3_fl2va.safetensors')
        sd_models = SimpleNamespace(
            checkpoints_list={LABEL: marker, 'H3 physical': physical},
            checkpoint_aliases={LABEL: marker, 'H3 physical': physical})
        opts = SimpleNamespace(sd_model_checkpoint='H3 physical', forge_preset=active,
                               pi_h3_output=output, forge_additional_modules=[])
        return SimpleNamespace(shared=SimpleNamespace(opts=opts), sd_models=sd_models)

    def test_video_output_routes_physical_checkpoint_to_native(self):
        modules = self.modules(output=VIDEO_OUTPUT)
        p = SimpleNamespace(override_settings={
            'sd_model_checkpoint': 'H3 physical', 'forge_preset': PRESET,
            OUTPUT_KEY: VIDEO_OUTPUT})
        original = Mock(return_value='native')
        original._pi_h3_process = False
        with patch('pi_h3.memory_policy.gpu_memory', return_value=None), patch.dict('sys.modules', {'modules': modules}), patch.object(
                forge, '_cancel'), patch.object(
                preset, 'native_defaults', return_value=('H3 physical', ['clip', 'video', 'audio'])):
            wrapped = forge._wrap_process(original)
            self.assertEqual(wrapped(p), 'native')
        original.assert_called_once_with(p)

    def test_synthetic_still_keeps_worker(self):
        modules = self.modules()
        p = SimpleNamespace(override_settings={
            'sd_model_checkpoint': LABEL, 'forge_preset': PRESET,
            OUTPUT_KEY: STILL_OUTPUT}, scripts=None)
        original = Mock()
        original._pi_h3_process = False
        with patch.dict('sys.modules', {'modules': modules}), patch.object(
                forge.runtime, 'generate', return_value='still') as generate:
            self.assertTrue(forge.selected(p))
            self.assertEqual(forge._wrap_process(original)(p), 'still')
        generate.assert_called_once()
        original.assert_not_called()

    def test_physical_safetensors_still_uses_worker(self):
        modules = self.modules(PRESET)
        with patch.dict('sys.modules', {'modules': modules}):
            self.assertTrue(forge.selected())
            modules.shared.opts.pi_h3_output = VIDEO_OUTPUT
            self.assertFalse(forge.selected())

    def test_legacy_api_preset_normalizes_to_video_and_native_defaults(self):
        modules = self.modules()
        p = SimpleNamespace(override_settings={
            'sd_model_checkpoint': LABEL, 'forge_preset': VIDEO_PRESET}, scripts=None)
        with patch.dict('sys.modules', {'modules': modules}), patch.object(
                preset, 'native_defaults', return_value=('H3 physical', ['clip', 'video', 'audio'])):
            forge._normalize_request(p)
        self.assertEqual(p.override_settings['forge_preset'], PRESET)
        self.assertEqual(p.override_settings[OUTPUT_KEY], VIDEO_OUTPUT)
        self.assertEqual(p.override_settings['sd_model_checkpoint'], 'H3 physical')
        self.assertEqual(p.override_settings['forge_additional_modules'], ['clip', 'video', 'audio'])

    def test_output_precedence_request_then_script_then_saved(self):
        modules = self.modules(output=STILL_OUTPUT)
        script = SimpleNamespace(_pi_h3=True, args_from=2, args_to=11)
        runner = SimpleNamespace(alwayson_scripts=[script])
        p = SimpleNamespace(override_settings={OUTPUT_KEY: VIDEO_OUTPUT}, scripts=runner,
                            script_args=[None, None] + [None] * 8 + [STILL_OUTPUT])
        with patch.dict('sys.modules', {'modules': modules}):
            self.assertEqual(forge.output_mode(p), VIDEO_OUTPUT)
            p.override_settings = {}
            self.assertEqual(forge.output_mode(p), STILL_OUTPUT)
            p.script_args[-1] = None
            self.assertEqual(forge.output_mode(p), STILL_OUTPUT)


class ConditionCacheTests(TestCase):
    def comfy(self):
        comfy = ModuleType('comfy')
        comfy.__path__ = []
        sd = ModuleType('comfy.sd')
        utils = ModuleType('comfy.utils')
        mm = ModuleType('comfy.model_management')
        model_type = type('MiniMaxH3', (), {})
        tokenizer_type = type('MiniMaxTokenizer', (), {})
        sd.load_diffusion_model = Mock(return_value=SimpleNamespace(model=model_type()))
        sd.CLIPType = SimpleNamespace(MINIMAX='MINIMAX')
        sd.load_clip = Mock(return_value=SimpleNamespace(tokenizer=tokenizer_type()))
        fsm = SimpleNamespace(_adaptive_decode=True, _finalize_pixels=True,
                              latents_mean=None, latents_std=None)
        sd.VAE = Mock(return_value=SimpleNamespace(first_stage_model=fsm))
        utils.load_torch_file = Mock(return_value={})
        mm.unload_all_models = Mock()
        comfy.sd, comfy.utils, comfy.model_management = sd, utils, mm
        return {'comfy': comfy, 'comfy.sd': sd, 'comfy.utils': utils,
                'comfy.model_management': mm}

    def test_model_switch_clears_conditioning_but_same_model_reuses_it(self):
        engine = Engine()
        engine.key = ('old',)
        engine.cond_cache = {'old prompt': object()}
        paths = {'dit': 'dit', 'clip': 'clip', 'vae': 'vae'}
        with patch.dict('sys.modules', self.comfy()), patch(
                'pi_h3.assets.fingerprint', return_value=('new',)):
            engine.load(paths)
        self.assertEqual(engine.cond_cache, {})
        engine.cond_cache['new prompt'] = object()
        with patch.dict('sys.modules', self.comfy()), patch(
                'pi_h3.assets.fingerprint', return_value=('new',)):
            engine.load(paths)
        self.assertIn('new prompt', engine.cond_cache)


class PresetBridgeTests(TestCase):
    def test_native_module_scan_skips_nvfp4_and_selects_both_vaes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ('encoder-nvfp4.safetensors', 'encoder-int4.safetensors',
                         'encoder-int8.safetensors',
                         'video.safetensors', 'audio.safetensors'):
                (root / name).touch()

            class H3Error(RuntimeError):
                pass

            def inspect(path):
                name = Path(path).name
                values = {
                    'encoder-nvfp4.safetensors': ('text_encoder', 'nvfp4'),
                    'encoder-int4.safetensors': ('text_encoder', 'int4_convrot'),
                    'encoder-int8.safetensors': ('text_encoder', 'int8'),
                    'video.safetensors': ('video_vae', 'plain'),
                    'audio.safetensors': ('audio_vae', 'plain'),
                }
                role, quantization = values[name]
                return SimpleNamespace(path=Path(path).resolve(), role=role, quantization=quantization)

            package = ModuleType('forge_h3')
            package.__path__ = []
            contracts = ModuleType('forge_h3.contracts')
            contracts.H3Error = H3Error
            models = ModuleType('forge_h3.models')
            models.inspect_model = inspect
            inventory = {'clip': [], 'vae': []}
            with patch.dict('sys.modules', {'forge_h3': package, 'forge_h3.contracts': contracts,
                                             'forge_h3.models': models}):
                selected = preset._native_modules(inventory, [root])
        self.assertEqual({Path(path).name for path in selected},
                         {'encoder-int4.safetensors', 'video.safetensors', 'audio.safetensors'})

    def test_native_module_scan_reuses_inventory_and_avoids_checkpoint_tree(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            components = root / 'components'
            checkpoints = root / 'Stable-diffusion'
            components.mkdir()
            checkpoints.mkdir()
            known_clip = checkpoints / 'named-qwen.safetensors'
            ignored_checkpoint = checkpoints / 'unrelated.safetensors'
            renamed_video = components / 'renamed-a.safetensors'
            renamed_audio = components / 'renamed-b.safetensors'
            for path in (known_clip, ignored_checkpoint, renamed_video, renamed_audio):
                path.touch()

            inspected = []

            class H3Error(RuntimeError):
                pass

            def inspect(path):
                path = Path(path)
                inspected.append(path.resolve())
                roles = {'named-qwen.safetensors': 'text_encoder',
                         'renamed-a.safetensors': 'video_vae',
                         'renamed-b.safetensors': 'audio_vae'}
                role = roles.get(path.name)
                return None if role is None else SimpleNamespace(
                    path=path.resolve(), role=role, quantization='plain')

            package = ModuleType('forge_h3')
            package.__path__ = []
            contracts = ModuleType('forge_h3.contracts')
            contracts.H3Error = H3Error
            models = ModuleType('forge_h3.models')
            models.inspect_model = inspect
            inventory = {'clip': [str(known_clip)], 'vae': []}
            with patch.dict('sys.modules', {'forge_h3': package, 'forge_h3.contracts': contracts,
                                             'forge_h3.models': models}):
                selected = preset._native_modules(inventory, [components])

        self.assertNotIn(ignored_checkpoint.resolve(), inspected)
        self.assertEqual({Path(path).name for path in selected},
                         {'named-qwen.safetensors', 'renamed-a.safetensors', 'renamed-b.safetensors'})

    def test_one_h3_preset_registers_output_setting_and_caches_native_defaults(self):
        def info(default=None):
            return SimpleNamespace(default=default, section=('ui_sd', 'SD'), component_args={})

        template = {
            'forge_checkpoint_sd': info(), 'forge_additional_modules_sd': info([]),
            'forge_unet_storage_dtype_sd': info('Automatic'),
        }
        for suffix, default in (
                ('t2i_sampler', 'Euler a'), ('i2i_sampler', 'Euler a'),
                ('t2i_scheduler', 'Automatic'), ('i2i_scheduler', 'Automatic'),
                ('t2i_step', 32), ('t2i_hr_step', 32), ('i2i_step', 32),
                ('t2i_cfg', 6.0), ('t2i_hr_cfg', 6.0), ('i2i_cfg', 6.0),
                ('t2i_width', 0), ('i2i_width', 0), ('t2i_height', 0), ('i2i_height', 0),
                ('t2i_batch_size', 1), ('i2i_batch_size', 1)):
            template['sd_' + suffix] = info(default)

        class PresetArch:
            choices = staticmethod(lambda: ['sd'])

        presets = ModuleType('modules_forge.presets')
        presets.PresetArch = PresetArch
        presets.register = lambda result: result.update(template)
        modules_forge = ModuleType('modules_forge')
        modules_forge.presets = presets
        options = {}
        opts = SimpleNamespace(data_labels=options, sd_checkpoint_dropdown_use_short=False,
                               forge_preset=VIDEO_PRESET, pi_h3_output=STILL_OUTPUT)
        opts.add_option = lambda key, value: options.setdefault(key, value)
        opts.set = lambda key, value: setattr(opts, key, value)
        physical = SimpleNamespace(filename='C:/models/minimax_h3_fl2va.safetensors',
                                   name='minimax_h3_fl2va', short_title='h3')
        modules = ModuleType('modules')
        modules.shared = SimpleNamespace(opts=opts, OptionInfo=lambda default, *args, **kwargs:
                                         SimpleNamespace(default=default, component_args=kwargs.get('component_args', {})))
        modules.sd_models = SimpleNamespace(checkpoints_list={'physical': physical})
        inventory = {'dit': [physical.filename], 'clip': [], 'vae': [], 'lora': []}
        native_modules = ['encoder.safetensors', 'video.safetensors', 'audio.safetensors']
        with patch.dict('sys.modules', {'modules': modules, 'modules_forge': modules_forge,
                                         'modules_forge.presets': presets}), patch.object(
                preset, 'scan', return_value=inventory), patch.object(
                preset, '_native_modules', return_value=native_modules) as native_scan:
            preset.install()

        native_scan.assert_called_once_with(inventory)

        self.assertEqual(options['forge_checkpoint_' + PRESET].default, LABEL)
        self.assertEqual(options[OUTPUT_KEY].default, STILL_OUTPUT)
        self.assertEqual(opts.forge_preset, PRESET)
        self.assertEqual(opts.pi_h3_output, VIDEO_OUTPUT)
        self.assertEqual(preset.native_defaults(), (physical.name, native_modules))
        self.assertEqual(PresetArch.choices(), ['sd', PRESET])

    def test_sampler_metadata_uses_effective_fallback(self):
        self.assertEqual(runtime._sampler_label('er_sde'), 'ER SDE')


if __name__ == '__main__':
    import unittest
    unittest.main()
