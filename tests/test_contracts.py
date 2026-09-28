import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import types
import unittest
from unittest.mock import patch, Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pi_h3 import assets, request, forge, runtime
from pi_h3.progress import Progress


class Requests(unittest.TestCase):
    def test_native_defaults(self):
        self.assertEqual(request.validate(1536,1536,50,1,'ER SDE','Simple'), ('er_sde','simple'))

    def test_bad_dimensions(self):
        for size in (0,63,513,4097):
            with self.subTest(size=size), self.assertRaises(ValueError):
                request.validate(size,512,20,1,'ER SDE','Simple')

    def test_mask_is_not_silently_ignored(self):
        with self.assertRaisesRegex(ValueError,'masks'):
            request.validate(512,512,20,1,'ER SDE','Simple',True,object(),object())

    def test_edit_requires_image(self):
        with self.assertRaisesRegex(ValueError,'Add an image'):
            request.validate(512,512,20,1,'ER SDE','Simple',True)

    def test_denoising_is_explicit(self):
        with self.assertRaisesRegex(ValueError,'Denoising'):
            request.validate(512,512,20,1,'ER SDE','Simple',True,object(),denoise=.75)

    def test_unsupported_native_features(self):
        for kwargs in ({'sampler':'Euler a'},{'scheduler':'unknown'},{'cfg':float('nan')},{'steps':0},{'hires':True}):
            data=dict(width=512,height=512,steps=20,cfg=1,sampler='ER SDE',scheduler='Simple')
            data.update(kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError): request.validate(**data)

    def test_lora_tags(self):
        self.assertEqual(request.parse_loras('cat <lora:minimax_h3_turbo:0.38>'), ('cat',[('minimax_h3_turbo',.38)]))

    def test_bad_adapter_fails(self):
        for tag in ('<lora:test:4>','<lyco:test:1>','<lora:test:broken>'):
            with self.assertRaises(ValueError): request.parse_loras(tag)


class Assets(unittest.TestCase):
    def test_user_renamed_components(self):
        for name,role in [('MiniMax-H3-FL2VA-Pruned-int8_convrot.safetensors','dit'),
                          ('Qwen3-VL-32B-MiniMax-H3-TextEncoder-nvfp4-AWQ--For-MiniMax-H3.safetensors','clip'),
                          ('MiniMax-H3-Video-VAE-FP16--For-MiniMax-H3.safetensors','vae')]:
            self.assertEqual(assets.classify(name),role)

    def test_audio_and_old_qwen_rejected(self):
        self.assertIsNone(assets.classify('MiniMax-H3-Audio-VAE-FP32.safetensors'))
        self.assertIsNone(assets.classify('qwen3vl_8b.safetensors'))

    def test_incomplete_download(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'model.safetensors'
            h=json.dumps({'x':{'dtype':'F32','shape':[2],'data_offsets':[0,8]}}).encode()
            path.write_bytes(struct.pack('<Q',len(h))+h+b'1234')
            with self.assertRaisesRegex(ValueError,'incomplete'): assets.header(path)

    def test_tensor_family_required(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'minimax_h3_fl2va.safetensors'
            h=json.dumps({'unrelated':{'dtype':'F32','shape':[1],'data_offsets':[0,4]}}).encode()
            path.write_bytes(struct.pack('<Q',len(h))+h+b'1234')
            with self.assertRaisesRegex(ValueError,'tensor structure'):
                assets.resolve('Auto','dit',{'dit':[str(path)]})

    def test_discovery_deduplicates_and_ignores_partial(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'minimax_h3_fl2va.safetensors'
            path.touch()
            path.with_suffix('.safetensors.partial').touch()
            self.assertEqual(assets.scan([folder,folder])['dit'],[str(path.resolve())])

    def test_missing_component_actionable(self):
        with self.assertRaisesRegex(ValueError,'Models'):
            assets.resolve('Auto','dit',{'dit':[]})


class Hooks(unittest.TestCase):
    def setUp(self):
        self.calls=[]
        def original(*args,**kwargs):
            self.calls.append((args,kwargs))
            return 'original-return'
        self.run=original
        runner=type('Runner',(),{'run':original})
        class CI:
            def register(item):
                sd.checkpoints_list[item.title]=item
                for alias in item.ids: sd.checkpoint_aliases[alias]=item
        sd=types.SimpleNamespace(CheckpointInfo=CI,checkpoints_list={},checkpoint_aliases={},list_models=original)
        self.sd=sd
        self.scripts=types.SimpleNamespace(ScriptRunner=runner)
        self.processing=types.SimpleNamespace(process_images=original)
        self.callbacks=types.SimpleNamespace(on_app_started=Mock(),on_script_unloaded=Mock())
        self.shared=types.SimpleNamespace(opts=types.SimpleNamespace(sd_model_checkpoint='ordinary',data_labels={}))
        modules=types.ModuleType('modules')
        modules.scripts=self.scripts; modules.processing=self.processing; modules.sd_models=sd
        modules.script_callbacks=self.callbacks; modules.shared=self.shared
        self.context=patch.dict(sys.modules,{'modules':modules})
        self.context.start()

    def tearDown(self): self.context.stop()

    def test_passthrough_all_arguments_and_return(self):
        forge.install()
        p=types.SimpleNamespace(override_settings={})
        with patch.object(runtime,'release'):
            self.assertEqual(self.processing.process_images(p,17, future='yes'),'original-return')
            runner=self.scripts.ScriptRunner()
            self.assertEqual(runner.run(p,23, future='yes'),'original-return')
        self.assertEqual(self.calls[0],((p,17),{'future':'yes'}))
        self.assertEqual(self.calls[1],((runner,p,23),{'future':'yes'}))

    def test_override_routes_only_h3(self):
        forge.install()
        p=types.SimpleNamespace(override_settings={'sd_model_checkpoint':forge.LABEL})
        with patch.object(runtime,'generate',return_value='H3 result') as generate:
            self.assertEqual(self.processing.process_images(p),'H3 result')
            generate.assert_called_once()

    def test_registration_idempotent_real_marker(self):
        forge.register(); forge.register()
        self.assertEqual(len(self.sd.checkpoints_list),1)
        item=self.sd.checkpoints_list[forge.LABEL]
        self.assertTrue(Path(item.filename).is_file())
        self.assertIsNone(item.calculate_shorthash())

    def test_wrappers_not_stacked(self):
        forge.install(); old=self.scripts.ScriptRunner.run
        forge.install()
        self.assertIs(self.scripts.ScriptRunner.run,old)

    def test_refresh_restores_marker(self):
        forge.install(); self.sd.checkpoints_list.clear()
        self.assertEqual(self.sd.list_models(new_argument=True),'original-return')
        self.assertIn(forge.LABEL,self.sd.checkpoints_list)

    def test_refresh_rejects_foreign_checkpoint_repair(self):
        previous = Mock()
        previous._pi_h3 = False
        opts = self.shared.opts
        opts.forge_preset = forge.PRESET
        opts.sd_model_checkpoint = forge.LABEL
        opts.data_labels['sd_model_checkpoint'] = types.SimpleNamespace(onchange=previous)
        opts.set = Mock(side_effect=lambda key, value, **kwargs: setattr(opts, key, value))
        opts.onchange = Mock(side_effect=lambda key, callback, call=False: setattr(
            opts.data_labels[key], 'onchange', callback))
        forge.register()
        def refresh(*args, **kwargs):
            self.sd.checkpoints_list.clear()
            opts.sd_model_checkpoint = 'ordinary'
            opts.data_labels['sd_model_checkpoint'].onchange()
            return 'refreshed'
        self.sd.list_models = refresh
        forge.install_selection()
        with patch.object(forge, '_cancel') as cancel:
            forge.install()
            self.assertEqual(self.sd.list_models(), 'refreshed')
        self.assertEqual(opts.sd_model_checkpoint, forge.LABEL)
        self.assertIn(forge.LABEL, self.sd.checkpoints_list)
        cancel.assert_not_called()
        previous.assert_not_called()


class ProgressTests(unittest.TestCase):
    def setUp(self):
        self.state=types.SimpleNamespace(assign_current_image=Mock())
        self.shared=types.SimpleNamespace(state=self.state,total_tqdm=Mock(),opts=types.SimpleNamespace(live_previews_enable=True))
        self.progress=Progress(self.shared,4,2)

    def test_no_double_count_on_duplicate_callback(self):
        self.progress.start(0,512,512)
        self.progress.advance(1); self.progress.advance(1); self.progress.advance(3)
        self.assertEqual(self.shared.total_tqdm.update.call_count,3)

    def test_decode_is_not_premature_completion(self):
        self.progress.start(0,512,512); self.progress.advance(4)
        self.assertEqual(self.state.sampling_steps,5)
        self.assertEqual(self.state.sampling_step,4)
        self.progress.finish('finished-image')
        self.assertEqual(self.state.sampling_step,5)
        self.state.assign_current_image.assert_called_with('finished-image')

    def test_batch_resets_current_only(self):
        self.progress.start(0,512,512); self.progress.advance(5)
        self.progress.start(1,512,512)
        self.assertEqual(self.state.job_no,1)
        self.assertEqual(self.state.sampling_step,0)

    def test_disabled_preview_final_is_still_published(self):
        self.shared.opts.live_previews_enable=False
        progress=Progress(self.shared,4,1)
        progress.start(0,512,512)
        self.state.assign_current_image.assert_not_called()
        progress.finish('final')
        self.state.assign_current_image.assert_called_with('final')


class Downloads(unittest.TestCase):
    def test_unapproved_download_does_not_access_network(self):
        from pi_h3.downloads import download
        with patch('requests.get') as get:
            self.assertIn('approval',download(['anything'],False))
            get.assert_not_called()

    def test_unknown_file_blocked(self):
        from pi_h3.downloads import download
        with self.assertRaisesRegex(ValueError,'Unknown'): download(['../../other'],True)

    def test_generation_does_not_import_downloader(self):
        self.assertNotIn('downloads', (ROOT/'pi_h3/runtime.py').read_text())


if __name__=='__main__': unittest.main()
