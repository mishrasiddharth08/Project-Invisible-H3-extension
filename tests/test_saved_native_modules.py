import sys,types,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pi_h3 import forge,preset

class SavedNativeModulesTests(unittest.TestCase):
    def test_components_require_both_vaes_and_compatible_encoder(self):
        def item(role,quant='plain'):return types.SimpleNamespace(role=role,quantization=quant)
        for quant,expected in [('nvfp4',False),('int4_convrot',True),('int8_convrot',True),('bf16',True)]:
            with self.subTest(quant=quant),patch('forge_h3.models.inspect_model',side_effect=[item('text_encoder',quant),item('video_vae'),item('audio_vae')]):
                self.assertEqual(preset.native_modules_valid(['te','video','audio']),expected)
        with patch('forge_h3.models.inspect_model',return_value=item('text_encoder','int4_convrot')):
            self.assertFalse(preset.native_modules_valid(['te']))
    def test_invalid_saved_modules_get_safe_defaults_but_explicit_requests_survive(self):
        modules=types.ModuleType('modules')
        modules.shared=types.SimpleNamespace(opts=types.SimpleNamespace(forge_preset='H3',pi_h3_output='Video',sd_model_checkpoint='physical',forge_additional_modules=['old-nvfp4']))
        modules.sd_models=types.SimpleNamespace(checkpoints_list={},checkpoint_aliases={})
        with patch.dict(sys.modules,{'modules':modules}),patch.object(preset,'native_defaults',return_value=('physical',['int4','video','audio'])),patch.object(preset,'native_modules_valid',return_value=False):
            p=types.SimpleNamespace(override_settings={})
            forge._normalize_request(p)
            self.assertEqual(p.override_settings['forge_additional_modules'],['int4','video','audio'])
            p=types.SimpleNamespace(override_settings={'forge_additional_modules':['manual']})
            forge._normalize_request(p)
            self.assertEqual(p.override_settings['forge_additional_modules'],['manual'])

    def test_explicit_empty_components_resolve_defaults(self):
        modules=types.ModuleType('modules')
        modules.shared=types.SimpleNamespace(opts=types.SimpleNamespace(forge_preset='H3',pi_h3_output='Video',sd_model_checkpoint='physical',forge_additional_modules=['valid-saved']))
        modules.sd_models=types.SimpleNamespace(checkpoints_list={},checkpoint_aliases={})
        with patch.dict(sys.modules,{'modules':modules}),patch.object(preset,'native_defaults',return_value=('physical',['int4','video','audio'])),patch.object(preset,'native_modules_valid',return_value=True):
            for empty in ([],None):
                with self.subTest(empty=empty):
                    p=types.SimpleNamespace(override_settings={'forge_additional_modules':empty})
                    forge._normalize_request(p)
                    self.assertEqual(p.override_settings['forge_additional_modules'],['int4','video','audio'])
