import sys,types,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pi_h3 import autolink,preset

class AutoLinkTests(unittest.TestCase):
    def test_invalid_automatic_video_set_is_replaced(self):
        resolver=types.SimpleNamespace(resolve=lambda *a,**k:(['nvfp4'],'minimax_h3','saved'))
        modules=types.ModuleType('modules');modules.shared=types.SimpleNamespace(opts=types.SimpleNamespace(forge_preset='H3',pi_h3_output='Video'))
        with patch.dict(sys.modules,{'model_autolink.integration':types.SimpleNamespace(_resolver=resolver),'modules':modules}),patch.object(preset,'native_modules_valid',side_effect=lambda p:p==['int4','video','audio']),patch.object(preset,'native_defaults',return_value=('dit',['int4','video','audio'])):
            autolink.install();wrapped=resolver.resolve;autolink.install()
            self.assertIs(wrapped,resolver.resolve)
            self.assertEqual(resolver.resolve('dit'),(['int4','video','audio'],'minimax_h3','H3 video compatibility'))
    def test_still_other_models_and_valid_video_sets_are_preserved(self):
        for output,family,valid in [('Still image','minimax_h3',False),('Video','qwen',False),('Video','minimax_h3',True)]:
            with self.subTest(output=output,family=family,valid=valid):
                original=(['selected'],family,'saved')
                resolver=types.SimpleNamespace(resolve=lambda *a,**k:original)
                modules=types.ModuleType('modules');modules.shared=types.SimpleNamespace(opts=types.SimpleNamespace(forge_preset='H3',pi_h3_output=output))
                with patch.dict(sys.modules,{'model_autolink.integration':types.SimpleNamespace(_resolver=resolver),'modules':modules}),patch.object(preset,'native_modules_valid',return_value=valid),patch.object(preset,'native_defaults') as defaults:
                    autolink.install();self.assertEqual(resolver.resolve('dit'),original);defaults.assert_not_called()
