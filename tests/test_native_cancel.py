import sys,types,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from unittest.mock import Mock,patch
from forge_h3.cancellation import guard_sample

class NativeCancelTests(unittest.TestCase):
    def test_cancelled_none_becomes_empty_batch_and_releases(self):
        for interrupted,skipped in ((True,False),(False,True)):
            with self.subTest(interrupted=interrupted):
                modules=types.ModuleType('modules')
                modules.shared=types.SimpleNamespace(state=types.SimpleNamespace(interrupted=interrupted,skipped=skipped))
                patches=types.ModuleType('forge_h3.native.patches');patches.end_sampling=Mock()
                p=types.SimpleNamespace(sample=Mock(return_value=None),h3_request=object(),sd_model=types.SimpleNamespace(release_generation=Mock()))
                with patch.dict(sys.modules,{'modules':modules,'forge_h3.native.patches':patches}):
                    guard_sample(p);wrapped=p.sample;guard_sample(p)
                    self.assertIs(wrapped,p.sample);self.assertEqual(p.sample(),[])
                    patches.end_sampling.assert_called_once();p.sd_model.release_generation.assert_called_once()
    def test_valid_results_and_unrelated_none_unchanged(self):
        modules=types.ModuleType('modules');modules.shared=types.SimpleNamespace(state=types.SimpleNamespace(interrupted=False,skipped=False))
        with patch.dict(sys.modules,{'modules':modules}):
            for value in (None,[1]):
                p=types.SimpleNamespace(sample=Mock(return_value=value),h3_request=object())
                guard_sample(p);self.assertIs(p.sample(),value)
