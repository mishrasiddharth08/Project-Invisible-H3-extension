import sys,types,unittest
from pathlib import Path
from unittest.mock import Mock,patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from forge_h3.ui import remember_output
from forge_h3.contracts import H3Error

class OutputPersistenceTests(unittest.TestCase):
    def test_primary_output_is_saved_before_component_callbacks(self):
        opts=types.SimpleNamespace(set=Mock(),save=Mock())
        modules=types.ModuleType('modules');modules.shared=types.SimpleNamespace(opts=opts,config_filename='test-config.json')
        with patch.dict(sys.modules,{'modules':modules}):
            for mode in ('Still image','Video'):remember_output(mode)
            with self.assertRaises(H3Error):remember_output('bad')
        self.assertEqual(opts.set.call_count,2);opts.set.assert_called_with('pi_h3_output','Video')
        self.assertEqual(opts.save.call_count,2);opts.save.assert_called_with('test-config.json')
