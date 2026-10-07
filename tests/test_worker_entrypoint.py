import runpy
import sys
from pathlib import Path
from unittest import TestCase
from unittest.mock import Mock, patch

class WorkerScriptTests(TestCase):
    def test_conditioning_runs_when_worker_is_loaded_as_a_script(self):
        path = Path(__file__).resolve().parents[1] / 'pi_h3/worker.py'
        with patch.object(sys, 'argv', [str(path)]), patch.object(sys, 'stdout', sys.stdout), patch.object(sys, 'path', list(sys.path)):
            module = runpy.run_path(str(path), run_name='h3_worker_script_test')
            engine = module['Engine']()
            engine.condition = Mock(return_value=['conditioning'])
            self.assertEqual(engine.cached_condition('portrait', '', [], 768, 1024, 1), (['conditioning'], ['conditioning']))
            engine.condition.assert_called_once()
