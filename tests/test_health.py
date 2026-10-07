from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch
from pi_h3 import health, forge, runtime
from forge_h3 import integration

class HealthTests(TestCase):
    def test_snapshot_is_scoped_and_does_not_expose_paths(self):
        modules = SimpleNamespace(shared=SimpleNamespace(opts=SimpleNamespace(
            forge_preset='H3', pi_h3_output='Video')))
        with patch.dict('sys.modules', {'modules': modules}), patch.object(forge, 'selected', return_value=False), patch.object(integration, 'native_selected', return_value=True), patch.object(runtime, '_worker', None):
            value = health.snapshot()
        self.assertEqual(value['backend'], 'native')
        self.assertEqual(value['output'], 'Video')
        self.assertEqual(value['output_source'], 'settings')
        self.assertIn('pi_h3_output=Video', value['parameter_status'])
        self.assertFalse(value['worker_running'])
        self.assertNotIn('path', str(value))
    def test_status_never_spawns_a_worker(self):
        modules = SimpleNamespace(shared=SimpleNamespace(opts=SimpleNamespace(
            forge_preset='sd', pi_h3_output='Still image')))
        with patch.dict('sys.modules', {'modules': modules}), patch.object(forge, 'selected', return_value=False), patch.object(integration, 'native_selected', return_value=False), patch.object(runtime, '_worker', None):
            self.assertEqual(health.snapshot()['backend'], 'inactive')
            self.assertIsNone(runtime._worker)
    def test_route_registration_is_idempotent(self):
        app = SimpleNamespace(get=Mock(return_value=Mock()))
        health.app_started(None, app);health.app_started(None, app)
        app.get.assert_called_once_with('/pi-h3/status')
