import inspect
import unittest.mock
import types
import unittest
from pi_h3.startup import optimize_callbacks

class Startup(unittest.TestCase):
    def test_caller_identity_and_explicit_filename(self):
        module = types.ModuleType('fake_callbacks'); module.__file__ = 'callback_host.py'
        exec(compile("def register(items):\n    return add_callback(items, 17, name='custom', category='test')\n", module.__file__, 'exec'), module.__dict__)
        def original(items, fun, *, name=None, category='unknown', filename=None):
            items.append((fun, name, category, filename)); return 23
        module.add_callback = original
        optimize_callbacks(module); first = module.add_callback
        optimize_callbacks(module); self.assertIs(first, module.add_callback)
        items=[]
        with unittest.mock.patch('inspect.stack', side_effect=AssertionError('expensive stack walk')):
            self.assertEqual(module.register(items),23)
        self.assertEqual(items,[(17,'custom','test',__file__)])
        module.add_callback(items, 19, filename='explicit.py')
        self.assertEqual(items[-1][-1],'explicit.py')
    def test_unsupported_signature_keeps_host(self):
        original=lambda items,fun: None
        module=types.SimpleNamespace(add_callback=original)
        optimize_callbacks(module);self.assertIs(module.add_callback,original)
