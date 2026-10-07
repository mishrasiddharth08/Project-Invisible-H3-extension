from __future__ import annotations

import importlib.util
from contextlib import ExitStack
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from forge_h3 import integration


class NativeSelectionTests(unittest.TestCase):
    def setUp(self):
        self.physical = types.SimpleNamespace(filename="physical-h3.safetensors")
        self.synthetic = types.SimpleNamespace(filename="synthetic-h3.safetensors", _pi_h3=True)
        self.opts = types.SimpleNamespace(
            forge_preset="H3", sd_model_checkpoint="physical", pi_h3_output="Video"
        )
        self.sd_models = types.SimpleNamespace(
            get_closet_checkpoint_match=lambda value: {
                "physical": self.physical,
                "synthetic": self.synthetic,
                "ordinary": types.SimpleNamespace(filename="sd15.safetensors"),
            }.get(value)
        )
        self.modules = types.ModuleType("modules")
        self.modules.shared = types.SimpleNamespace(opts=self.opts)
        self.modules.sd_models = self.sd_models
        self.modules_patch = patch.dict(sys.modules, {"modules": self.modules})
        self.modules_patch.start()
        self.addCleanup(self.modules_patch.stop)

    def test_synthetic_checkpoint_is_never_native(self):
        self.opts.sd_model_checkpoint = "synthetic"
        self.assertIsNone(integration.checkpoint_info("synthetic"))
        self.assertFalse(integration.native_selected())

    def test_still_output_disables_native_route(self):
        self.opts.pi_h3_output = "Still image"
        with patch.object(integration, "inspect_model") as inspect:
            self.assertFalse(integration.native_selected())

    def test_sd_preset_disables_native_route(self):
        self.opts.forge_preset = "sd"
        with patch.object(integration, "inspect_model") as inspect:
            self.assertIsNone(integration.checkpoint_info("physical"))
            self.assertFalse(integration.native_selected())
        inspect.assert_not_called()

    def test_physical_h3_uses_native_route(self):
        h3 = types.SimpleNamespace(role="dit")
        with patch.object(integration, "inspect_model", return_value=h3):
            self.assertTrue(integration.native_selected())
            p = types.SimpleNamespace(override_settings={"sd_model_checkpoint": "physical"})
            self.assertIs(integration.select_h3(p), self.physical)

    def test_request_override_can_leave_global_still_preset(self):
        self.opts.forge_preset = "H3"
        p = types.SimpleNamespace(
            override_settings={
                "forge_preset": "H3 Video",
                "pi_h3_output": "Video",
                "sd_model_checkpoint": "physical",
            }
        )
        with patch.object(integration, "inspect_model", return_value=types.SimpleNamespace(role="dit")):
            self.assertTrue(integration.native_selected(p))
            self.assertIs(integration.select_h3(p), self.physical)

    def test_request_override_can_enter_still_route(self):
        p = types.SimpleNamespace(
            override_settings={
                "forge_preset": "H3",
                "pi_h3_output": "Still image",
                "sd_model_checkpoint": "physical",
            }
        )
        with patch.object(integration, "inspect_model") as inspect:
            self.assertFalse(integration.native_selected(p))
            self.assertIsNone(integration.select_h3(p))

    def test_non_h3_checkpoint_stays_on_forge_route(self):
        self.opts.sd_model_checkpoint = "ordinary"
        with patch.object(integration, "inspect_model", return_value=None):
            self.assertFalse(integration.native_selected())

    def test_physical_gguf_still_uses_native_route(self):
        self.physical.filename = "physical-h3.gguf"
        self.opts.pi_h3_output = "Still image"
        with patch.object(integration, "inspect_model", return_value=types.SimpleNamespace(role="dit")):
            self.assertTrue(integration.native_selected())


class NativeScriptTests(unittest.TestCase):
    def load_script(self, problems):
        import forge_h3.native as native

        callbacks = []
        callback_api = types.SimpleNamespace(
            **{
                name: (lambda callback, name=name: callbacks.append((name, callback)))
                for name in (
                    "on_ui_settings",
                    "on_before_ui",
                    "on_after_component",
                    "on_ui_tabs",
                    "on_app_started",
                    "on_script_unloaded",
                )
            }
        )
        modules = types.ModuleType("modules")
        modules.script_callbacks = callback_api
        modules.scripts = types.SimpleNamespace(Script=object, AlwaysVisible=True)
        modules.shared = types.SimpleNamespace(
            opts=types.SimpleNamespace(add_option=Mock()), OptionInfo=Mock()
        )
        source = ROOT / "scripts" / "native_h3.py"
        spec = importlib.util.spec_from_file_location(
            f"native_h3_test_{len(problems)}_{id(callbacks)}", source
        )
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        fake_patches = types.ModuleType("forge_h3.native.patches")
        fake_patches.apply = Mock()
        with (
            patch.dict(sys.modules, {"modules": modules, "forge_h3.native.patches": fake_patches}),
            patch.object(native, "patches", fake_patches, create=True),
            patch("forge_h3.native.compat.check", return_value=problems),
            patch("forge_h3.native.compat.report", return_value="compatibility failed"),
        ):
            spec.loader.exec_module(module)
        return module, callbacks

    def test_incompatible_forge_installs_no_hooks(self):
        module, callbacks = self.load_script(["missing Forge API"])
        self.assertFalse(module.ENABLED)
        self.assertEqual(callbacks, [])
        self.assertEqual(module.STATUS, "compatibility failed")

    def test_callbacks_run_only_for_native_selection(self):
        module, callbacks = self.load_script([])
        self.assertTrue(module.ENABLED)
        self.assertEqual(len(callbacks), 6)
        script = module.Script()
        p, processed = object(), object()
        calls = [
            "script_before_process",
            "process",
            "before_sampling",
            "after_sampling",
            "postprocess",
        ]
        mocks = {name: Mock() for name in calls}
        with ExitStack() as stack:
            stack.enter_context(patch.object(module.integration, "native_selected", return_value=False))
            for name, mock in mocks.items():
                stack.enter_context(patch.object(module.integration, name, mock))
            script.before_process(p)
            script.process(p)
            script.process_before_every_sampling(p, noise=object())
            script.post_sample(p, object())
            script.postprocess(p, processed)
        self.assertTrue(all(mock.call_count == 0 for mock in mocks.values()))

        with ExitStack() as stack:
            stack.enter_context(patch.object(module.integration, "native_selected", return_value=True))
            for name, mock in mocks.items():
                stack.enter_context(patch.object(module.integration, name, mock))
            script.before_process(p)
            script.process(p)
            script.process_before_every_sampling(p, noise=object())
            script.post_sample(p, object())
            script.postprocess(p, processed)
        self.assertTrue(all(mock.call_count == 1 for mock in mocks.values()))

    def test_native_script_keeps_three_arguments(self):
        module, _callbacks = self.load_script([])
        script = module.Script()
        panel = types.SimpleNamespace(inputs=[object(), object(), object()], audio_shift=object())
        with patch.object(module.ui, "Panel", return_value=panel):
            self.assertEqual(len(script.ui(False)), 3)

    def test_native_callback_uses_primary_output_resolver(self):
        p = types.SimpleNamespace()
        with patch("pi_h3.forge.output_mode", return_value="Still image"), patch.object(
                integration, "_before_process") as run:
            integration.before_process(p, "Video", True, 3.0)
        run.assert_called_once_with(p, "Still image", True, 3.0)


class NativePolicyTests(unittest.TestCase):
    def test_native_uses_the_one_h3_preset(self):
        source = (ROOT / "forge_h3" / "native" / "presets.py").read_text(encoding="utf-8")
        self.assertIn('PRESET = "H3"', source)
        self.assertNotIn('PRESET = "H3 Video"', source)

    def test_generate_never_downloads_taeh3(self):
        source = (ROOT / "forge_h3" / "native" / "patches.py").read_text(encoding="utf-8")
        self.assertNotIn("sd_vae_taesd.download_model(", source)
        self.assertIn("if os.path.isfile(path):", source)


if __name__ == "__main__":
    unittest.main()
