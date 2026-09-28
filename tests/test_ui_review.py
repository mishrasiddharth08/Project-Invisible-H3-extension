from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

from pi_h3 import forge, runtime

ROOT = Path(__file__).resolve().parents[1]


class HookReview(TestCase):
    def _selection_modules(self, checkpoint=forge.LABEL, preset='H3', previous=None):
        opts = SimpleNamespace(
            sd_model_checkpoint=checkpoint,
            forge_preset=preset,
            data_labels={'sd_model_checkpoint': SimpleNamespace(onchange=previous)})
        opts.set = Mock(side_effect=lambda key, value, **kwargs: setattr(opts, key, value))
        opts.onchange = Mock(side_effect=lambda key, callback, call=False: setattr(
            opts.data_labels[key], 'onchange', callback))
        marker = SimpleNamespace(filename='C:/models/h3.safetensors', _pi_h3=True)
        ordinary = SimpleNamespace(filename='C:/models/ordinary.safetensors')
        sd_models = SimpleNamespace(
            checkpoints_list={forge.LABEL: marker, 'ordinary': ordinary}, checkpoint_aliases={})
        return opts, SimpleNamespace(shared=SimpleNamespace(opts=opts), sd_models=sd_models)

    def test_nested_foreign_restore_is_rejected_without_cancel(self):
        opts, modules = self._selection_modules()
        def previous():
            opts.sd_model_checkpoint = 'ordinary'
            opts.data_labels['sd_model_checkpoint'].onchange()
        opts.data_labels['sd_model_checkpoint'].onchange = previous
        with patch.dict('sys.modules', {'modules': modules}), patch.object(forge, '_cancel') as cancel:
            forge.install_selection()
            opts.data_labels['sd_model_checkpoint'].onchange()
        self.assertEqual(opts.sd_model_checkpoint, forge.LABEL)
        cancel.assert_not_called()

    def test_direct_api_checkpoint_switch_is_not_rejected(self):
        previous = Mock()
        previous._pi_h3 = False
        opts, modules = self._selection_modules(previous=previous)
        with patch.dict('sys.modules', {'modules': modules}), patch.object(forge, '_cancel') as cancel:
            forge.install_selection()
            opts.sd_model_checkpoint = 'ordinary'
            opts.data_labels['sd_model_checkpoint'].onchange()
        self.assertEqual(opts.sd_model_checkpoint, 'ordinary')
        cancel.assert_called_once_with('option changed: sd_model_checkpoint')
        previous.assert_called_once()

    def test_marker_and_physical_alias_are_same_checkpoint(self):
        marker = SimpleNamespace(filename='C:/models/h3.safetensors')
        physical = SimpleNamespace(filename='C:/models/h3.safetensors')
        sd_models = SimpleNamespace(
            checkpoints_list={forge.LABEL: marker, 'h3-file': physical}, checkpoint_aliases={})
        modules = SimpleNamespace(sd_models=sd_models)
        with patch.dict('sys.modules', {'modules': modules}):
            self.assertTrue(forge._same_checkpoint(forge.LABEL, 'h3-file'))
            physical.filename = 'C:/models/other-h3.safetensors'
            self.assertFalse(forge._same_checkpoint(forge.LABEL, 'h3-file'))

    def test_each_effective_process_chain_can_be_wrapped_once(self):
        original = Mock(return_value='ordinary')
        wrapped = forge._wrap_process(original)
        self.assertIs(forge._wrap_process(wrapped), wrapped)
        p = SimpleNamespace(override_settings={})
        with patch.object(forge, 'selected', return_value=False), patch.object(runtime, 'release'):
            self.assertEqual(wrapped(p, future=True), 'ordinary')
        original.assert_called_once_with(p, future=True)

    def test_h3_wrapper_routes_before_foreign_chain(self):
        foreign_calls = []
        def foreign(*args, **kwargs):
            foreign_calls.append((args, kwargs))
            raise AssertionError('foreign chain must not run')
        p = SimpleNamespace(override_settings={})
        with patch.object(forge, 'selected', return_value=True), patch.object(
                forge, 'options', return_value={}), patch.object(runtime, 'generate', return_value='H3'):
            self.assertEqual(forge._wrap_process(foreign)(p), 'H3')
        self.assertEqual(foreign_calls, [])

    def test_saved_h3_preset_restores_native_checkpoint(self):
        opts = SimpleNamespace(forge_preset='H3')
        opts.set = Mock(side_effect=lambda key, value: setattr(opts, key, value))
        sd_models = SimpleNamespace(checkpoints_list={}, checkpoint_aliases={})
        modules = SimpleNamespace(shared=SimpleNamespace(opts=opts), sd_models=sd_models)
        with patch.dict('sys.modules', {'modules': modules}):
            forge.restore_saved_selection()
        self.assertEqual(opts.sd_model_checkpoint, forge.LABEL)
        self.assertEqual(opts.forge_checkpoint_H3, forge.LABEL)
        self.assertEqual(opts.forge_additional_modules, [])

    def test_callbacks_return_after_forge_clears_them(self):
        callback_map = {key: [] for key in (
            'callbacks_app_started', 'callbacks_before_ui', 'callbacks_script_unloaded')}
        def register(category):
            return lambda callback: callback_map[category].append(SimpleNamespace(callback=callback))
        callbacks = SimpleNamespace(
            callback_map=callback_map,
            on_app_started=register('callbacks_app_started'),
            on_before_ui=register('callbacks_before_ui'),
            on_script_unloaded=register('callbacks_script_unloaded'),
        )
        forge._ensure_callback(callbacks, 'callbacks_app_started', callbacks.on_app_started, forge._ready)
        forge._ensure_callback(callbacks, 'callbacks_app_started', callbacks.on_app_started, forge._ready)
        self.assertEqual(len(callback_map['callbacks_app_started']), 1)
        for values in callback_map.values():
            values.clear()
        forge._ensure_callback(callbacks, 'callbacks_app_started', callbacks.on_app_started, forge._ready)
        forge._ensure_callback(callbacks, 'callbacks_before_ui', callbacks.on_before_ui, forge._ready)
        forge._ensure_callback(callbacks, 'callbacks_script_unloaded', callbacks.on_script_unloaded, runtime.selection_changed)
        self.assertTrue(all(len(values) == 1 for values in callback_map.values()))

    def test_ui_bindings_are_unique_and_cleared_before_rebuild(self):
        forge._UI_BINDINGS.clear()
        binding = (object(), True)
        forge.register_ui_binding(*binding)
        forge.register_ui_binding(*binding)
        self.assertEqual(forge._UI_BINDINGS, [binding])
        with patch.object(forge, 'install_selection'), patch.object(
                forge, 'install_forge_selection'), patch.object(
                forge, 'install_forge_ui_sync'), patch.object(
                forge, 'restore_saved_selection'), patch.object(forge, 'register'):
            modules = SimpleNamespace(processing=SimpleNamespace(process_images=lambda p: p))
            with patch.dict('sys.modules', {'modules': modules}):
                forge._ready()
        self.assertEqual(forge._UI_BINDINGS, [])


class UiReview(TestCase):
    def test_native_compact_ui_and_unique_state_ids(self):
        text = (ROOT / 'scripts' / 'engine.py').read_text(encoding='utf-8')
        self.assertNotIn('on_ui_tabs', text)
        self.assertNotIn('on_after_component', text)
        self.assertNotIn('Context.root_block.load', text)
        self.assertIn('register_ui_binding(panel, is_img2img)', text)
        self.assertNotIn('main_entry.ui_checkpoint', text)
        for control in ('dit', 'clip', 'vae', 'lora', 'strength', 'memory', 'keep'):
            self.assertIn(f"pi_h3_{{mode}}_{control}", text)
        self.assertIn("pi_h3_i2i_ref_{i}", text)
        forge_text = (ROOT / 'pi_h3' / 'forge.py').read_text(encoding='utf-8')
        self.assertIn('gr.update(value=1.0) if active', forge_text)

    def test_progress_adds_one_current_bar_and_labels_native_overall(self):
        text = (ROOT / 'javascript' / 'h3-progress.js').read_text(encoding='utf-8')
        self.assertEqual(text.count("document.createElement('div')"), 2)
        self.assertIn('Current image ${index}/${total}', text)
        self.assertIn('Overall:', text)
        self.assertIn("classList.add('pi-h3-dual-progress')", text)
        self.assertIn("classList.remove('pi-h3-dual-progress')", text)
        css = (ROOT / 'style.css').read_text(encoding='utf-8')
        self.assertIn('.pi-h3-dual-progress { margin-top: 24px; }', css)
        self.assertIn('.pi-h3-dual-progress > .pi-h3-current { top: -38px;', css)


if __name__ == '__main__':
    import unittest
    unittest.main()
