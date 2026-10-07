from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch
from forge_h3 import ui


class QualityModeCallbackTests(TestCase):
    def test_real_output_event_resets_shift_in_both_tabs(self):
        for tab in ('txt2img', 'img2img'):
            with self.subTest(tab=tab):
                def component():
                    return SimpleNamespace(change=Mock(), value=3.0, label='native')
                controls = {name: component() for name in ui.NATIVE_CONTROLS}
                shift = controls['distilled_cfg_scale']
                primary = component()
                checkpoint, modules = component(), component()
                registry = {f'{tab}_{name}': value for name, value in controls.items()}
                registry.update({'setting_sd_model_checkpoint': checkpoint, 'setting_sd_modules': modules,
                                 f'{tab}_h3_duration': component(), f'{tab}_h3_output': primary})
                panel = ui.Panel.__new__(ui.Panel)
                panel.tab, panel.bound, panel.attempts = tab, False, []
                for name in ('accordion', 'audio', 'audio_shift', 'status', 'summary', 'saved'):
                    setattr(panel, name, component())
                with patch.dict(ui.COMPONENTS, registry, clear=True), patch.object(ui, '_in_blocks', return_value=True), \
                        patch.object(ui.gr.context.Context, 'root_block', SimpleNamespace(load=Mock())), \
                        patch.object(ui, 'remember_output'), \
                        patch('pi_h3.preset.native_defaults', return_value=('H3 physical', [
                            'C:/encoder.safetensors', 'C:/video.safetensors', 'C:/audio.safetensors'])):
                    panel.bind('quality regression')
                    event = primary.change.call_args_list[0]
                    self.assertIs(event.kwargs['outputs'][-1], shift)
                    updates = event.args[0]('Video')
                    self.assertEqual(updates[-1]['value'], 12.0)
                    # Switching back also produces all required native updates.
                    still = event.args[0]('Still image')
                    self.assertEqual(len(still), len(event.kwargs['outputs']))
                    self.assertEqual(still[-1]['value'], 12.0)
