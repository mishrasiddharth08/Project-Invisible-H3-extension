from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock

from pi_h3.worker import Engine


class WorkerPhaseMemoryTests(TestCase):
    def test_conditioning_offloads_only_clip_and_vae_before_sample(self):
        events = []
        engine = Engine()
        dit_patcher = object()
        clip_patcher = object()
        vae_patcher = object()
        engine.model = SimpleNamespace(patcher=dit_patcher)
        engine.clip = SimpleNamespace(patcher=clip_patcher)
        engine.vae = SimpleNamespace(patcher=vae_patcher)
        cached = {'prompt-key': object()}
        engine.cond_cache = cached

        def condition(*args):
            events.append('condition')
            return 'positive', 'negative'

        engine.cached_condition = Mock(side_effect=condition)

        class Memory:
            @staticmethod
            def unload_model_and_clones(patcher, unload_additional_models=True):
                events.append(('offload', patcher, unload_additional_models))

        request = {'prompt': 'portrait', 'negative': '', 'references': [], 'cfg': 1}
        result = engine.conditioning_for_sample(request, 768, 1024, Memory)

        self.assertEqual(result, ('positive', 'negative'))
        self.assertEqual(events, [
            'condition',
            ('offload', clip_patcher, False),
            ('offload', vae_patcher, False),
        ])
        self.assertNotIn(('offload', dit_patcher, False), events)
        self.assertIs(engine.cond_cache, cached)

    def test_missing_patcher_is_harmless(self):
        engine = Engine()
        engine.clip = SimpleNamespace()
        engine.vae = None
        engine.cached_condition = Mock(return_value=('positive', 'negative'))
        memory = SimpleNamespace(unload_model_and_clones=Mock())
        request = {'prompt': 'portrait', 'negative': '', 'references': [], 'cfg': 1}

        self.assertEqual(engine.conditioning_for_sample(request, 768, 1024, memory),
                         ('positive', 'negative'))
        memory.unload_model_and_clones.assert_not_called()
