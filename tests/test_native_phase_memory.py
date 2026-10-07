from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock
import importlib.util

path = Path(__file__).resolve().parents[1] / 'forge_h3/native/phase_memory.py'
spec = importlib.util.spec_from_file_location('h3_phase_memory_test', path)
phase = importlib.util.module_from_spec(spec)
spec.loader.exec_module(phase)


class NativePhaseMemoryTests(TestCase):
    def test_only_completed_own_conditioning_is_offloaded(self):
        clip, vae, audio, dit, foreign = [object() for _ in range(5)]
        loaded = [SimpleNamespace(model=p, device='cuda:0') for p in (clip, vae, audio, dit, foreign)]
        memory = SimpleNamespace(current_loaded_models=loaded, free_memory=Mock())
        engine = SimpleNamespace(forge_objects=SimpleNamespace(clip=SimpleNamespace(patcher=clip),
                                 vae=SimpleNamespace(patcher=vae), unet=SimpleNamespace(patcher=dit)),
                                 audio_vae=SimpleNamespace(patcher=audio), cached_condition=object())
        cache = engine.cached_condition
        self.assertEqual(phase.release_conditioning(engine, memory), 3)
        call = memory.free_memory.call_args
        self.assertEqual(call.args, (1e30, 'cuda:0'))
        kept = call.kwargs['keep_loaded']
        self.assertEqual(len(kept), 2)
        self.assertIs(kept[0], loaded[3])
        self.assertIs(kept[1], loaded[4])
        self.assertIs(engine.cached_condition, cache)

    def test_no_loaded_conditioning_does_not_touch_other_models(self):
        foreign = SimpleNamespace(model=object(), device='cuda:0')
        memory = SimpleNamespace(current_loaded_models=[foreign], free_memory=Mock())
        self.assertEqual(phase.release_conditioning(SimpleNamespace(), memory), 0)
        memory.free_memory.assert_not_called()

    def test_devices_are_handled_without_unloading_foreign_models(self):
        clip, vae, foreign = object(), object(), object()
        loaded = [SimpleNamespace(model=clip, device='cuda:0'), SimpleNamespace(model=vae, device='cpu'),
                  SimpleNamespace(model=foreign, device='cuda:0')]
        memory = SimpleNamespace(current_loaded_models=loaded, free_memory=Mock())
        engine = SimpleNamespace(forge_objects=SimpleNamespace(clip=SimpleNamespace(patcher=clip),
                                                             vae=SimpleNamespace(patcher=vae)))
        self.assertEqual(phase.release_conditioning(engine, memory), 2)
        self.assertEqual([c.args[1] for c in memory.free_memory.call_args_list], ['cuda:0', 'cpu'])
        for call in memory.free_memory.call_args_list:
            self.assertIs(call.kwargs['keep_loaded'][0], loaded[2])
