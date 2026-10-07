from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch
from enum import Enum
from pi_h3 import memory_policy as m


class MemoryPolicyTests(TestCase):
    def test_all_six_physical_tiers_and_free_memory_pressure(self):
        for total in (8,10,12,16,24,32):
            with self.subTest(total=total):
                flags=m.worker_flags('auto',total,total-1)
                self.assertNotIn('--lowvram',flags)
                self.assertNotIn('--novram',flags)
                self.assertIn('--reserve-vram',flags)
                pressure=m.worker_flags('auto',total,4)
                self.assertGreater(float(pressure[-1]),float(flags[-1]))
        self.assertIn('--lowvram',m.worker_flags('lowvram',32,30))

    def test_invalid_or_unknown_memory_cannot_force_full_residency(self):
        self.assertEqual(m.worker_flags('auto'),['--novram'])
        self.assertEqual(m.worker_flags('cpu'),['--cpu'])
        with self.assertRaises(ValueError): m.budget_gib(float('nan'))
        with self.assertRaises(ValueError): m.validate('7')
        self.assertEqual(m.budget_gib(8,'32'),8)

    def test_explicit_budget_reserves_unused_capacity(self):
        flags=m.worker_flags('auto',32,31,'8')
        self.assertNotIn('--lowvram',flags)
        self.assertAlmostEqual(float(flags[-1]),25.2)
        for budget,cap in [(8,1),(10,1),(12,1),(16,1),(24,2),(32,4)]:
            self.assertEqual(m.tile_batch_limit(budget),cap)

    def test_native_restores_globals_and_preserves_dimensions_on_failure(self):
        class State(Enum):
            LOW_VRAM=1
            NORMAL_VRAM=2
            HIGH_VRAM=3
        mm=SimpleNamespace(SETTING_RESERVED_VRAM=-1,vram_state=State.HIGH_VRAM,
                           set_vram_to=State.HIGH_VRAM,VRAMState=State)
        p=SimpleNamespace(_pi_h3_native_memory=True,width=1152,height=768,batch_size=124)
        with patch.dict('sys.modules',{'backend':SimpleNamespace(memory_management=mm)}), \
             patch.object(m,'selected',return_value='8'),patch.object(m,'gpu_memory',return_value=(32,31)):
            with self.assertRaisesRegex(RuntimeError,'test failure'):
                with m.native_scope(p):
                    self.assertEqual(mm.vram_state,State.NORMAL_VRAM)
                    self.assertEqual(m.active_tile_limit(),1)
                    self.assertAlmostEqual(mm.SETTING_RESERVED_VRAM/m.GIB,25.2,places=5)
                    self.assertEqual((p.width,p.height,p.batch_size),(1152,768,124))
                    raise RuntimeError('test failure')
        self.assertEqual((mm.SETTING_RESERVED_VRAM,mm.vram_state,mm.set_vram_to),(-1,State.HIGH_VRAM,State.HIGH_VRAM))
        self.assertEqual(m.active_tile_limit(),4)

    def test_ordinary_request_does_not_touch_backend(self):
        with patch.object(m,'gpu_memory',side_effect=AssertionError('ordinary engine touched')):
            with m.native_scope(SimpleNamespace()): pass
