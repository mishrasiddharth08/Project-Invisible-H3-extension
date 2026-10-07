import sys,threading,time,types,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pi_h3 import refresh
from functools import wraps

class RefreshTests(unittest.TestCase):
    def test_active_refreshes_serialize_complete_foreign_chain(self):
        running=0;peak=0;gate=threading.Barrier(3)
        def original():
            nonlocal running,peak
            running+=1;peak=max(peak,running);time.sleep(.02);running-=1
            return 'registered'
        models=types.SimpleNamespace(list_models=original)
        refresh.install(models,lambda:True);wrapped=models.list_models
        refresh.install(models,lambda:True);self.assertIs(wrapped,models.list_models)
        results=[]
        def run():
            gate.wait();results.append(models.list_models())
        threads=[threading.Thread(target=run) for _ in range(2)]
        for t in threads:t.start()
        gate.wait()
        for t in threads:t.join(2)
        self.assertEqual(peak,1);self.assertEqual(results,['registered']*2)
    def test_other_presets_delegate_arguments_and_result(self):
        sentinel=object();calls=[]
        def original(*args,**kwargs):calls.append((args,kwargs));return sentinel
        models=types.SimpleNamespace(list_models=original)
        refresh.install(models,lambda:False)
        self.assertIs(models.list_models(2,refresh=True),sentinel)
        self.assertEqual(calls,[((2,),{'refresh':True})])
    def test_foreign_wrappers_cannot_copy_the_guard_marker(self):
        models=types.SimpleNamespace(list_models=lambda: 'complete')
        refresh.install(models,lambda:True)
        inner=models.list_models
        @wraps(inner)
        def foreign():return inner()
        models.list_models=foreign
        refresh.install(models,lambda:True)
        self.assertIsNot(models.list_models,foreign)
        self.assertIs(models.list_models._pi_h3_refresh_lock,models.list_models)
        self.assertEqual(models.list_models(),'complete')
