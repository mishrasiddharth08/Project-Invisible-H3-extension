import base64
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pi_h3 import runtime


class Processing:
    class StableDiffusionProcessingImg2Img: pass
    @staticmethod
    def fix_seed(p): pass
    @staticmethod
    def Processed(p, images, **kwargs):
        return types.SimpleNamespace(images=images,**kwargs)


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.folder=Path(self.temporary.name)
        self.requests=[]
        self.closed=[]
        owner=self
        class Worker:
            def __init__(self,mode):
                self.mode=mode
                self.process=types.SimpleNamespace(poll=lambda:None)
            def wait_ready(self,cancelled): return {'type':'ready'}
            def generate(self,request,cancelled,update):
                owner.requests.append({**request,'reference_pixels':[Image.open(p).getpixel((0,0)) for p in request['references']]})
                update({'type':'progress','step':request['steps'],'steps':request['steps']})
                Image.new('RGB',(request['width'],request['height']),'red').save(request['output'])
                return request['output']
            def close(self): owner.closed.append(self)
        self.state=types.SimpleNamespace(interrupted=False,skipped=False,assign_current_image=Mock())
        self.shared=types.SimpleNamespace(state=self.state,opts=types.SimpleNamespace(live_previews_enable=True,samples_save=True,samples_format='png'),
            total_tqdm=Mock(),prompt_styles=types.SimpleNamespace(apply_styles_to_prompt=lambda p,s:p+' styled',apply_negative_styles_to_prompt=lambda p,s:p+' negative-style'))
        def save_image(image,path,basename,seed,prompt,extension,info,p):
            file=Path(path)/f'{seed}.{extension}'
            image.save(file)
            return str(file),None
        self.saver=Mock(side_effect=save_image)
        modules=types.ModuleType('modules')
        modules.shared=self.shared;modules.processing=Processing;modules.images=types.SimpleNamespace(save_image=self.saver)
        backend=types.ModuleType('backend')
        backend.memory_management=types.SimpleNamespace(unload_all_models=Mock())
        self.patches=[patch.dict(sys.modules,{'modules':modules,'backend':backend}),patch.object(runtime,'Worker',Worker),
            patch.object(runtime,'scan',return_value={'lora':[]}),patch.object(runtime,'resolve',side_effect=lambda x,k,i:str(self.folder/(k+'.safetensors')))]
        for p in self.patches: p.start()
        runtime._worker=None
        runtime._selection_cancel.clear()

    def tearDown(self):
        runtime.release()
        for p in reversed(self.patches): p.stop()
        self.temporary.cleanup()

    def request(self,editing=False,**kwargs):
        p=Processing.StableDiffusionProcessingImg2Img() if editing else types.SimpleNamespace()
        values=dict(prompt='teapot',negative_prompt='',styles=[],width=512,height=512,steps=4,cfg_scale=1,
            sampler_name='ER SDE',scheduler='Simple',seed=10,batch_size=1,n_iter=1,subseed=0,
            denoising_strength=1,init_images=[],extra_generation_params={},outpath_samples=str(self.folder))
        values.update(kwargs)
        for key,value in values.items(): setattr(p,key,value)
        return p

    def test_batch_seeds_and_saved_gallery_pixels(self):
        result=runtime.generate(self.request(batch_size=2,n_iter=2),{})
        self.assertEqual(result.all_seeds,[10,11,12,13])
        self.assertEqual(len(result.images),4)
        for seed,image in zip(result.all_seeds,result.images):
            with Image.open(self.folder/f'{seed}.png') as saved:
                self.assertEqual(saved.tobytes(),image.tobytes())
        self.assertIsNone(runtime._worker)

    def test_no_save_request(self):
        result=runtime.generate(self.request(do_not_save_samples=True),{})
        self.assertEqual(len(result.images),1)
        self.saver.assert_not_called()

    def test_edit_primary_then_extra_and_prompt_labels(self):
        p=self.request(True,init_images=[Image.new('RGB',(32,32),'red')])
        runtime.generate(p,{'refs':[Image.new('RGB',(32,32),'blue')]})
        self.assertEqual(self.requests[0]['reference_pixels'],[(255,0,0),(0,0,255)])
        self.assertEqual(self.requests[0]['prompt'],'<Picture 1> teapot')

    def test_txt2img_never_uses_stale_edit_references(self):
        p=self.request(init_images=[Image.new('RGB',(32,32),'red')])
        runtime.generate(p,{'refs':[Image.new('RGB',(32,32),'blue')]})
        self.assertEqual(self.requests[0]['references'],[])

    def test_keep_loaded_then_selection_releases(self):
        runtime.generate(self.request(),{'keep_loaded':True})
        self.assertIsNotNone(runtime._worker)
        runtime.selection_changed()
        self.assertIsNone(runtime._worker)
        self.assertEqual(len(self.closed),1)

    def test_native_styles_applied(self):
        runtime.generate(self.request(styles=['test']),{})
        self.assertEqual(self.requests[0]['prompt'],'teapot styled')

    def test_save_failure_is_not_success(self):
        self.saver.side_effect=lambda *a,**k:('does-not-exist',None)
        with self.assertRaisesRegex(RuntimeError,'could not save'):
            runtime.generate(self.request(),{})
        self.assertIsNone(runtime._worker)

    def test_explicit_unsupported_controls_fail_before_worker(self):
        for control in ('restore_faces','tiling'):
            with self.subTest(control=control),self.assertRaises(ValueError):
                runtime.generate(self.request(**{control:True}),{})
        self.assertEqual(self.requests,[])

if __name__=='__main__': unittest.main()
