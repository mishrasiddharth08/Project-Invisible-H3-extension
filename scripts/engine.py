"""Only native Forge selectors and conditional accordions; no extra page."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gradio as gr
from modules import scripts, script_callbacks
from pi_h3 import forge, preset
from pi_h3.assets import scan
from pi_h3.downloads import FILES, download
from pi_h3.config import LABEL

class Script(scripts.Script):
    _pi_h3 = True

    def title(self):
        return 'Project Invisible H3'

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        inventory = scan()
        mode = 'i2i' if is_img2img else 't2i'
        with gr.Accordion('H3 · Still images', open=False, visible=forge.selected(), elem_id=f'pi_h3_{mode}_panel', elem_classes=['pi-h3-panel']) as panel:
            gr.Markdown('Use **Generate** above. Start with **ER SDE / Simple / CFG 1 / 50 steps**. Width and height: multiples of 32. '
                        + ('Set **Denoising strength to 1**. Describe edits using `<Picture 1>`.' if is_img2img else 'For editing, use **img2img**.'))
            with gr.Accordion('Models · manual download recommended', open=False):
                gr.Markdown('Get the H3 model, Qwen3-VL-32B encoder and **video VAE** from '
                    '[Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main). '
                    'Place them under `models/MiniMax-H3/`. Existing shared model folders are also scanned. '
                    '**Generate never downloads models.**')
                dit = gr.Dropdown(['Auto'] + inventory['dit'], value='Auto', label='H3 checkpoint', elem_id=f'pi_h3_{mode}_dit')
                clip = gr.Dropdown(['Auto'] + inventory['clip'], value='Auto', label='H3 text encoder', elem_id=f'pi_h3_{mode}_clip')
                vae = gr.Dropdown(['Auto'] + inventory['vae'], value='Auto', label='H3 video VAE', elem_id=f'pi_h3_{mode}_vae')
                lora = gr.Dropdown(['(none)'] + inventory['lora'], value='(none)', label='Optional H3 LoRA', elem_id=f'pi_h3_{mode}_lora')
                refresh = gr.Button('Refresh local files', size='sm', elem_id=f'pi_h3_{mode}_refresh')
                def refresh_files():
                    found = scan()
                    return [gr.update(choices=['Auto']+found[k], value='Auto') for k in ('dit','clip','vae')] + [gr.update(choices=['(none)']+found['lora'], value='(none)')]
                refresh.click(refresh_files, outputs=[dit, clip, vae, lora])
                with gr.Accordion('Optional automatic download', open=False):
                    files = gr.CheckboxGroup(list(FILES), value=[], label='Choose files; these models are large', elem_id=f'pi_h3_{mode}_download_files')
                    consent = gr.Checkbox(False, label='I approve downloading the selected files and have reviewed the model license', elem_id=f'pi_h3_{mode}_download_consent')
                    button = gr.Button('Download selected models', elem_id=f'pi_h3_{mode}_download')
                    status = gr.Textbox(label='Download status', interactive=False, elem_id=f'pi_h3_{mode}_download_status')
                    button.click(download, inputs=[files, consent], outputs=[status])
            with gr.Accordion('Advanced', open=False):
                strength = gr.Slider(-2, 2, value=0.38, step=0.01, label='Selected LoRA strength', elem_id=f'pi_h3_{mode}_strength')
                memory = gr.Dropdown(['auto', 'lowvram', 'cpu'], value='auto', label='Memory mode', elem_id=f'pi_h3_{mode}_memory')
                keep = gr.Checkbox(False, label='Keep model in memory between runs (switching models always releases it)', elem_id=f'pi_h3_{mode}_keep')
                gr.Markdown('Previews show a quick approximation from real sampling data. Final images use the full Fizgig decoder. '
                            'Masks, Hires fix, tiling and video/audio output are not supported by this still extension.')
            refs = []
            if is_img2img:
                with gr.Accordion('Extra reference images · optional', open=False):
                    refs = [gr.Image(type='pil', label=f'Picture {i}', elem_id=f'pi_h3_i2i_ref_{i}') for i in range(2, 10)]
        forge.register_ui_binding(panel, is_img2img)
        return [dit, clip, vae, lora, strength, memory, keep, *refs]


try:
    forge.install()
    preset.install()
except Exception as error:
    print('[PI-H3] Integration unavailable:', error)
    raise
