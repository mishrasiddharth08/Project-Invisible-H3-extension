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
        refs: list = []
        with gr.Accordion('H3 · Still images', open=False, visible=forge.selected(), elem_id=f'pi_h3_{mode}_panel', elem_classes=['pi-h3-panel']) as panel:
            gr.Markdown('Best: ER SDE / Simple / CFG 1 / 50 steps · multiples of 32 · ≥3 MP recommended'
                        + (' · Denoise = 1 · use `<Picture 1>` for edits' if is_img2img else ' · editing via img2img'))
            with gr.Tabs(elem_id=f'pi_h3_{mode}_tabs'):
                with gr.Tab('Models'):
                    with gr.Row():
                        dit = gr.Dropdown(['Auto'] + inventory['dit'], value='Auto', label='Model', elem_id=f'pi_h3_{mode}_dit', scale=2)
                        vae = gr.Dropdown(['Auto'] + inventory['vae'], value='Auto', label='VAE', elem_id=f'pi_h3_{mode}_vae', scale=1)
                    clip = gr.Dropdown(['Auto'] + inventory['clip'], value='Auto', label='Text encoder', elem_id=f'pi_h3_{mode}_clip')
                    with gr.Row():
                        lora = gr.Dropdown(['(none)'] + inventory['lora'], value='(none)', label='LoRA (optional)', elem_id=f'pi_h3_{mode}_lora', scale=2)
                        strength = gr.Slider(-2, 2, value=0.38, step=0.01, label='Strength', elem_id=f'pi_h3_{mode}_strength', scale=1)
                with gr.Tab('Memory'):
                    with gr.Row():
                        memory = gr.Dropdown(['auto', 'lowvram', 'cpu'], value='auto', label='Memory mode', elem_id=f'pi_h3_{mode}_memory')
                        keep = gr.Checkbox(False, label='Keep model in memory', elem_id=f'pi_h3_{mode}_keep')
                    drift = gr.Checkbox(is_img2img, label='Pixel-drift fix: realign edits to the source image', elem_id=f'pi_h3_{mode}_drift')
                with gr.Tab('Files'):
                    gr.Markdown('[Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main) · place under `models/MiniMax-H3/` · **Generate never downloads**')
                    refresh = gr.Button('Refresh local files', size='sm', elem_id=f'pi_h3_{mode}_refresh')
                    def refresh_files():
                        found = scan()
                        return [gr.update(choices=['Auto']+found[k], value='Auto') for k in ('dit','clip','vae')] + [gr.update(choices=['(none)']+found['lora'], value='(none)')]
                    refresh.click(refresh_files, outputs=[dit, clip, vae, lora])
                    with gr.Row():
                        files = gr.CheckboxGroup(list(FILES), value=[], label='Automatic download (large files)', elem_id=f'pi_h3_{mode}_download_files')
                    with gr.Row():
                        consent = gr.Checkbox(False, label='I approve downloading and have reviewed the model license', elem_id=f'pi_h3_{mode}_download_consent')
                        button = gr.Button('Download selected', elem_id=f'pi_h3_{mode}_download')
                    status = gr.Textbox(label='Download status', interactive=False, elem_id=f'pi_h3_{mode}_download_status')
                    button.click(download, inputs=[files, consent], outputs=[status])
                if is_img2img:
                    with gr.Tab('References'):
                        with gr.Row():
                            refs.extend(gr.Image(type='pil', label=f'Picture {i}', elem_id=f'pi_h3_i2i_ref_{i}') for i in range(2, 6))
                        with gr.Row():
                            refs.extend(gr.Image(type='pil', label=f'Picture {i}', elem_id=f'pi_h3_i2i_ref_{i}') for i in range(6, 10))
        forge.register_ui_binding(panel, is_img2img)
        return [dit, clip, vae, lora, strength, memory, keep, drift, *refs]


try:
    forge.install()
    preset.install()
except Exception as error:
    print('[PI-H3] Integration unavailable:', error)
    raise
