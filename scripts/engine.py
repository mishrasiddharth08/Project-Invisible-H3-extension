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
        tab = 'img2img' if is_img2img else 'txt2img'
        refs: list = []
        # Preset + Autolink resolve checkpoint/TE/VAE; LoRA works via <lora:name:strength>
        # prompt tags. All four stay as hidden components to preserve the script_args contract.
        dit = gr.Dropdown(['Auto'] + inventory['dit'], value='Auto', label='H3 checkpoint', visible=False, elem_id=f'pi_h3_{mode}_dit')
        clip = gr.Dropdown(['Auto'] + inventory['clip'], value='Auto', label='H3 text encoder', visible=False, elem_id=f'pi_h3_{mode}_clip')
        vae = gr.Dropdown(['Auto'] + inventory['vae'], value='Auto', label='H3 VAE', visible=False, elem_id=f'pi_h3_{mode}_vae')
        lora = gr.Dropdown(['(none)'] + inventory['lora'], value='(none)', label='H3 LoRA', visible=False, elem_id=f'pi_h3_{mode}_lora')
        strength = gr.Slider(-2, 2, value=0.38, step=0.01, label='LoRA strength', visible=False, elem_id=f'pi_h3_{mode}_strength')
        def refresh_files():
            found = scan()
            return [gr.update(choices=['Auto']+found[k], value='Auto') for k in ('dit','clip','vae')] + [gr.update(choices=['(none)']+found['lora'], value='(none)')]
        with gr.Accordion('H3', open=False, visible=forge.ui_active(), elem_id=f'pi_h3_{mode}_panel', elem_classes=['pi-h3-panel']) as panel:
            output = gr.Radio(
                ['Still image', 'Video'], value=forge.output_mode(), label='Output',
                elem_id=f'{tab}_h3_output',
            )
            gr.Markdown('Choose **Still image** or **Video** here. H3 selects the matching local engine and safe defaults.'
                        + (' · Denoise = 1 for H3 image/video conditioning.' if is_img2img else '')
                        + ' · LoRA via `<lora:name:strength>`')
            with gr.Tabs(elem_id=f'pi_h3_{mode}_tabs'):
                with gr.Tab('Memory'):
                    with gr.Row():
                        memory = gr.Dropdown(['auto', 'lowvram', 'cpu'], value='auto', label='Memory mode', elem_id=f'pi_h3_{mode}_memory')
                        keep = gr.Checkbox(False, label='Keep model in memory', elem_id=f'pi_h3_{mode}_keep')
                    drift = gr.Checkbox(is_img2img, label='Pixel-drift fix: realign edits to the source image', elem_id=f'pi_h3_{mode}_drift')
                with gr.Tab('Files'):
                    gr.Markdown('[Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main) · place under `models/MiniMax-H3/` · **Generate never downloads**')
                    with gr.Row():
                        refresh = gr.Button('Refresh local files', size='sm', elem_id=f'pi_h3_{mode}_refresh')
                        button = gr.Button('Download selected', size='sm', elem_id=f'pi_h3_{mode}_download')
                    files = gr.CheckboxGroup(list(FILES), value=[], label='Automatic download (large files)', elem_id=f'pi_h3_{mode}_download_files')
                    consent = gr.Checkbox(False, label='I approve downloading and have reviewed the model license', elem_id=f'pi_h3_{mode}_download_consent')
                    status = gr.Textbox(label='Download status', interactive=False, elem_id=f'pi_h3_{mode}_download_status')
                    refresh.click(refresh_files, outputs=[dit, clip, vae, lora])
                    button.click(download, inputs=[files, consent], outputs=[status])
                if is_img2img:
                    with gr.Tab('References'):
                        with gr.Row():
                            refs.extend(gr.Image(type='pil', label=f'Picture {i}', elem_id=f'pi_h3_i2i_ref_{i}') for i in range(2, 6))
                        with gr.Row():
                            refs.extend(gr.Image(type='pil', label=f'Picture {i}', elem_id=f'pi_h3_i2i_ref_{i}') for i in range(6, 10))
        forge.register_ui_binding(panel, is_img2img)
        # Output is appended so every existing script/API argument keeps its position.
        return [dit, clip, vae, lora, strength, memory, keep, drift, *refs, output]


try:
    forge.install()
    preset.install()
except Exception as error:
    print('[PI-H3] Integration unavailable:', error)
    raise

from modules import script_callbacks
from pi_h3 import health
script_callbacks.on_app_started(health.app_started)
