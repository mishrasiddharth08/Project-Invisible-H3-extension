"""Small H3 accordion and native-control events for Forge's Gradio interface."""

import html
import logging
from pathlib import Path

import gradio as gr

from .contracts import AUDIO_SHIFT, DEFAULT_FRAMES, FPS, H3Error
from .integration import checkpoint_info
from .models import ROLE_LABELS, inspect_model
from .ui_state import frame_view, preset_frame_view

COMPONENTS = {}
NATIVE_CONTROLS = ("batch_size", "batch_count", "sampling", "scheduler", "cfg_scale", "steps", "width", "height", "distilled_cfg_scale")
PANELS = []
logger = logging.getLogger("forge_h3")

OUTPUT_DEFAULTS = {
    "Still image": (1, 1, "ER SDE", "Simple", 1.0, 50, 1536, 1536, 12.0),
    "Video": (DEFAULT_FRAMES, 1, "Res Multistep", "Simple", 1.0, 20, 1152, 768, 12.0),
}


def output_selection(output):
    """Return the checkpoint, modules and native control values for the primary H3 output choice."""
    if output not in OUTPUT_DEFAULTS:
        raise H3Error("H3 output must be Still image or Video.")
    if output == "Still image":
        from pi_h3.config import LABEL
        return LABEL, [], OUTPUT_DEFAULTS[output]
    from pi_h3.preset import native_defaults
    checkpoint, modules = native_defaults()
    if not checkpoint or len(modules) != 3:
        raise H3Error("H3 Video needs the physical H3 checkpoint, a compatible text encoder, video VAE and audio VAE.")
    return checkpoint, [Path(path).name for path in modules], OUTPUT_DEFAULTS[output]


def remember_output(output):
    from modules import shared
    from pi_h3.config import OUTPUT_KEY
    if output not in OUTPUT_DEFAULTS:
        raise H3Error('H3 output must be Still image or Video.')
    shared.opts.set(OUTPUT_KEY, output)
    shared.opts.save(shared.config_filename)


def _in_blocks():
    # Forge also creates components outside the UI build (on page load); events can only be bound inside it
    return gr.context.Context.root_block is not None


def capture(component, **kwargs):
    elem_id = getattr(component, "elem_id", None)
    if elem_id:
        COMPONENTS[elem_id] = component
        if elem_id in ("txt2img_batch_size", "img2img_batch_size"):
            key = elem_id.replace("batch_size", "h3_duration")
            if key not in COMPONENTS:
                COMPONENTS[key] = gr.Markdown(value="", visible=False, elem_id=key)
        if elem_id in ("setting_sd_model_checkpoint", "setting_sd_modules", "forge_ui_preset"):
            bind_all(elem_id)


class Panel:
    def __init__(self, is_img2img):
        self.tab = "img2img" if is_img2img else "txt2img"
        self.is_img2img = is_img2img
        self.saved = gr.State({"active": False})
        with gr.Accordion("MiniMax H3", open=False, visible=False, elem_id=f"{self.tab}_h3_panel") as self.accordion:
            # Keep the native script's first argument for API compatibility. The visible,
            # authoritative selector lives in scripts/engine.py.
            self.output = gr.State("Video")
            self.audio = gr.Checkbox(value=True, label="Include generated audio", elem_id=f"{self.tab}_h3_audio")
            # the audio stream's own flow shift; Shift (the h3 preset slider) is the video one
            self.audio_shift = gr.Slider(minimum=1.0, maximum=20.0, step=0.5, value=AUDIO_SHIFT, label="Audio shift",
                                         elem_id=f"{self.tab}_h3_audio_shift")
            if is_img2img:
                gr.Markdown("With an FL2VA checkpoint the input image is the **first frame**; for a **last frame** too, add "
                            "one image to the **ImageStitch Integrated** gallery. With a **Ref2VA** checkpoint the input "
                            "image is `<Picture 1>` and the gallery holds the next reference pictures, up to 9 in all. "
                            "Denoising strength is not used.")
            else:
                gr.Markdown("With an FL2VA checkpoint, one image in the **ImageStitch Integrated** gallery is the **last "
                            "frame** (img2img gives the first). With a **Ref2VA** checkpoint the gallery holds up to 9 "
                            "reference pictures, `<Picture 1>`, `<Picture 2>`... in order.")
            self.status = gr.Markdown("Select the H3 text encoder, video VAE and audio VAE in VAE / Text Encoder.")
            with gr.Accordion("Components", open=False):
                self.summary = gr.Markdown("")
        self.bound = False
        self.attempts = []
        PANELS.append(self)

    @property
    def inputs(self):
        return [self.output, self.audio, self.audio_shift]

    @property
    def needed(self):
        ids = [f"{self.tab}_{name}" for name in NATIVE_CONTROLS]
        return ids + ["setting_sd_model_checkpoint", "setting_sd_modules", f"{self.tab}_h3_duration",
                      f"{self.tab}_h3_output"]

    def bind(self, trigger="?"):
        if self.bound:
            return
        if not _in_blocks():
            self.attempts.append(f"{trigger}: outside Blocks")
            return
        ids = [f"{self.tab}_{name}" for name in NATIVE_CONTROLS]
        needed = self.needed
        if any(name not in COMPONENTS for name in needed):
            self.attempts.append(f"{trigger}: waiting for {', '.join(n for n in needed if n not in COMPONENTS)}")
            logger.debug("Waiting for H3 native controls for %s.", self.tab)
            return
        native = [COMPONENTS[name] for name in ids]
        checkpoint = COMPONENTS["setting_sd_model_checkpoint"]
        modules = COMPONENTS["setting_sd_modules"]
        preset = COMPONENTS.get("forge_ui_preset")
        duration = COMPONENTS[f"{self.tab}_h3_duration"]
        primary_output = COMPONENTS[f"{self.tab}_h3_output"]
        defaults = [{key: getattr(c, key) for key in ("minimum", "maximum", "step", "label", "visible", "choices", "interactive")
                     if hasattr(c, key)} for c in native]
        preset_input = [preset] if preset is not None else []
        inputs = [checkpoint, primary_output, modules, self.saved] + native + preset_input
        outputs = [self.accordion, self.audio, self.audio_shift, duration, self.status, self.summary, self.saved] + native + preset_input

        def update(value, output, module_values, saved, *values):
            preset_value = values[-1] if preset is not None else None
            values = values[:len(native)]
            info = checkpoint_info(value, preset_value)
            error = ""
            active = False
            if info:
                try:
                    item = inspect_model(info.filename)
                    active = item is not None and item.role == "dit"
                except H3Error as exc:
                    error = str(exc)
            saved = dict(saved or {"active": False})
            entering = active and not saved.get("active")
            leaving = not active and saved.get("active")
            if entering:
                saved["native"] = [dict(config, value=value) for config, value in zip(defaults, values)]
                if preset_value is not None:
                    from modules_forge.presets import is_video
                    saved["native"][0].update(preset_frame_view(is_video(preset_value), values[0]))
            updates = [gr.update() for _ in native]
            frames = values[0]
            if active:
                frames = DEFAULT_FRAMES if entering else frames
                updates[0] = gr.update(**frame_view(True, output, frames))
                frames = frame_view(True, output, frames)["value"]
                updates[1] = gr.update(value=1, visible=False)
                if entering:
                    # the starting point of the reference workflows; every Forge sampler and schedule works
                    updates[2] = gr.update(value="Res Multistep")
                    updates[3] = gr.update(value="Simple")
                    updates[4] = gr.update(value=1.0)
                    updates[5] = gr.update(value=20)
                    updates[6] = gr.update(value=OUTPUT_DEFAULTS["Video"][6])
                    updates[7] = gr.update(value=OUTPUT_DEFAULTS["Video"][7])
                    updates[8] = gr.update(value=OUTPUT_DEFAULTS["Video"][8])
            elif leaving:
                updates = [gr.update(**config) for config in saved.get("native", [])]
                saved.pop("native", None)
            saved["active"] = active
            summary = ""
            if active:
                try:
                    from .integration import module_paths
                    from .models import resolve_components
                    components = resolve_components(info.filename, module_paths(module_values))
                    mode = ("**Mode:** Ref2VA, reference pictures" if components.mode == "ref2va"
                            else "**Mode:** FL2VA, first and last frame")
                    summary = "  \n".join([mode] + [f"**{ROLE_LABELS[m.role].title()}:** {html.escape(m.path.name)} ({m.quantization})"
                                                    for m in components.models])
                    error = ""
                except H3Error as exc:
                    error = str(exc)
            status = html.escape(error) if error else ("H3 generates audio jointly. This checkbox controls audio in the exported video." if output == "Video" else "Still image uses the first frame of a 5-frame H3 generation.")
            return [gr.update(visible=active), gr.update(visible=active and output == "Video"),
                    gr.update(visible=active and output == "Video"),
                    gr.update(value=f"{frames} frames / {FPS} FPS = {frames / FPS:.2f} seconds" if active else "",
                              visible=active and output == "Video"), status, summary, saved] + updates + (
                                  [gr.update()] if preset is not None else [])

        def select_output(output):
            checkpoint_value, module_values, values = output_selection(output)
            remember_output(output)
            return [gr.update(value=checkpoint_value), gr.update(value=module_values)] + [
                gr.update(value=value) for value in values
            ]

        primary_output.change(
            select_output, inputs=[primary_output], outputs=[checkpoint, modules] + native,
            queue=False, show_progress=False,
        )
        for event in (checkpoint.change, primary_output.change, modules.change):
            event(update, inputs=inputs, outputs=outputs, queue=False, show_progress=False)
        # Frame changes update duration without overwriting any other native controls.
        native[0].change(lambda n, output: gr.update(value=f"{int(n)} frames / {FPS} FPS = {int(n) / FPS:.2f} seconds"),
                         inputs=[native[0], primary_output], outputs=[duration], queue=False, show_progress=False)
        gr.context.Context.root_block.load(update, inputs=inputs, outputs=outputs, queue=False, show_progress=False)
        self.bound = True
        logger.info("H3 native controls connected for %s.", self.tab)


def bind_all(trigger="ui_tabs"):
    for panel in PANELS:
        panel.bind(trigger)
    return []


def check_bindings(*_args):
    # Forge also builds panel instances it never renders; a tab is fine as long as one of its panels is bound
    for tab in dict.fromkeys(panel.tab for panel in PANELS):
        panels = [panel for panel in PANELS if panel.tab == tab]
        if any(panel.bound for panel in panels):
            continue
        missing = sorted({name for panel in panels for name in panel.needed if name not in COMPONENTS})
        attempts = [attempt for panel in panels for attempt in panel.attempts]
        logger.error("H3 native controls were not found for %s (missing: %s; attempts: %s); restart Forge after "
                     "updating it.", tab, ", ".join(missing) or "none", "; ".join(attempts) or "none")


def reset():
    COMPONENTS.clear()
    PANELS.clear()
