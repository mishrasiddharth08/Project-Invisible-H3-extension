"""H3 Image status and native-control events for Forge's Gradio interface."""

import html
import logging

import gradio as gr

from .contracts import H3Error
from .integration import checkpoint_info
from .models import ROLE_LABELS, inspect_model, is_image_checkpoint
from .profiles import H3_IMAGE

COMPONENTS = {}
NATIVE_CONTROLS = ("sampling", "scheduler", "cfg_scale")
# H3 does not support Forge's built-in Hires. fix pass. The separate Refiner remains available.
HIDDEN_ACCORDIONS = ("hr",)
PANELS = []
logger = logging.getLogger("forge_h3")


def _in_blocks():
    # Forge also creates components outside the UI build (on page load); events can only be bound inside it
    return gr.context.Context.root_block is not None


def capture(component, **kwargs):
    elem_id = getattr(component, "elem_id", None)
    if elem_id:
        COMPONENTS[elem_id] = component
        if elem_id in ("setting_sd_model_checkpoint", "setting_sd_modules", "forge_ui_preset"):
            bind_all(elem_id)
        for panel in PANELS:
            panel.bind_dimensions()


class Panel:
    def __init__(self, is_img2img):
        self.tab = "img2img" if is_img2img else "txt2img"
        self.is_img2img = is_img2img
        self.saved = gr.State({"active": False})
        with gr.Group(visible=False, elem_id=f"{self.tab}_pi_h3_image_status") as self.accordion:
            choices = [("Use Settings", "settings"), ("Full size", "1"), ("50%", "1/2"),
                       ("33% Lanczos", "33% Lanczos")]
            with gr.Accordion("Downscale", open=False):
                self.output_scale = gr.Dropdown(choices=choices, value="settings", label="Output size",
                                                elem_id=f"{self.tab}_h3_output_scale")
            self.status = gr.Markdown("Select the H3 text encoder and VAE in VAE / Text Encoder.")
            if is_img2img:
                gr.Markdown("With **[H3] Enable Reference** on (Settings â†’ MiniMax H3), img2img is an **edit** with "
                            "MiniMax-H3-Image or P-H3-Image: the input image is `<Picture 1>`, the picture to edit; the "
                            "**ImageStitch Integrated** gallery holds the references, `<Picture 2>` onwards (16 pictures "
                            "in all for MiniMax-H3-Image, 8 for P-H3-Image). Name them in the prompt (\"the hat from "
                            "<Picture 2>\"). The picture is generated at Width x Height; Resize mode and Denoising "
                            "strength are not used. With the setting off, img2img is Forge's own: the input image is "
                            "noised at Denoising strength, with either model.")
            with gr.Accordion("Components", open=False):
                self.summary = gr.Markdown("")
        self.dimension_bound = False
        self.dimension_saved = gr.State({"active": False})
        self.bound = False
        self.attempts = []
        PANELS.append(self)

    def bind_dimensions(self):
        """Limit this override to native MiniMax-H3-Image; never change Forge globally."""
        if self.dimension_bound or not _in_blocks():
            return
        ids = [f"{self.tab}_{side}" for side in ("width", "height")]
        if any(key not in COMPONENTS for key in ids + ["setting_sd_model_checkpoint"]):
            return
        sliders = [COMPONENTS[key] for key in ids]
        defaults = [{key: getattr(c, key) for key in ("minimum", "maximum", "step")}
                    for c in sliders]
        checkpoint = COMPONENTS["setting_sd_model_checkpoint"]

        def dimensions_update(value, saved, width, height):
            from .dimensions import dimension_updates
            from .models import H3_IMAGE as IMAGE_VARIANT
            info = checkpoint_info(value)
            active = False
            if info:
                try:
                    item = inspect_model(info.filename)
                    active = item is not None and item.variant == IMAGE_VARIANT
                except H3Error:
                    pass
            state, updates = dimension_updates(active, saved, (width, height), defaults)
            return [state] + [gr.update(**update) for update in updates]

        inputs = [checkpoint, self.dimension_saved] + sliders
        outputs = [self.dimension_saved] + sliders
        checkpoint.change(dimensions_update, inputs=inputs, outputs=outputs, queue=False, show_progress=False)
        gr.context.Context.root_block.load(dimensions_update, inputs=inputs, outputs=outputs,
                                          queue=False, show_progress=False)
        self.dimension_bound = True

    @property
    def inputs(self):
        return [self.output_scale]

    @property
    def needed(self):
        ids = [f"{self.tab}_{name}" for name in NATIVE_CONTROLS]
        return ids + ["setting_sd_model_checkpoint", "setting_sd_modules"]

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
        defaults = [{key: getattr(c, key) for key in ("minimum", "maximum", "step", "label", "visible", "choices", "interactive")
                     if hasattr(c, key)} for c in native]
        preset_input = [preset] if preset is not None else []
        hidden = [(COMPONENTS[key], COMPONENTS[f"{key}-checkbox"]) for key in (f"{self.tab}_{name}" for name in HIDDEN_ACCORDIONS)
                  if key in COMPONENTS and f"{key}-checkbox" in COMPONENTS]
        hidden_defaults = [accordion.visible for accordion, _ in hidden]
        inputs = [checkpoint, modules, self.saved] + native + preset_input + [checkbox for _, checkbox in hidden]
        outputs = ([self.accordion, self.status, self.summary, self.saved] + native + preset_input
                   + [component for pair in hidden for component in pair])

        def update(value, module_values, saved, *values):
            hidden_values = values[len(native) + len(preset_input):]
            values = values[:len(native)]
            info = checkpoint_info(value)
            error = ""
            active = False
            if info:
                try:
                    item = inspect_model(info.filename)
                    active = is_image_checkpoint(item)
                except H3Error as exc:
                    error = str(exc)
            saved = dict(saved or {"active": False})
            entering = active and not saved.get("active")
            leaving = not active and saved.get("active")
            if entering:
                saved["native"] = [dict(config, value=value) for config, value in zip(defaults, values)]
                saved["hidden_values"] = hidden_values
            summary = ""
            profile = None
            if active:
                try:
                    from .integration import module_paths
                    from .models import resolve_components
                    components = resolve_components(info.filename, module_paths(module_values))
                    profile = components.profile
                    experimental = f" (experimental: {components.experimental})" if components.experimental else ""
                    summary = "  \n".join([f"**Model:** {profile.label}{experimental}"]
                                          + [f"**{ROLE_LABELS[m.role].title()}:** {html.escape(m.path.name)} ({m.quantization})"
                                             for m in components.models])
                    error = ""
                except H3Error as exc:
                    error = str(exc)
            updates = [gr.update() for _ in native]
            # the profile's recipe when its text encoder and VAE are selected, else the one saved last, on entering
            # and when the selected modules change it
            if active and profile is not None and (entering or saved.get("profile") != profile.name):
                sampler, scheduler, cfg = preset_settings(self.tab, profile)
                updates[0] = gr.update(value=sampler)
                updates[1] = gr.update(value=scheduler)
                updates[2] = gr.update(value=cfg) if cfg is not None else gr.update()
                saved["profile"] = profile.name
            elif leaving:
                updates = [gr.update(**config) for config in saved.get("native", [])]
                saved.pop("native", None)
                saved.pop("profile", None)
            saved["active"] = active
            accordion_updates = hidden_updates(hidden_defaults, active, leaving, saved.get("hidden_values", ()))
            if leaving:
                saved.pop("hidden_values", None)
            if error:
                status = html.escape(error)
            elif profile is H3_IMAGE:
                status = ""
            elif profile is not None:
                shift = f"Shift {profile.shift:g}, " if profile.shift_used else ""
                status = (f"{profile.label} makes one still image per generation: {profile.sampler}, {profile.scheduler}, "
                          f"{profile.steps} steps, CFG {profile.cfg:g}, {shift}best at about {profile.size}x{profile.size} "
                          f"(the {profile.name} UI preset).")
            else:
                status = "H3 makes one still image per generation."
            return ([gr.update(visible=active), status, summary, saved] + updates
                    + ([gr.update()] if preset is not None else []) + accordion_updates)

        for event in (checkpoint.change, modules.change):
            event(update, inputs=inputs, outputs=outputs, queue=False, show_progress=False)
        gr.context.Context.root_block.load(update, inputs=inputs, outputs=outputs, queue=False, show_progress=False)
        self.bound = True
        logger.info("H3 native controls connected for %s.", self.tab)


def preset_settings(tab, profile=H3_IMAGE):
    """The sampler, schedule and CFG saved for the profile's preset in Settings, which start from the profile's recipe
    (every Forge sampler and schedule works); a CFG of 0 there keeps the current one (None), as Forge's presets do."""
    kind = "i2i" if tab == "img2img" else "t2i"
    try:
        from modules import shared
        opts = shared.opts
    except (ImportError, AttributeError):
        opts = None
    sampler = getattr(opts, f"{profile.name}_{kind}_sampler", None) or profile.sampler
    scheduler = getattr(opts, f"{profile.name}_{kind}_scheduler", None) or profile.scheduler
    cfg = getattr(opts, f"{profile.name}_{kind}_cfg", None)
    if cfg is None:
        cfg = profile.cfg
    return sampler, scheduler, cfg if cfg > 0 else None


def hidden_updates(visibility, active, leaving, saved_values):
    """The accordion and its separate checkbox (which Forge reads as the generation input)."""
    if active:
        return [gr.update(visible=False, open=False), gr.update(value=False)] * len(visibility)
    if leaving:
        updates = []
        for index, visible in enumerate(visibility):
            if index < len(saved_values):
                checked = saved_values[index]
                updates.extend((gr.update(visible=visible, open=checked), gr.update(value=checked)))
            else:
                updates.extend((gr.update(visible=visible), gr.update()))
        return updates
    return [gr.update(), gr.update()] * len(visibility)


def bind_all(trigger="ui_tabs"):
    for panel in PANELS:
        panel.bind(trigger)
        panel.bind_dimensions()
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
