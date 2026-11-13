"""Optional real Gradio 4.40 build; no web server, GPU or Forge installation."""

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from test_contracts import DIT, PH3_TE, PH3_VAE, TE, checkpoint


@unittest.skipUnless(importlib.util.find_spec("gradio"), "Gradio is optional for CPU tests")
class GradioSmoke(unittest.TestCase):
    def test_forge_selectors_created_after_generation_interfaces(self):
        import gradio as gr

        from pi_h3_image import ui
        ui.reset()
        self.addCleanup(ui.reset)
        interfaces = []
        for tab in ("txt2img", "img2img"):
            with gr.Blocks() as interface:
                native = [gr.Dropdown(["Euler"], value="Euler", elem_id=f"{tab}_sampling"),
                          gr.Dropdown(["Simple"], value="Simple", elem_id=f"{tab}_scheduler"),
                          gr.Slider(1, 24, value=1, elem_id=f"{tab}_cfg_scale")]
                for component in native:
                    ui.capture(component)
                ui.Panel(tab == "img2img")
            interfaces.append(interface)
        # Forge calls ui_tabs before the outer Blocks creates the model selectors.
        ui.bind_all()
        self.assertFalse(any(panel.bound for panel in ui.PANELS))
        with gr.Blocks() as demo:
            for interface in interfaces:
                interface.render()
            preset = gr.Dropdown(["sd"], value="sd", elem_id="forge_ui_preset")
            checkpoint_selector = gr.Dropdown(["H3"], value="H3", elem_id="setting_sd_model_checkpoint")
            module_selector = gr.Dropdown([], multiselect=True, elem_id="setting_sd_modules")
            for component in (preset, checkpoint_selector, module_selector):
                ui.capture(component)
            self.assertTrue(all(panel.bound for panel in ui.PANELS))
            handlers = len(demo.fns)
            ui.capture(module_selector)
            self.assertEqual(len(demo.fns), handlers)
        self.assertEqual(len({id(fn.fn) for fn in demo.fns.values()
                              if getattr(fn.fn, "__name__", "") == "update"}), 2)

    def test_native_image_dimensions_only_and_restore(self):
        import gradio as gr
        from pi_h3_image import ui
        from pi_h3_image.models import H3_IMAGE, PH3_IMAGE, H3
        ui.reset()
        self.addCleanup(ui.reset)
        with gr.Blocks() as demo:
            ui.capture(gr.Dropdown(["model"], elem_id="setting_sd_model_checkpoint"))
            for tab in ("txt2img", "img2img"):
                ui.Panel(tab == "img2img")
                ui.capture(gr.Slider(64, 2048, value=1024, step=8, elem_id=f"{tab}_width"))
                ui.capture(gr.Slider(64, 2048, value=1536, step=8, elem_id=f"{tab}_height"))
        self.assertTrue(all(p.dimension_bound for p in ui.PANELS))
        handlers = {id(f.fn): f.fn for f in demo.fns.values()
                    if getattr(f.fn, "__name__", "") == "dimensions_update"}
        self.assertEqual(len(handlers), 2)
        limits = [f.fn for f in demo.fns.values() if getattr(f.fn, "__name__", "") == "limit"]
        self.assertEqual(len(limits), 4)
        for limit in limits:
            inactive = limit({}, 8192, 8192)
            self.assertNotIn("value", inactive[0])
            for width, height in ((5472, 3040), (3040, 5472), (4096, 4096), (8192, 2048), (2048, 8192)):
                unchanged = limit({"active": True}, width, height)
                self.assertEqual([u["value"] for u in unchanged], [width, height])
            reduced = limit({"active": True}, 8192, 8192)
            self.assertLessEqual(reduced[0]["value"] * reduced[1]["value"], 16_777_216)
        for fn in handlers.values():
            with patch.object(ui, "checkpoint_info", return_value=types.SimpleNamespace(filename="model")):
                with patch.object(ui, "inspect_model", return_value=types.SimpleNamespace(variant=H3_IMAGE)):
                    entered = fn("model", {}, 1024, 1536)
                self.assertEqual(entered[1]["maximum"], 8192)
                self.assertEqual(entered[2]["step"], 32)
                for variant in (PH3_IMAGE, H3, "other", None):
                    item = None if variant is None else types.SimpleNamespace(variant=variant)
                    with patch.object(ui, "inspect_model", return_value=item):
                        unchanged = fn("model", {}, 1024, 1536)
                        restored = fn("model", entered[0], 4096, 4096)
                    self.assertNotIn("maximum", unchanged[1])
                    self.assertEqual(restored[1]["maximum"], 2048)
                    self.assertEqual(restored[2]["value"], 1536)

    def test_the_panel_has_no_generation_inputs(self):
        import gradio as gr

        from pi_h3_image import ui
        ui.reset()
        self.addCleanup(ui.reset)
        for is_img2img in (False, True):
            with gr.Blocks():
                panel = ui.Panel(is_img2img)
            self.assertEqual(panel.inputs, [])

    def test_real_blocks_build_and_model_switching(self):
        import gradio as gr

        from pi_h3_image import ui
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            dit = checkpoint(root / "p-h3-image_w4a8.safetensors", DIT)
            components = [checkpoint(root / (name + ".safetensors"), tensors)
                          for name, tensors in (("encoder", PH3_TE), ("vae", PH3_VAE))]
            info = types.SimpleNamespace(filename=str(dit), title="Community", name="Community")
            modules = types.ModuleType("modules")
            # the ph3 preset as the user saved it in Settings (issue #5); img2img left at its defaults
            modules.shared = types.SimpleNamespace(opts=types.SimpleNamespace(
                ph3_t2i_sampler="Euler", ph3_t2i_scheduler="Automatic", ph3_t2i_cfg=3.5))
            modules.paths = types.SimpleNamespace(models_path=str(root))
            modules.sd_models = types.SimpleNamespace(checkpoint_aliases={"Community": info}, checkpoints_list={"Community": info},
                                                      get_closet_checkpoint_match={"Community": info}.get)
            forge = types.ModuleType("modules_forge")
            forge.main_entry = types.SimpleNamespace(module_list={p.name: str(p) for p in components})
            with patch.dict(sys.modules, {"modules": modules, "modules_forge": forge}):
                ui.reset()
                with gr.Blocks() as demo:
                    ckpt = gr.Dropdown(["Regular", "Community"], value="Regular", elem_id="setting_sd_model_checkpoint")
                    mods = gr.Dropdown([p.name for p in components], value=[p.name for p in components], multiselect=True, elem_id="setting_sd_modules")
                    preset = gr.Dropdown(["sd", "ph3"], value="sd", elem_id="forge_ui_preset")
                    ui.capture(ckpt)
                    ui.capture(mods)
                    ui.capture(preset)
                    for tab in ("txt2img", "img2img"):
                        panel_native = [gr.Dropdown(["Euler", "DPM++ 2M"], value="DPM++ 2M", elem_id=f"{tab}_sampling"),
                                        gr.Dropdown(["Automatic", "Simple"], value="Automatic", elem_id=f"{tab}_scheduler"),
                                        gr.Slider(1, 24, value=7, elem_id=f"{tab}_cfg_scale")]
                        for component in panel_native:
                            ui.capture(component)
                        if tab == "txt2img":
                            # Forge's InputAccordion: an accordion and the hidden checkbox Forge reads
                            with gr.Accordion("Hires. fix", elem_id="txt2img_hr") as hires:
                                pass
                            ui.capture(hires)
                            ui.capture(gr.Checkbox(value=True, visible=False, elem_id="txt2img_hr-checkbox"))
                            # the Refiner, as Forge Neo's processing script names it
                            with gr.Accordion("Refiner", elem_id="txt2img_enable") as refiner:
                                pass
                            ui.capture(refiner)
                            ui.capture(gr.Checkbox(value=False, visible=False, elem_id="txt2img_enable-checkbox"))
                        ui.Panel(tab == "img2img")
                    self.assertEqual(ui.bind_all(), [])
                self.assertTrue(all(p.bound for p in ui.PANELS))
                self.assertTrue(demo.config["dependencies"])
                fn = next(f.fn for f in demo.fns.values() if getattr(f.fn, "__name__", "") == "update")
                # outputs: accordion, status, summary, state, sampler, scheduler, CFG, preset, then the hidden
                # accordions with their checkboxes
                enter = fn("Community", [p.name for p in components], {"active": False}, "DPM++ 2M", "Automatic", 7, "sd")
                self.assertTrue(enter[0]["visible"])
                self.assertIn("encoder.safetensors (plain)", enter[2])
                self.assertIn("**Model:** P-H3-Image", enter[2])
                self.assertIn("P-H3-Image makes one still image", enter[1])
                self.assertEqual([enter[i]["value"] for i in (4, 5, 6)], ["Euler", "Automatic", 3.5])
                from pi_h3_image.profiles import H3_IMAGE, PH3_IMAGE
                self.assertEqual(ui.preset_settings("img2img", PH3_IMAGE), ("Euler", "P-H3-Image", 1.0))
                self.assertEqual(ui.preset_settings("img2img"), ("H3-Image LCM", "Simple", 1.0))
                self.assertEqual(ui.preset_settings("txt2img", H3_IMAGE), ui.preset_settings("img2img"))
                modules.shared.opts.ph3_t2i_cfg = 0
                self.assertIsNone(ui.preset_settings("txt2img", PH3_IMAGE)[2])
                modules.shared.opts.ph3_t2i_cfg = 3.5
                # MiniMax H3 with its own text encoder is the video model: the status line says what to select
                video = fn("Community", [str(checkpoint(root / "h3_te.safetensors", TE)), components[1].name], enter[3],
                           "Euler", "Automatic", 3.5, "sd")
                self.assertIn("audio-video checkpoint makes still images with the P-H3-Image text encoder", video[1])
                # a missing module is reported in the status line
                partial = fn("Community", [components[0].name], enter[3], "Euler", "Automatic", 3.5, "sd")
                self.assertIn("Select the H3 VAE", partial[1])
                leave = fn("Regular", [], partial[3], "Euler", "Simple", 1, "sd")
                self.assertEqual(leave[4]["value"], "DPM++ 2M")
                self.assertEqual(leave[6]["value"], 7)
                self.assertFalse(leave[0]["visible"])
                # the UI preset stays free: switching it is how the user leaves H3, and "ph3" is a preset of its own
                self.assertNotIn("interactive", enter[7])
                self.assertNotIn("interactive", leave[7])
                # Hires. fix is hidden and turned off with H3, shown again after
                self.assertEqual((enter[8]["visible"], enter[9]["value"]), (False, False))
                self.assertEqual((enter[10]["visible"], enter[11]["value"]), (False, False))
                self.assertTrue(leave[8]["visible"] and leave[10]["visible"])
                self.assertNotIn("value", leave[9])
                ui.reset()


if __name__ == "__main__":
    unittest.main()
