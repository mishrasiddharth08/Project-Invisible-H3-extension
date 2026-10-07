from pathlib import Path
import threading
from .config import (ROOT, LABEL, PRESET, LEGACY_VIDEO_PRESET, OUTPUT_KEY,
                     OUTPUTS, STILL_OUTPUT, VIDEO_OUTPUT)
from . import runtime
from .assets import classify, scan

KEYS = ('dit', 'clip', 'vae', 'lora', 'strength', 'memory', 'keep_loaded', 'drift')
_UI_BINDINGS = []
_selection_state = threading.local()


def _normalize_output(value):
    aliases = {'image': STILL_OUTPUT, 'still': STILL_OUTPUT, 'still image': STILL_OUTPUT,
               'video': VIDEO_OUTPUT}
    return aliases.get(str(value or '').strip().lower())


def effective_preset(p=None):
    from modules import shared
    overrides = getattr(p, 'override_settings', {}) or {}
    preset = overrides.get('forge_preset', getattr(shared.opts, 'forge_preset', None))
    return PRESET if preset == LEGACY_VIDEO_PRESET else preset


def output_mode(p=None, runner=None, with_source=False):
    from modules import shared
    overrides = getattr(p, 'override_settings', {}) or {}
    explicit = _normalize_output(overrides.get(OUTPUT_KEY))
    if explicit:
        return (explicit, 'request override') if with_source else explicit
    runner = runner or getattr(p, 'scripts', None)
    for script in getattr(runner, 'alwayson_scripts', []) or []:
        if not getattr(script, '_pi_h3', False):
            continue
        values = list(getattr(p, 'script_args', []) or [])[script.args_from:script.args_to]
        scripted = _normalize_output(values[-1] if values else None)
        if scripted:
            return (scripted, 'primary H3 control') if with_source else scripted
    legacy = (overrides.get('forge_preset') == LEGACY_VIDEO_PRESET
              or (not overrides and getattr(shared.opts, 'forge_preset', None) == LEGACY_VIDEO_PRESET))
    value = VIDEO_OUTPUT if legacy else _normalize_output(getattr(shared.opts, OUTPUT_KEY, STILL_OUTPUT)) or STILL_OUTPUT
    source = 'legacy API preset' if legacy else 'settings'
    return (value, source) if with_source else value


def selected(p=None, runner=None):
    from modules import shared, sd_models
    overrides = getattr(p, 'override_settings', {}) or {}
    value = overrides.get('sd_model_checkpoint', shared.opts.sd_model_checkpoint)
    return selected_value(value, effective_preset(p), output_mode(p, runner))


def selected_value(value, preset=None, output=None):
    from modules import sd_models
    if preset is None:
        try:
            from modules import shared
            preset = getattr(shared.opts, 'forge_preset', None)
        except ImportError:
            pass
    if preset == LEGACY_VIDEO_PRESET:
        preset = PRESET
    if output is None:
        output = output_mode()
    if preset != PRESET or output == VIDEO_OUTPUT:
        return False
    item = sd_models.checkpoints_list.get(value) or sd_models.checkpoint_aliases.get(value)
    synthetic = value == LABEL or bool(getattr(item, '_pi_h3', False))
    if synthetic:
        return True
    filename = getattr(item, 'filename', value or '')
    physical = classify(filename) == 'dit'
    return physical and Path(filename).suffix.lower() != '.gguf'


def _normalize_request(p, runner=None):
    from modules import shared
    overrides = dict(getattr(p, 'override_settings', {}) or {})
    if overrides.get('forge_preset') == LEGACY_VIDEO_PRESET:
        overrides['forge_preset'] = PRESET
        overrides.setdefault(OUTPUT_KEY, VIDEO_OUTPUT)
    p.override_settings = overrides
    if effective_preset(p) == PRESET and output_mode(p, runner) == VIDEO_OUTPUT:
        from . import preset
        checkpoint, modules = preset.native_defaults()
        value = overrides.get('sd_model_checkpoint', shared.opts.sd_model_checkpoint)
        from modules import sd_models
        item = sd_models.checkpoints_list.get(value) or sd_models.checkpoint_aliases.get(value)
        if (value == LABEL or getattr(item, '_pi_h3', False)) and checkpoint:
            overrides['sd_model_checkpoint'] = checkpoint
        if (('forge_additional_modules' in overrides and not overrides['forge_additional_modules'])
                or ('forge_additional_modules' not in overrides
                    and not preset.native_modules_valid(getattr(shared.opts, 'forge_additional_modules', []) or []))):
            overrides['forge_additional_modules'] = modules


def _checkpoint_identity(value):
    from modules import sd_models
    item = sd_models.checkpoints_list.get(value) or sd_models.checkpoint_aliases.get(value)
    filename = getattr(item, 'filename', None)
    return str(Path(filename).resolve()).casefold() if filename else str(value or '').casefold()


def _same_checkpoint(left, right):
    return _checkpoint_identity(left) == _checkpoint_identity(right)


def _cancel(reason):
    if runtime._worker is not None:
        print(f'[PI-H3] cancelling dedicated worker: {reason}')
    runtime.selection_changed()


def register():
    from modules import sd_models
    if LABEL in sd_models.checkpoints_list:
        return
    ci = object.__new__(sd_models.CheckpointInfo)
    # A real H3 checkpoint keeps strict checkpoint managers from trying to
    # fingerprint the metadata-only fallback marker.
    try:
        inventory = scan()
    except ImportError:
        inventory = {'dit': []}
    ci.filename = inventory['dit'][0] if inventory['dit'] else str(ROOT / 'resources' / 'h3.json')
    ci.name = ci.title = ci.short_title = ci.model_name = ci.name_for_extra = LABEL
    ci.hash = 'pi-h3'
    ci.sha256 = ci.shorthash = None
    ci.metadata = {}
    ci.is_safetensors = False
    ci._pi_h3 = True
    # Keep the real model path free for H3's native Video route.
    ci.ids = [LABEL]
    ci.calculate_shorthash = lambda: None
    ci.register()


def options(runner, p):
    result = {}
    for script in getattr(runner, 'alwayson_scripts', []) or []:
        if getattr(script, '_pi_h3', False):
            values = list(getattr(p, 'script_args', []) or [])[script.args_from:script.args_to]
            result = {**dict(zip(KEYS, values[:len(KEYS)])), 'refs': values[len(KEYS):len(KEYS)+8]}
            break
    from modules import shared, sd_models
    value = (getattr(p, 'override_settings', {}) or {}).get('sd_model_checkpoint', shared.opts.sd_model_checkpoint)
    item = sd_models.checkpoints_list.get(value) or sd_models.checkpoint_aliases.get(value)
    filename = getattr(item, 'filename', '')
    if not getattr(item, '_pi_h3', False) and classify(filename) == 'dit':
        result['dit'] = filename
    # Native "VAE / Text Encoder" multi-dropdown wins over the hidden extension selectors.
    for module in getattr(shared.opts, 'forge_additional_modules', []) or []:
        role = classify(str(module))
        if role in ('clip', 'vae'):
            result[role] = str(module)
    return result


def install_selection():
    from modules import shared
    for key in ('sd_model_checkpoint', 'forge_preset'):
        item = shared.opts.data_labels.get(key)
        if item is None or getattr(item.onchange, '_pi_h3', False):
            continue
        previous = item.onchange
        def changed(*args, previous=previous, key=key, **kwargs):
            if (key == 'sd_model_checkpoint' and getattr(shared.opts, 'forge_preset', None) == PRESET
                    and not selected() and (getattr(_selection_state, 'h3_callback', False)
                                            or getattr(_selection_state, 'h3_refresh', False))
                    and not getattr(_selection_state, 'explicit_leave', False)):
                # Some older extension callbacks repair their own saved model
                # while Forge has temporarily cleared the checkpoint registry.
                # H3's preset remains authoritative during that refresh.
                shared.opts.set('sd_model_checkpoint', LABEL, run_callbacks=False)
                shared.opts.set('forge_checkpoint_' + PRESET, LABEL, run_callbacks=False)
                print('[PI-H3] ignored transient foreign checkpoint restore while preset H3 is active')
                return None
            if ((key == 'forge_preset' and shared.opts.forge_preset != PRESET)
                    or (key == 'sd_model_checkpoint' and not selected()
                        and not getattr(_selection_state, 'explicit_leave', False))):
                _cancel(f'option changed: {key}')
            if previous:
                nested_guard = (key == 'sd_model_checkpoint'
                                and getattr(shared.opts, 'forge_preset', None) == PRESET
                                and selected())
                old_guard = getattr(_selection_state, 'h3_callback', False)
                if nested_guard:
                    _selection_state.h3_callback = True
                try:
                    return previous(*args, **kwargs)
                finally:
                    _selection_state.h3_callback = old_guard
        changed._pi_h3 = True
        shared.opts.onchange(key, changed, call=False)


def restore_saved_selection():
    """Honor Forge's saved H3 preset after other extensions restore theirs."""
    from modules import shared
    if getattr(shared.opts, 'forge_preset', None) != PRESET:
        return
    if output_mode() == VIDEO_OUTPUT:
        from . import preset
        checkpoint, modules = preset.native_defaults()
        if checkpoint:
            shared.opts.set('sd_model_checkpoint', checkpoint)
            shared.opts.set('forge_checkpoint_' + PRESET, checkpoint)
            shared.opts.set('forge_additional_modules', modules)
            shared.opts.set('forge_additional_modules_' + PRESET, modules)
        return
    current = getattr(shared.opts, 'sd_model_checkpoint', None)
    checkpoint = current if selected_value(current) else LABEL
    shared.opts.set('sd_model_checkpoint', checkpoint)
    shared.opts.set('forge_checkpoint_' + PRESET, checkpoint)
    shared.opts.set('forge_additional_modules', [])
    shared.opts.set('forge_additional_modules_' + PRESET, [])


def _wrap_process(original):
    if getattr(original, '_pi_h3_process', False):
        return original
    def process(p, *args, **kwargs):
        _normalize_request(p, getattr(p, 'scripts', None))
        if selected(p):
            return runtime.generate(p, options(getattr(p, 'scripts', None), p))
        # A concurrent API request can switch engines without changing the
        # global dropdown first. Mark the H3 run cancelled before closing its
        # worker so it exits as a normal model switch, not a worker crash.
        _cancel('ordinary process_images request')
        return original(p, *args, **kwargs)
    process._pi_h3_process = True
    process._pi_h3_original = original
    return process


def _current_script_defaults(defaults, runner):
    """Extend an API startup snapshot if Forge finishes building script controls later."""
    scripts = list(getattr(runner, 'scripts', []) or [])
    if not any(getattr(script, '_pi_h3', False) for script in scripts):
        return defaults
    required = max([1] + [getattr(script, 'args_to', 1) for script in scripts])
    if len(defaults) >= required:
        return defaults
    result = list(defaults) + [None] * (required - len(defaults))
    for script in scripts:
        start, stop = getattr(script, 'args_from', 0), getattr(script, 'args_to', 0)
        controls = list(getattr(script, 'controls', []) or [])
        if stop > len(defaults) and len(controls) == stop - start:
            result[start:stop] = [getattr(control, 'value', None) for control in controls]
    return result


def install_api_argument_guard():
    """Protect early API calls from Forge's stale script-default snapshot."""
    import sys
    module = sys.modules.get('modules.api.api')
    api = getattr(module, 'Api', None) if module else None
    original = getattr(api, 'init_script_args', None) if api else None
    if not callable(original) or getattr(original, '_pi_h3_defaults', False):
        return
    def init_script_args(self, request, defaults, selectable_scripts, selectable_idx,
                         runner, **kwargs):
        current = _current_script_defaults(defaults, runner)
        return original(self, request, current, selectable_scripts, selectable_idx,
                        runner, **kwargs)
    init_script_args._pi_h3_defaults = True
    init_script_args._pi_h3_original = original
    api.init_script_args = init_script_args


def install_forge_selection():
    """Bypass stock checkpoint loading only for the worker-owned H3 marker."""
    try:
        from modules import shared
        from modules_forge import main_entry
    except ImportError:
        return
    original = getattr(main_entry, 'checkpoint_change', None)
    if not callable(original) or getattr(original, '_pi_h3_checkpoint', False):
        return
    def checkpoint_change(ckpt_name, preset, save=True, refresh=True):
        if preset == LEGACY_VIDEO_PRESET:
            preset = PRESET
        from modules import sd_models
        item = sd_models.checkpoints_list.get(ckpt_name) or sd_models.checkpoint_aliases.get(ckpt_name)
        if (preset == PRESET and output_mode() == VIDEO_OUTPUT
                and (ckpt_name == LABEL or getattr(item, '_pi_h3', False))):
            from . import preset as preset_module
            physical, modules = preset_module.native_defaults()
            if physical:
                ckpt_name = physical
                shared.opts.set('forge_additional_modules', modules)
                shared.opts.set('forge_additional_modules_' + PRESET, modules)
        if not selected_value(ckpt_name, preset, output_mode()):
            if selected():
                _cancel(f'checkpoint leaving H3: {ckpt_name!r}')
            _selection_state.explicit_leave = True
            try:
                return original(ckpt_name, preset, save=save, refresh=refresh)
            finally:
                _selection_state.explicit_leave = False
        current = getattr(shared.opts, 'sd_model_checkpoint', None)
        changed = not _same_checkpoint(current, ckpt_name)
        effective = ckpt_name if changed else current
        shared.opts.set('sd_model_checkpoint', effective)
        if preset is not None:
            shared.opts.set('forge_checkpoint_' + preset, effective)
        shared.opts.set('forge_additional_modules', [])
        if preset is not None:
            shared.opts.set('forge_additional_modules_' + preset, [])
        if save:
            shared.opts.save(shared.config_filename)
        if changed:
            _cancel(f'H3 checkpoint changed: {current!r} -> {ckpt_name!r}')
        return changed
    checkpoint_change._pi_h3_checkpoint = True
    checkpoint_change._pi_h3_original = original
    main_entry.checkpoint_change = checkpoint_change


def register_ui_binding(panel, is_img2img=False):
    binding = (panel, bool(is_img2img))
    if binding not in _UI_BINDINGS:
        _UI_BINDINGS.append(binding)


def ui_active(value=None):
    from modules import shared, sd_models
    if getattr(shared.opts, 'forge_preset', None) != PRESET:
        return False
    value = value or shared.opts.sd_model_checkpoint
    item = sd_models.checkpoints_list.get(value) or sd_models.checkpoint_aliases.get(value)
    return (value == LABEL or bool(getattr(item, '_pi_h3', False))
            or classify(getattr(item, 'filename', value or '')) == 'dit')


def _ui_load_state(has_denoise):
    import gradio as gr
    from modules import shared
    if getattr(shared.opts, 'forge_preset', None) == PRESET and not selected():
        restore_saved_selection()
    value = getattr(shared.opts, 'sd_model_checkpoint', None)
    active = ui_active(value)
    modules = list(getattr(shared.opts, 'forge_additional_modules', []) or []) if output_mode() == VIDEO_OUTPUT else []
    updates = [gr.update(value=value), gr.update(value=modules) if active else gr.skip(), gr.update(visible=active, open=False)]
    if has_denoise:
        updates.append(gr.update(value=1.0) if active else gr.skip())
    return updates


def _ui_selection_state(value, has_denoise):
    import gradio as gr
    from modules import shared
    active = ui_active(value)
    backend_active = ui_active()
    # Page initialization can briefly emit the checkpoint embedded before H3
    # restores its saved state. Never cancel an active H3 request for that
    # stale browser value; a real switch is cancelled by the backend option hook.
    if not active and not backend_active:
        _cancel(f'UI selected non-H3 checkpoint: {value!r}')
    updates = [gr.update(visible=active),
               gr.update(value=[]) if active and selected() else gr.skip()]
    if has_denoise:
        updates.append(gr.update(value=1.0) if active else gr.skip())
    return updates


def install_forge_ui_sync():
    """Register after Forge's own page-load preset callback, so H3 wins last."""
    try:
        from modules_forge import main_entry
    except ImportError:
        return
    original = getattr(main_entry, 'forge_main_entry', None)
    if not callable(original) or getattr(original, '_pi_h3_ui_sync', False):
        return
    def forge_main_entry(*args, **kwargs):
        result = original(*args, **kwargs)
        from gradio.context import Context
        checkpoint, modules = main_entry.ui_checkpoint, main_entry.ui_vae
        for panel, is_img2img in list(_UI_BINDINGS):
            denoise = main_entry.get_a1111_ui_component('img2img', 'Denoising strength') if is_img2img else None
            change_outputs = [panel, modules] + ([denoise] if denoise is not None else [])
            checkpoint.change(
                lambda value, has_denoise=denoise is not None: _ui_selection_state(value, has_denoise),
                inputs=[checkpoint], outputs=change_outputs, queue=False, show_progress='hidden')
            outputs = [checkpoint, modules, panel] + ([denoise] if denoise is not None else [])
            Context.root_block.load(
                lambda has_denoise=denoise is not None: _ui_load_state(has_denoise),
                outputs=outputs, queue=False, show_progress='hidden')
        return result
    forge_main_entry._pi_h3_ui_sync = True
    forge_main_entry._pi_h3_original = original
    main_entry.forge_main_entry = forge_main_entry


def _ready(*args, **kwargs):
    import sys
    from modules import processing, sd_models, shared
    from . import refresh, autolink
    autolink.install()
    refresh.install(sd_models, lambda: getattr(shared.opts, 'forge_preset', None) == PRESET)
    processing.process_images = _wrap_process(processing.process_images)
    for name in ('modules.api.api', 'modules.txt2img', 'modules.img2img'):
        module = sys.modules.get(name)
        current = getattr(module, 'process_images', None) if module else None
        if callable(current):
            module.process_images = _wrap_process(current)
    install_selection()
    install_api_argument_guard()
    install_forge_selection()
    install_forge_ui_sync()
    if not args and not kwargs:
        _UI_BINDINGS.clear()
    restore_saved_selection()
    register()


def _ensure_callback(script_callbacks, category, register, callback):
    callback_map = getattr(script_callbacks, 'callback_map', None)
    current = callback_map.get(category, []) if isinstance(callback_map, dict) else []
    if any(getattr(item, 'callback', None) is callback for item in current):
        return
    register(callback)


def install():
    from modules import scripts, processing, sd_models, script_callbacks
    if not all(callable(x) for x in (getattr(scripts.ScriptRunner, 'run', None),
            getattr(processing, 'process_images', None), getattr(sd_models, 'list_models', None))):
        raise RuntimeError('Forge hooks changed; H3 disabled. Other engines are unchanged.')
    if not getattr(scripts.ScriptRunner.run, '_pi_h3', False):
        original_run, original_list = scripts.ScriptRunner.run, sd_models.list_models
        def run(runner, p, *args, **kwargs):
            _normalize_request(p, runner)
            if selected(p, runner):
                return runtime.generate(p, options(runner, p))
            _cancel('ordinary ScriptRunner request')
            return original_run(runner, p, *args, **kwargs)
        def listing(*args, **kwargs):
            from modules import shared
            guard = (getattr(shared.opts, 'forge_preset', None) == PRESET and selected())
            old_guard = getattr(_selection_state, 'h3_refresh', False)
            if guard:
                _selection_state.h3_refresh = True
            try:
                result = original_list(*args, **kwargs)
            finally:
                register()
                _selection_state.h3_refresh = old_guard
            restore_saved_selection()
            return result
        run._pi_h3 = listing._pi_h3 = True
        scripts.ScriptRunner.run, sd_models.list_models = run, listing
    processing.process_images = _wrap_process(processing.process_images)
    _ensure_callback(script_callbacks, 'callbacks_app_started', script_callbacks.on_app_started, _ready)
    before_ui = getattr(script_callbacks, 'on_before_ui', None)
    if callable(before_ui):
        _ensure_callback(script_callbacks, 'callbacks_before_ui', before_ui, _ready)
    _ensure_callback(script_callbacks, 'callbacks_script_unloaded', script_callbacks.on_script_unloaded, runtime.selection_changed)
    install_forge_selection()
    install_forge_ui_sync()
    install_api_argument_guard()
    register()
