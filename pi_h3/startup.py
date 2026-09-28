"""Avoid inspecting every stack frame to find a callback's source file."""
import functools
import inspect
import sys


def optimize_callbacks(module):
    original = module.add_callback
    if getattr(original, '_pi_h3_fast_source', False):
        return
    if 'filename' not in inspect.signature(original).parameters:
        return  # Older host: retain its own callback behavior.
    @functools.wraps(original)
    def add_callback(callbacks, fun, *, name=None, category='unknown', filename=None):
        if filename is None:
            frame = sys._getframe(1)
            try:
                while frame is not None and frame.f_code.co_filename in (__file__, module.__file__):
                    frame = frame.f_back
                filename = frame.f_code.co_filename if frame is not None else 'unknown file'
            finally:
                del frame
        return original(callbacks, fun, name=name, category=category, filename=filename)
    add_callback._pi_h3_fast_source = True
    module.add_callback = add_callback


def install():
    from modules import script_loading
    callbacks = sys.modules.get('modules.script_callbacks')
    if callbacks is not None:
        optimize_callbacks(callbacks)
        return
    original = script_loading.load_module
    if getattr(original, '_pi_h3_deferred_source', False):
        return
    @functools.wraps(original)
    def load_module(path):
        callbacks = sys.modules.get('modules.script_callbacks')
        if callbacks is not None:
            optimize_callbacks(callbacks)
            if script_loading.load_module is load_module:
                script_loading.load_module = original
        return original(path)
    load_module._pi_h3_deferred_source = True
    script_loading.load_module = load_module
