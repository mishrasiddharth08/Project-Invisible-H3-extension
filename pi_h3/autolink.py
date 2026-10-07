"""Keep AutoLink's automatic saved set compatible with H3 Video."""
import sys
from functools import wraps

def install():
    module = sys.modules.get('model_autolink.integration')
    resolver = getattr(module, '_resolver', None)
    if resolver is None:
        return
    original = resolver.resolve
    if getattr(original, '_pi_h3_native_components', None) is original:
        return
    @wraps(original)
    def resolve(*args, **kwargs):
        selected, family, source = original(*args, **kwargs)
        from modules import shared
        from .config import PRESET, VIDEO_OUTPUT
        from .forge import output_mode
        if (family == 'minimax_h3' and getattr(shared.opts, 'forge_preset', None) == PRESET
                and output_mode() == VIDEO_OUTPUT):
            from .preset import native_defaults, native_modules_valid
            if not native_modules_valid(selected):
                _, compatible = native_defaults()
                if native_modules_valid(compatible):
                    return compatible, family, 'H3 video compatibility'
        return selected, family, source
    resolve._pi_h3_native_components = resolve
    resolver.resolve = resolve
