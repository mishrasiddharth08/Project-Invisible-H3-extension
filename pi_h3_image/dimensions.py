"""Model-local 16,777,216-pixel slider range; no Forge imports or GPU work."""
import math
from .contracts import GRID, MIN_SIDE, H3Error

MAX_PIXELS = 16_777_216
MAX_SIDE = 8192  # UI range, not an upstream per-axis support claim.

def fit_dimensions(width, height, changed=None):
    sides = [max(MIN_SIDE, min(MAX_SIDE, round(v / GRID) * GRID)) for v in (width, height)]
    if sides[0] * sides[1] > MAX_PIXELS:
        if changed is None:
            scale = math.sqrt(MAX_PIXELS / (sides[0] * sides[1]))
            sides = [max(MIN_SIDE, int(v * scale) // GRID * GRID) for v in sides]
        else:
            other = 1 - changed
            sides[other] = max(MIN_SIDE, (MAX_PIXELS // sides[changed]) // GRID * GRID)
    return tuple(sides)

def validate_dimensions(profile_name, width, height):
    if profile_name == "h3image" and width * height > MAX_PIXELS:
        raise H3Error("MiniMax-H3-Image output is limited to 16,777,216 pixels; reduce Width or Height.")



def dimension_updates(active, saved, values, defaults):
    saved = dict(saved or {"active": False})
    if active and not saved.get("active"):
        saved["sliders"] = [dict(config, value=value) for config, value in zip(defaults, values)]
        updates = [dict(minimum=MIN_SIDE, maximum=MAX_SIDE, step=GRID,
                        value=max(MIN_SIDE, min(MAX_SIDE, round(value / GRID) * GRID)))
                   for value in fit_dimensions(*values)]
    elif not active and saved.get("active"):
        updates = saved.pop("sliders")
    else:
        updates = [{}, {}]
    saved["active"] = active
    return saved, updates
