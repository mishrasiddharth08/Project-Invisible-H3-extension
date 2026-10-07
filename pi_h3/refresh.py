"""Serialize the complete checkpoint-refresh chain while H3 is selected."""
from functools import wraps
import threading

LOCK = threading.RLock()

def install(sd_models, active):
    original = sd_models.list_models
    if getattr(original, '_pi_h3_refresh_lock', None) is original:
        return
    @wraps(original)
    def listing(*args, **kwargs):
        if active():
            with LOCK:
                return original(*args, **kwargs)
        return original(*args, **kwargs)
    listing._pi_h3_refresh_lock = listing
    sd_models.list_models = listing
