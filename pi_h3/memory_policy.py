"""H3-only soft residency budgets; never resize or change sampling math."""
import contextlib
import math
import threading

GIB = 2**30
KEY = 'pi_h3_vram_profile'
CHOICES = ('Auto', '8', '10', '12', '16', '24', '32')
_lock = threading.RLock()
_local = threading.local()


def validate(value):
    value = str(value or 'Auto')
    if value not in CHOICES:
        raise ValueError('H3 VRAM profile must be Auto, 8, 10, 12, 16, 24 or 32.')
    return value


def selected(p=None):
    from modules import shared
    overrides = getattr(p, 'override_settings', {}) or {}
    return validate(overrides.get(KEY, getattr(shared.opts, KEY, 'Auto')))


def budget_gib(total, profile='Auto'):
    validate(profile)
    total = float(total)
    if not math.isfinite(total) or total <= 0:
        raise ValueError('Cannot determine H3 GPU memory safely.')
    return min(total, float(profile)) if profile != 'Auto' else total


def tile_batch_limit(budget):
    return 1 if budget < 20 else 2 if budget < 28 else 4


def worker_flags(mode, total=None, free=None, profile='Auto'):
    if mode == 'cpu':
        return ['--cpu']
    if mode not in ('auto', 'lowvram'):
        raise ValueError('Invalid H3 memory mode.')
    validate(profile)
    if total is None or free is None:
        return ['--novram']  # Detection failure must never force full residency.
    budget = budget_gib(total, profile)
    free = float(free)
    if not math.isfinite(free):
        return ['--novram']
    free = max(0, min(free, budget))
    # NORMAL mode already stages weights; LOW/NO force the whole giant encoder to CPU.
    # Keep explicit lowvram as a user choice. For Auto, reserve more under pressure.
    tier = '--lowvram' if mode == 'lowvram' else None
    reserve = max(1.2, float(total) - budget + 1.2 + max(0, min(budget, 8) - free))
    return ([tier] if tier else []) + ['--reserve-vram', str(round(reserve, 3))]



def gpu_memory():
    import torch
    if not torch.cuda.is_available():
        return None
    device = torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(device)
    return total / GIB, free / GIB


def active_tile_limit():
    return getattr(_local, 'tile_limit', 4)


@contextlib.contextmanager
def native_scope(p):
    """Temporarily budget Forge residency for one serialized native H3 request."""
    if not getattr(p, '_pi_h3_native_memory', False):
        yield
        return
    profile = selected(p)
    stats = gpu_memory()
    if stats is None:
        yield
        return
    from backend import memory_management as mm
    total, free = stats
    budget = budget_gib(total, profile)
    with _lock:
        old = (mm.SETTING_RESERVED_VRAM, mm.vram_state, mm.set_vram_to,
               getattr(_local, 'tile_limit', None))
        # Existing user headroom remains authoritative when it is larger.
        mm.SETTING_RESERVED_VRAM = max(old[0], int((total - budget + 1.2) * GIB))
        # NORMAL still supports partial weight loading, while keeping encoding on GPU.
        # LOW forces this Forge text encoder to CPU and is prohibitively slow.
        state = mm.VRAMState.NORMAL_VRAM
        mm.vram_state = mm.set_vram_to = state
        _local.tile_limit = tile_batch_limit(budget)
        print(f'[PI-H3] VRAM profile {profile}: {budget:.2f} GiB soft budget, '
              f'{free:.2f} GiB free, VAE tile batch <= {_local.tile_limit}')
        try:
            yield
        finally:
            mm.SETTING_RESERVED_VRAM, mm.vram_state, mm.set_vram_to = old[:3]
            if old[3] is None:
                del _local.tile_limit
            else:
                _local.tile_limit = old[3]
