"""Keep Forge's native Stop result iterable for H3 only."""
def guard_sample(p):
    original = getattr(p, 'sample', None)
    if not callable(original):
        return
    if getattr(original, '_pi_h3_cancel', False) is True:
        return
    def sample(*args, **kwargs):
        result = original(*args, **kwargs)
        if result is None:
            from modules import shared
            if (shared.state.interrupted or shared.state.skipped) and getattr(p, 'h3_request', None) is not None:
                from .native.patches import end_sampling
                end_sampling()
                p.sd_model.release_generation()
                return []
        return result
    sample._pi_h3_cancel = True
    p.sample = sample
