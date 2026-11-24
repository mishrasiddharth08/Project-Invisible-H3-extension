"""Keep Forge's native Stop result iterable and clear H3 sampling state on failure."""


import logging

logger = logging.getLogger("forge_h3")

def guard_sample(p):
    original = getattr(p, 'sample', None)
    if not callable(original):
        return
    if getattr(original, '_pi_h3_cancel', False) is True:
        return
    def sample(*args, **kwargs):
        if getattr(p, "h3_request", None) is None:
            return original(*args, **kwargs)
        from .native.patches import end_sampling

        try:
            result = original(*args, **kwargs)
        except BaseException:
            # Releasing a failed generation must not hide the sampler's original error.
            try:
                p.sd_model.release_generation()
            except Exception:
                logger.exception("Could not release the failed H3 generation.")
            raise
        finally:
            end_sampling()
        if result is None:
            from modules import shared
            if shared.state.interrupted or shared.state.skipped:
                p.sd_model.release_generation()
                return []
        return result
    sample._pi_h3_cancel = True
    p.sample = sample
