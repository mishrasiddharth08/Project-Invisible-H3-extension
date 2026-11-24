"""Carry Forge API field provenance through to H3 Image's processing hook."""

import logging
from contextvars import ContextVar
from functools import wraps
from inspect import signature

_SHIFT_FIELD = "distilled_cfg_scale"
_request = ContextVar("pi_h3_api_request", default=None)
logger = logging.getLogger("forge_h3")


def apply_shift(p, profile, opts):
    if not getattr(p, "is_api", False):
        if getattr(opts, "forge_preset", None) != profile.name:
            p.distilled_cfg_scale = profile.shift
        return

    current = _request.get()
    if current is None:
        # Some integrations call processing directly. A numeric default cannot tell us if
        # the caller supplied it; leave the value alone rather than guessing.
        return
    request, kind = current
    fields = getattr(request, "model_fields_set", None)
    if fields is None:
        fields = getattr(request, "__fields_set__", None)
    if fields is not None and _SHIFT_FIELD not in fields:
        shift = getattr(opts, f"{profile.name}_{kind}_dcfg", profile.shift)
        p.distilled_cfg_scale = profile.shift if shift is None else shift


def _wrap(endpoint, kind):
    endpoint_signature = signature(endpoint)
    request_parameter = tuple(endpoint_signature.parameters)[1]

    @wraps(endpoint)
    def wrapped(*args, **kwargs):
        request = endpoint_signature.bind(*args, **kwargs).arguments[request_parameter]
        # Keep the request object, not a snapshot of the supplied fields: Forge resolves
        # infotext inside the endpoint and adds those fields to the Pydantic field set.
        token = _request.set((request, kind))
        try:
            return endpoint(*args, **kwargs)
        finally:
            _request.reset(token)

    # FastAPI uses the original signature to resolve the request body. The wrapper
    # must also accept the original request parameter name as a keyword argument.
    return wrapped


def patch_api(api_cls):
    for name, kind in (("text2imgapi", "t2i"), ("img2imgapi", "i2i")):
        endpoint = getattr(api_cls, name)
        if getattr(endpoint, "_pi_h3_api_shift", False):
            continue
        wrapper = _wrap(endpoint, kind)
        wrapper._pi_h3_api_shift = True
        setattr(api_cls, name, wrapper)


def install():
    """Call before Forge binds the API route methods."""
    try:
        from modules.shared_cmd_options import cmd_opts
        if not (getattr(cmd_opts, "api", False) or getattr(cmd_opts, "nowebui", False)):
            return
        from modules.api.api import Api
        patch_api(Api)
    except Exception:
        logger.warning("MiniMax H3 could not preserve API Shift field provenance.", exc_info=True)
