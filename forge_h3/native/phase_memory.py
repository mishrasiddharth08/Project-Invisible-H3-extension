"""Release completed H3 conditioning weights before native video sampling."""


def release_conditioning(engine, memory):
    objects = getattr(engine, 'forge_objects', None)
    targets = [getattr(getattr(objects, name, None), 'patcher', None)
               for name in ('clip', 'vae')]
    targets.append(getattr(getattr(engine, 'audio_vae', None), 'patcher', None))
    targets = [patcher for patcher in targets if patcher is not None]
    loaded = list(memory.current_loaded_models)
    selected = [item for item in loaded if any(item.model is patcher for patcher in targets)]
    if not selected:
        return 0
    # Keep the DiT and every unrelated loaded model. Force a full offload of
    # only this engine's completed encoder/VAE weights, retaining prompt data.
    keep = [item for item in loaded if not any(item is target for target in selected)]
    devices = []
    for item in selected:
        if item.device not in devices:
            devices.append(item.device)
    for device in devices:
        memory.free_memory(1e30, device, keep_loaded=keep)
    return len(selected)
