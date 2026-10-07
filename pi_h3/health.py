"""Small, scoped, read-only diagnostics inspired by AiKimi's runtime status."""
native_enabled = False


def snapshot():
    from modules import shared
    from . import forge, runtime
    from forge_h3.integration import native_selected
    worker = runtime._worker
    worker_alive = bool(worker is not None and worker.process.poll() is None)
    output, source = forge.output_mode(with_source=True)
    backend = 'still worker' if forge.selected() else 'native' if native_selected() else 'inactive'
    return {'preset': getattr(shared.opts, 'forge_preset', None), 'backend': backend,
            'output': output, 'output_source': source,
            'parameter_status': f'pi_h3_output={output} ({source})',
            'native_enabled': native_enabled, 'worker_running': worker_alive,
            'worker_memory_mode': worker.mode if worker_alive else None}


def app_started(_demo, app):
    if getattr(app, '_pi_h3_health', False):
        return
    app.get('/pi-h3/status')(snapshot)
    app._pi_h3_health = True
