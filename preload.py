"""Install a lightweight callback-registration optimization before UI scripts."""
import importlib.util
from pathlib import Path

def preload(parser):
    path = Path(__file__).parent / 'pi_h3' / 'startup.py'
    spec = importlib.util.spec_from_file_location('_pi_h3_startup', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.install()
