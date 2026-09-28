"""Install worker-only helpers locally; never replace Forge's torch or packages."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
TARGET = ROOT / 'vendor' / 'python'


def main():
    if not (ROOT / 'vendor' / 'ComfyUI' / 'comfy' / 'ldm' / 'minimax' / 'model.py').is_file():
        raise RuntimeError('Incomplete H3 extension: install the complete release with its pinned backend.')
    # These are backend helpers, not a separate Python environment.
    for package, version in [('comfy_aimdo', '0.5.5'), ('comfy_kitchen', '0.2.35')]:
        if not (TARGET / f'{package}-{version}.dist-info').is_dir():
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-deps', '--target', str(TARGET),
                                   package.replace('_', '-') + '==' + version])


if __name__ == '__main__':
    main()
