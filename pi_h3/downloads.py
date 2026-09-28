"""Explicit selected-file downloads only. Generate never calls this module."""
import hashlib
import json
from pathlib import Path
import shutil
import threading
from .config import ROOT, models_root
from .assets import header

CATALOG = json.loads((ROOT / 'resources' / 'catalog.json').read_text(encoding='utf-8'))
FILES = {entry['filename']: entry for entry in CATALOG['files']}
LOCK = threading.Lock()


def download(selected, approved):
    if not approved:
        return 'Tick approval for the selected model downloads first. Manual download is also available.'
    if not selected:
        return 'Select the model files you want to download.'
    if any(name not in FILES for name in selected):
        raise ValueError('Unknown model selection.')
    import requests
    with LOCK:
        folder = models_root() / 'MiniMax-H3'
        folder.mkdir(parents=True, exist_ok=True)
        needed = sum(FILES[name]['size'] for name in selected if not (folder/name).exists())
        if shutil.disk_usage(folder).free < needed + 1024**3:
            return 'Not enough free disk space for the selected files.'
        result = []
        for name in selected:
            destination = folder / name
            if destination.exists():
                header(destination)
                result.append('Kept existing ' + name)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix('.safetensors.partial')
            digest = hashlib.sha256()
            url = f'https://huggingface.co/{CATALOG["repo"]}/resolve/{CATALOG["revision"]}/{name}'
            try:
                with requests.get(url, stream=True, timeout=(20, 120)) as response:
                    response.raise_for_status()
                    with temporary.open('wb') as stream:
                        for chunk in response.iter_content(4 * 1024 * 1024):
                            stream.write(chunk)
                            digest.update(chunk)
                expected = FILES[name]
                if temporary.stat().st_size != expected['size'] or digest.hexdigest() != expected['sha256']:
                    raise ValueError('Downloaded file failed integrity check: ' + name)
                temporary.replace(destination)
                result.append('Downloaded and verified ' + name)
            finally:
                temporary.unlink(missing_ok=True)
        return '\n'.join(result) + '\nClick Refresh local files.'
