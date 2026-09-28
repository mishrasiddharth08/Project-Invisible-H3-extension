import threading
import tempfile
from pathlib import Path
from PIL import Image
from .config import settings
from .assets import scan, resolve, header
from .request import validate, parse_loras
from .progress import Progress
from .transport import Worker

LOCK = threading.RLock()
_worker = None
_selection_cancel = threading.Event()


def release():
    global _worker
    worker, _worker = _worker, None
    if worker is not None:
        worker.close()


def selection_changed():
    _selection_cancel.set()
    release()


def generate(p, options):
    global _worker
    from modules import shared, processing, images
    with LOCK:
        _selection_cancel.clear()
        editing = isinstance(p, processing.StableDiffusionProcessingImg2Img)
        refs = list(getattr(p, 'init_images', []) or []) if editing else []
        if editing and len(refs) > 1:
            raise ValueError('Use one primary img2img image and the extra H3 reference slots; img2img batch input is unsupported.')
        refs += [image for image in options.get('refs', []) if image is not None] if editing else []
        if len(refs) > 9:
            raise ValueError('H3 supports at most nine reference images.')
        sampler, scheduler = validate(p.width, p.height, p.steps, float(p.cfg_scale), p.sampler_name,
            getattr(p, 'scheduler', 'Simple'), editing, refs[0] if refs else None,
            getattr(p, 'image_mask', None), getattr(p, 'denoising_strength', 1.0), getattr(p, 'enable_hr', False))
        if getattr(p, 'restore_faces', False) or getattr(p, 'tiling', False):
            raise ValueError('Disable Restore faces and Tiling for H3 stills.')
        if getattr(p, 'subseed_strength', 0) or getattr(p, 'seed_resize_from_w', 0) > 0 or getattr(p, 'seed_resize_from_h', 0) > 0:
            raise ValueError('H3 does not support variation seeds or seed resizing. Set these controls to zero.')
        inventory = scan()
        paths = {k: resolve(options.get(k), k, inventory) for k in ('dit', 'clip', 'vae')}
        processing.fix_seed(p)
        count = int(p.batch_size) * int(p.n_iter)
        if not 1 <= count <= 64:
            raise ValueError('Choose a total batch of 1–64 images.')
        progress = Progress(shared, int(p.steps), count)
        output, seeds, prompts, negatives, infos = [], [], [], [], []
        cancelled = lambda: bool(shared.state.interrupted or shared.state.skipped or _selection_cancel.is_set())
        try:
            for index in range(count):
                if shared.state.interrupted or _selection_cancel.is_set():
                    break
                shared.state.skipped = False
                prompt = p.prompt[index % len(p.prompt)] if isinstance(p.prompt, list) and p.prompt else p.prompt
                negative = p.negative_prompt[index % len(p.negative_prompt)] if isinstance(p.negative_prompt, list) and p.negative_prompt else p.negative_prompt
                styles = getattr(p, 'styles', [])
                if styles:
                    prompt = shared.prompt_styles.apply_styles_to_prompt(prompt, styles)
                    negative = shared.prompt_styles.apply_negative_styles_to_prompt(negative, styles)
                prompt, tags = parse_loras(prompt)
                if editing and '<Picture ' not in prompt:
                    prompt = '<Picture 1> ' + prompt
                adapters = []
                if options.get('lora') not in (None, '', '(none)'):
                    tags.append((options['lora'], float(options.get('strength', 0.38))))
                for name, strength in tags:
                    matches = [path for path in inventory['lora'] if path == name or Path(path).stem == name]
                    if len(matches) != 1:
                        raise ValueError('Choose an unambiguous H3 LoRA file: ' + name)
                    header(matches[0])
                    adapters.append({'path': matches[0], 'strength': strength})
                seed = (int(p.seed) + index) % (2**63)
                mode = options.get('memory', 'auto')
                if mode not in ('auto', 'lowvram', 'cpu'):
                    raise ValueError('Invalid H3 memory mode.')
                progress.start(index, p.width, p.height)
                try:
                    from backend import memory_management
                except ImportError as error:
                    raise RuntimeError('Forge memory interface changed; H3 cannot safely load. Update the extension.') from error
                # Free Forge-side checkpoint/TE/VAE residency before every H3 request; the worker owns the GPU.
                memory_management.unload_all_models()
                if _worker is None or _worker.mode != mode or _worker.process.poll() is not None:
                    release()
                    _worker = Worker(mode)
                    _worker.wait_ready(cancelled)
                worker = _worker
                with tempfile.TemporaryDirectory(prefix='pi-h3-') as temporary:
                    references = []
                    for n, image in enumerate(refs):
                        path = Path(temporary) / f'reference-{n}.png'
                        if not isinstance(image, Image.Image):
                            raise ValueError('H3 reference must be an image.')
                        image.convert('RGB').save(path)
                        references.append(str(path))
                    request = dict(paths=paths, prompt=prompt, negative=negative, references=references,
                        adapters=adapters, seed=seed, width=int(p.width), height=int(p.height), steps=int(p.steps),
                        cfg=float(p.cfg_scale), sampler=sampler, scheduler=scheduler, preview=progress.preview,
                        preview_interval=max(1.0, float(settings()['preview_interval'])), output=str(Path(temporary)/'result.png'))
                    try:
                        result_path = worker.generate(request, cancelled, progress.update)
                    except (InterruptedError, RuntimeError, OSError):
                        if not cancelled():
                            raise
                        release()
                        if shared.state.skipped and not shared.state.interrupted and not _selection_cancel.is_set():
                            continue
                        break
                    with Image.open(result_path) as image:
                        result = image.copy()
                if result.size != (p.width, p.height):
                    raise RuntimeError('H3 returned incorrect dimensions; output was not saved.')
                if editing and options.get('drift') and refs:
                    # Pixel-drift fix (Mozer's PixelDriftFix): realign the edit to the source framing.
                    from .drift import align
                    try:
                        aligned = align(refs[0], result)
                    except Exception as error:
                        print('[PI-H3] Pixel-drift fix skipped:', error)
                        aligned = None
                    if aligned is not None and aligned.size == (p.width, p.height):
                        result = aligned
                    elif aligned is not None:
                        print('[PI-H3] Pixel-drift fix skipped: aligned size', aligned.size,
                              'does not match the requested', (p.width, p.height))
                progress.finish(result)
                info = (f'{prompt}\nNegative prompt: {negative}\nSteps: {p.steps}, Sampler: {p.sampler_name}, '
                        f'Schedule type: {scheduler}, CFG scale: {p.cfg_scale}, Seed: {seed}, Size: {p.width}x{p.height}, '
                        f'Model: {Path(paths["dit"]).name}, H3 encoder: {Path(paths["clip"]).name}, '
                        f'H3 VAE: {Path(paths["vae"]).name}, H3 decode: Fizgig group 5 frame 3, '
                        f'H3 references: {len(refs)}, H3 adapters: {adapters}')
                p.sd_model_name = Path(paths['dit']).stem
                p.sd_vae_name = Path(paths['vae']).stem
                p.extra_generation_params.update({'H3 decode': 'Fizgig group 5 frame 3', 'H3 references': len(refs)})
                result.info['parameters'] = info
                if shared.opts.samples_save and not getattr(p, 'do_not_save_samples', False):
                    extension = getattr(shared.opts, 'samples_format', 'png')
                    saved = images.save_image(result, p.outpath_samples, '', seed, prompt, extension=extension, info=info, p=p)
                    if not saved or not saved[0] or not Path(saved[0]).is_file():
                        raise RuntimeError('H3 generated an image, but Forge could not save it.')
                output.append(result)
                seeds.append(seed)
                prompts.append(prompt)
                negatives.append(negative)
                infos.append(info)
            return processing.Processed(p, output, seed=seeds[0] if seeds else int(p.seed),
                info=infos[0] if infos else 'H3 stopped.', all_seeds=seeds, all_prompts=prompts,
                all_negative_prompts=negatives, infotexts=infos)
        except BaseException:
            release()
            raise
        finally:
            progress.close()
            if not options.get('keep_loaded', False) or cancelled():
                release()
