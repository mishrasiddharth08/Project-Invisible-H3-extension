# Project Invisible — H3

MiniMax H3 still images and reference editing inside **Forge Neo**.
Select **H3** under **UI Preset**, then use the normal **Generate** button.

[Project page](https://mishrasiddharth08.github.io/Project-Invisible-H3-extension/) · [Test results](VALIDATION.md) · [Quantization support](QUANTIZATION.md)

![H3 portrait](docs/assets/portrait.png)

## Project Invisible

Use the familiar Forge workflow: native preset/checkpoint selectors, txt2img,
img2img, Generate, Stop, gallery and saving. Special controls appear only for H3.
No additional tab, virtual environment, ComfyUI server or Forge core changes.
The actual H3 backend runs in a dedicated process using Forge's Python. Its
files and helper packages live inside this extension; closing that process
releases its RAM and VRAM. Other engines retain their ordinary paths.

## Installation

Download this repository as ZIP and extract it into Forge’s `extensions` folder,
or use Extensions → Install from URL with:

```text
https://github.com/mishrasiddharth08/Project-Invisible-H3-extension.git
```


1. Copy the complete `project-invisible-minimax-h3` folder into Forge's `extensions` folder.
2. Restart Forge completely, then refresh the browser.
3. Select **H3** under **UI Preset**. The checkpoint selector shows
   **PROJECT INVISIBLE — MiniMax H3 Still**.
4. Open **H3 · Still images → Models** if you need to choose specific files.

The installer installs two small backend packages into `vendor/python`, never
into Forge's shared environment. Full backend source is bundled at a pinned
revision. Do not replace it with a random ComfyUI revision.

## Models — reuse your downloads

Manual downloading is recommended. Obtain files from
[Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main).
Read the model's license before downloading or using it.

| Required component | Recommended example |
| --- | --- |
| H3 model | `minimax_h3_fl2va_pruned_int8_convrot.safetensors` |
| H3 encoder | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` |
| H3 **video** VAE | `minimax_h3_video_vae_fp16.safetensors` or its `int8_convrot` version |

The audio VAE and separate single-frame VAE are **not** used for stills.
The three recommended quantized files total approximately 37 GiB.

Existing folders are scanned recursively:

* `models/Stable-diffusion/` — including your `MINIMAX H3` folder
* `models/diffusion_models/`
* `models/text_encoder/` and `models/text_encoders/`
* `models/VAE/`
* `models/Lora/`
* `models/MiniMax-H3/` — recommended for a new installation

Renamed H3 files with hyphens are supported. Tensor headers and the real loader
check component compatibility; the filename alone is not proof. For other
locations, add absolute directories to `model_roots` in `config.json`.
Click **Refresh local files** after adding files.

Optional downloads: open **Models → Optional automatic download**, select the
exact files, tick approval and click **Download selected models**. Files are
verified against pinned upstream SHA256 hashes. Existing complete files are
reused. **Generate never downloads weights or tokenizers.**

## Make an image

* **txt2img:** enter a prompt and press Generate.
* Use **ER SDE**, **Simple**, **CFG 1**, **50 steps** without a Turbo LoRA.
* The preset starts at **1536 × 1536**. Width and height must be multiples of
  **32**, from 64 through 4096. Larger images need more memory and time.
* For final quality, consider approximately 3 MP or higher. Small, few-step
  renders are useful for testing but do not represent H3's best quality.
* Native seed, steps, dimensions, CFG, batch size and batch count are honored.
  Batch images run sequentially to reduce peak memory.
* Supported samplers: ER SDE, Euler, Heun, DPM++ 2M, Res Multistep.
  Schedules: Simple, Normal, Beta, Karras; Automatic maps explicitly to Simple.
* CFG above 1 encodes the negative prompt and adds guidance computation.

## Edit an image

**Experimental with FL2VA:** the supplied Fizgig reference workflow can edit
images with this checkpoint, but Ref2VA is the model trained for reference
tasks. Face identity, exact preservation and nine-reference quality are not
guaranteed. Start at 768 × 1024 for editing tests. Large references make the
32B vision encoder much slower before the first sampling step; Stop remains
available. Increase resolution only after checking the edit.

1. Open the existing **img2img** tab.
2. Add your photo to the normal image input.
3. Set **Denoising strength to 1**. This is H3 reference conditioning, not SD noise blending.
4. Example: `<Picture 1> Change the teapot to blue. Keep the composition.`
5. Optional additional images are under **Extra reference images**. Their
   order is Picture 2 through Picture 9 after the primary image.

All editing belongs in img2img; references in another tab cannot leak into txt2img.
Empty extra slots are omitted. Use consecutive slots so prompt numbering stays clear.
Masks, inpainting, Hires fix, tiling, face restoration, variation seeds and
seed resizing are rejected with instructions instead of being silently ignored.
This extension outputs still RGB images, not video, audio or transparency.

## LoRAs

H3 LoRAs can be chosen in the compact controls or inserted using native
`<lora:filename:strength>` prompt tags. Keep filenames identifiable as H3.
Only adapter keys that match the H3 model are accepted. Adapters are applied
through the backend's quantization-aware patcher and removed after each image.
Non-H3 adapters are rejected; they never fall through to an unrelated loader.

Fizgig's published still example uses its **v4 step-600 EMA** Turbo adapter at
**0.38**, with **20 steps**. That adapter is not included. Your 4-step/8-step
video Turbo files are different adapters; do not assume the same settings or
still-image quality. No Turbo adapter is automatically enabled.

## Progress and memory

* Exactly two bars: **current image** plus Forge's **overall batch** bar.
* Enable Forge's **Live previews** to see a starting gradient followed by
  approximate previews calculated from actual intermediate H3 latents.
* Preview updates target one second when a sampling callback is available.
  A long model step can take longer. Previews never force a full VAE transfer.
* The full Fizgig decoder produces the final image after sampling. Decoding
  is included in progress; the last sampling step is not reported as a saved image.
* Automatic memory management/offload is the default. **lowvram** reduces GPU
  residency; **cpu** is a slow fallback. Sufficient system RAM is still necessary.
* By default, worker memory is released after each request. **Keep model in
  memory** permits reuse between runs. Selecting another preset/checkpoint
  always stops this extension's active job and releases its worker.
* Stop cancels the dedicated worker, including while encoding or decoding.
  It never kills Forge or another extension's process.

## Compatibility and honest limits

The backend is ComfyUI's real MiniMax H3 implementation, not an SD/Flux pipeline.
Fizgig's original single-frame latent and group decoder are retained unchanged.
Only the backend is used: no ComfyUI interface or server is launched.

NVIDIA CUDA is the tested path. AMD ROCm and CPU depend on backend/kernel
support; NVFP4 in particular is hardware dependent. DirectML is not supported.
No claim is made that every GPU, every VRAM size or every future Forge update
will work. A changed host hook fails with an error rather than patching Forge.
See `VALIDATION.md` for the actual tests and limits of this release.
See `QUANTIZATION.md` for component-specific formats and tested coverage.

To uninstall: close Forge and remove this extension's folder. Your downloaded
models and generated images are separate and remain available.

## Troubleshooting

* **No H3 preset:** restart Forge fully; check the terminal for `[PI-H3]` errors.
* **Missing model:** open Models, verify all three paths, then Refresh local files.
* **Wrong component:** select the H3 Qwen3-VL-32B encoder and H3 video VAE.
* **Out of memory:** reduce image dimensions, use lowvram and close other GPU jobs.
* **Stale browser controls:** refresh the browser after restarting Forge.
* **Need help:** include the complete terminal error, model filenames and settings.

## Credits and licenses

Thanks to Peter Neill / ShootTheSound for Fizgig H3 Still, ComfyUI contributors,
MiniMax, Haoming02 / Forge Neo, the Forge / AUTOMATIC1111 community, and the
Project Invisible extensions used as integration references.

See `NOTICE`, `LICENSE`, `vendor/ComfyUI/LICENSE` and
`pi_h3/vendor/FIZGIG-LICENSE`. Model licenses remain separate.

## Special Thanks

Special thanks to:

- [r/sdforall](https://www.reddit.com/r/sdforall/) — community discussion and testing
- [r/SECourses](https://www.reddit.com/r/SECourses/) — community discussion and testing
- [r/malcolmrey](https://www.reddit.com/r/malcolmrey/) — community discussion and testing
- [**Haoming02 / sd-webui-forge-classic (neo branch)**](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo) — the Forge Neo tree this extension targets
- [**ShootTheSound / Peter Neill**](https://github.com/shootthesound) and [**ComfyUI-Fizgig-H3-Still**](https://github.com/shootthesound/ComfyUI-Fizgig-H3-Still/tree/main) — H3 still-image latent and decoder implementation
- [**Adeliox**](https://github.com/Adeliox) — original Klein Head Swap
- [Alissonerdx](https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap) — BFS (Best Face Swap) workflow and LoRAs
- [PozzettiAndrea / ComfyUI-SAM3](https://github.com/PozzettiAndrea/ComfyUI-SAM3) and [Meta SAM3](https://github.com/facebookresearch/sam3) — segmentation workflow inspiration across Project Invisible
- [**ComfyUI**](https://github.com/comfyanonymous/ComfyUI) — H3 backend and upstream sampler/scheduler coverage
- The Forge / AUTOMATIC1111 community — for the extension ecosystem this plugs into
- Project Invisible extensions — memory policy, GPU compatibility and extension philosophy

Thank you to the wider Forge, Diffusers, Qwen, DeGrid and open-source communities.
Head-swap, BFS and SAM3 acknowledgments recognize the wider ecosystem; those tools are not bundled H3 features.
