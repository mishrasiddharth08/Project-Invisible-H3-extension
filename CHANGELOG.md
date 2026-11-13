# Changes

## 2026-11-13 (image preview)

- Add MiniMax-H3-Image-only 32-grid dimension controls, up to 8192 per side
  with a strict 16,000,000-pixel area cap and restoration for other models.
- Match current Forge text-encoder loading, vision settings and Shift calls.
- Keep Pruna image detection ahead of the legacy audio-layout fallback.
- Honor per-request reference mode for ordinary img2img.
- Validate native image, Pruna and legacy still/video routes in an isolated
  Forge Neo 2.32 session; retain the draft while Pruna reference quality is unresolved.
- Record measured settings, timings, negative tests and remaining gaps in VALIDATION.md.

## 2026-10-07

- Start video at 1152 x 768, 20 steps and 124 frames.
- Resolve compatible H3 components for empty API selections.
- Clarify that compact memory controls govern the still worker.
- Publish an actual 768p video, refreshed UI and infographic.
- Document observed VRAM and plain-LoRA versus other adapter performance.
- Validation: 129 main + 106 native CPU tests; real 768p generation,
  empty-component live regression, single H3 preset and UI mode switching.

### VRAM profile update

- Add Auto and 8/10/12/16/24/32 GiB soft residency profiles.
- Keep GPU encoding available; account for free memory in worker Auto mode.
- Bound native VAE tile batches and move smaller-profile pixel canvases to CPU.
- Reserve decode activation space and release completed conditioning weights.
- Restore Forge memory settings after success, cancellation or exceptions.
- Restore Shift on output changes and order mode-switch callbacks.
- 245 CPU tests pass; six native profile smoke requests passed on one 32 GiB GPU.

- Fix quantized still-worker offload tensor version counters; real 8 GiB soft-profile request passes.
