# Changes

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
