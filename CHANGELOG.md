# Changes

## H3 audit fixes

- Keep Refiner visible and restore the previous Hires. fix checkbox when leaving
  H3 Image models. Keep the factory output size at Full size while allowing the
  Downscale dropdown to use saved Settings until explicitly overridden.
- Clear legacy H3 sampling state after failure, bound native H3 Image reference
  allocations before resizing, and apply the Pruna edit prefix to prompt lists.
- Respect explicit API Shift values and use the corresponding tab's preset when
  the request omits Shift. Downscale by exact 2 x 2 averaging at 50% on an even grid.

## 2026-11-21

- Add a small Downscale accordion to txt2img and img2img with Full size as the
  default, plus optional 50% and 33% Lanczos output sizes.

## 2026-11-20

- Add txt2img and img2img Config-Presets entries for ER-SDE, Beta, four steps
  and CFG 1. Dimensions and adapters use the current selections.

## 2026-11-18 (image preview)

- Use affirmative forehead and hair-part guidance for MiniMax-H3-Image while
  preserving explicit requests for a bindi or sindoor.
- Remove the MiniMax-H3-Image sampler hint from the Forge panel.

## 2026-11-13 (image preview)

- Add MiniMax-H3-Image-only 32-grid dimension controls, up to 8192 per side
  with a strict 16,777,216-pixel area cap and restoration for other models.
  Raise the initial decimal 16 MP cap to permit the requested 4096-square output.
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
