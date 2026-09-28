# Validation — September 28, 2026

Tested on Forge Neo 2.29.1, RTX 5090 (32 GB), 96 GB RAM, using the
existing FL2VA int8_convrot model, Qwen3-VL-32B NVFP4 AWQ encoder and
H3 video VAE FP16. A separate Forge test instance preserved normal settings.

## Passed

- 55 automated tests passed, plus Python compilation.

- Native API and UI portrait generation: 1536 × 2048, ER SDE/Simple,
  CFG 1, 50 steps, no LoRA. API baseline: 86.42 seconds; UI natural portrait:
  approximately 88 seconds. Visible pores and fine lines; no obvious waxy
  smoothing in the inspected samples. This is visual judgment, not a general
  quality guarantee.
- Portrait editing: 768 × 1024, 30 steps, 82.14 seconds. White shirt changed
  to green; face, hair and composition stayed visually consistent.
- Same prompt and seed with different references reproduced the supplied
  portrait and teapot, confirming actual reference influence.
- Real LoRA off/on/off: adapter changed pixels; removal restored the exact
  original pixels (maximum difference 0).
- Two-image batch returned seeds 34567 and 34568.
- Browser refresh during generation preserved H3 and returned both images.
  Fixed the temporary checkpoint-list conflict with Ideogram.
- Native Stop during reference encoding returned no image and released GPU
  memory to 553 MiB total device use. Stop during sampling returned no image,
  wrote no output and exited the worker.
- Generation recovered after Stop. Keep-loaded mode retained the worker;
  switching to a non-H3 checkpoint exited it successfully.

## Limits

A later cold 3.15 MP UI run took 294 seconds to reach step 34/50 before
manual Stop. A 12 GB reserve experiment did not reach its first step within
the test window and was stopped; it is not enabled. The earlier 86-second
result is not a guaranteed speed. Investigate load/first-step latency on an
idle GPU before claiming consistent high speed. Separate GPU contexts were
present during the later tests, but their effect was not established.

See QUANTIZATION.md: integer INT4 detection is covered; real integer INT4
H3 inference and GGUF support are not established.

FL2VA reference editing is experimental; Ref2VA is the trained reference model.
The 3.15 MP reference edit was stopped after 393 seconds of conditioning;
use smaller editing tests first. Nine-reference fidelity, other GPUs, AMD,
CPU-only performance and low-VRAM configurations are not validated.
No skin filter or Forge core patch was added. The pre-existing core change
in modules/launch_utils.py was preserved.

Detailed logs and generated samples are in the sibling development workspace
`project-invisible-ideogram-4/work/h3-research`.
