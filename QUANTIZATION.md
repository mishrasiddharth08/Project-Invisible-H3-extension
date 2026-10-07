# Unified native route

H3 Video output uses Eduardo Abreu’s native Forge backend at revision 8ec38de. It implements GGUF diffusion models (Q2_K–Q8_0), W4A8, INT4 and INT8 formats supported by that backend and Forge. Native NVFP4-AWQ text encoding is explicitly rejected because upstream observed incorrect prompt conditioning. Use a compatible INT4, INT8 or BF16 encoder for the native route.

The dedicated still worker remains `.safetensors` only and supports its locally tested NVFP4-AWQ encoder. A physical GGUF checkpoint automatically uses the native backend, including when Output is Still image. This means GGUF is implemented in the combined extension even though the dedicated worker itself does not load GGUF.

# H3 quantization support

This extension loads ComfyUI native `.safetensors` files. A filename such as
`int4` is not proof of its format: the file must contain valid H3 tensors and
native `comfy_quant` metadata understood by the pinned backend.

## Status matrix

| Component | Format | Pinned official file | Status |
| --- | --- | --- | --- |
| DiT | BF16 / pruned BF16 | Yes · 66.28 / 40.23 GB | Official; not locally tested |
| DiT | FP8 scaled, pruned | Yes · 20.96 GB | Official; not locally tested |
| DiT | INT8 TensorWise + ConvRot, full/pruned | Yes · 34.04 / 20.97 GB | **Pruned locally tested** |
| DiT | integer INT4 W4A4 ConvRot | No | Backend format supported; H3 conversion unverified |
| DiT | integer INT4 W4A8 asymmetric | No | Backend format supported; H3 conversion unverified |
| DiT | NVFP4 | No | Backend format supported; no pinned H3 DiT file |
| Text encoder | BF16 | Yes · 51.51 GB | Official; not locally tested |
| Text encoder | INT8 TensorWise + ConvRot | Yes · 27.14 GB | Official; not locally tested |
| Text encoder | NVFP4 AWQ | Yes · 15.69 GB | **Locally tested**; FP4, not integer INT4 |
| Text encoder | INT4 ConvRot | Approved external file | **Locally tested on the native route** |
| Video VAE | FP16 | Yes · 5.21 GB | **Locally tested** |
| Video VAE | INT8 ConvRot | Yes · 2.81 GB | Official; not locally tested |
| Video VAE | INT4 / FP8 / NVFP4 | No | No supported H3 VAE file established |
| Diffusion model | GGUF Q2_K–Q8_0 | External files | Implemented by native route; full local generation unverified |

The dedicated worker's tested combination is the user's pruned INT8 ConvRot DiT,
NVFP4-AWQ text encoder and FP16 video VAE. The native route was separately tested
with the pruned INT8 ConvRot DiT, Qwen3-VL-32B INT4 ConvRot text encoder, FP16
video VAE and FP32 audio VAE. It completed native Still image, video with audio
and silent-video requests. This does not validate every format in the table.

## Native backend formats

The pinned backend recognizes `float8_e4m3fn`, `float8_e5m2`, `mxfp8`,
`nvfp4`, `int8_tensorwise`, `convrot_w4a4`, and `asym_w4a8_int8` metadata.
The last two are the native integer INT4 routes: 4-bit weight/4-bit activation
and 4-bit weight/8-bit activation. Hardware kernels, correct conversion and H3
tensor compatibility are still required.

Detection accepts those metadata names without claiming inference success for
every file. Unknown pickle files remain blocked. GGUF files are accepted only
through the native route; do not rename a GGUF or arbitrary INT4 model to
`.safetensors`.

## Speed and memory

File size is only a storage clue. Peak VRAM also includes text/image tokens,
reference latents, attention workspace, VAE work and allocator reserve. Lower
bit width may reduce transfer and residency, but kernel support can make a
smaller format slower.

On the RTX 5090 test host, the current files loaded at approximately 20.00 GB
for the DiT, 14.96 GB for the text encoder, and 4.97 GB for the VAE in separate
stages. Earlier large still-image runs do not establish a measured peak for this video test. Compare formats
with the same prompt, seed, dimensions, steps, sampler, schedule and cold/warm
state; record load-to-first-step time, steps/second, peak dedicated/shared GPU
memory, final pixels and visual quality.

Sources: `resources/catalog.json` at pinned Comfy-Org revision
`bf92c4091e333e69b8ca1998e0a669f15cb0832b`, and the pinned backend files
`comfy/quant_ops.py`, `comfy/ops.py`, and `comfy/sd.py`.


### Adapter performance

Native Forge can parse H3-compatible LoRA, LoHA, LoKr and DoRA weights when
their tensor keys match the model. Only plain 2D LoRA uses H3's optimized
quantized low-rank path. LoHA, LoKr, DoRA, LoCon/mid-weight, offset and
transformed patches use Forge's compatibility path, which may dequantize
weights each forward and require more time and VRAM. Full H3 GPU generation
with LoHA, LoKr and DoRA files remains unverified; CPU fallback tests do not
establish their full-model performance.

### Measured video memory and time

The 1152 x 768, 124-frame, 20-step RTX 5090 quality run took 335.82 seconds
including loading. A single total-GPU-memory observation was 26,784 MiB
(26.16 GiB); this was not peak telemetry or an isolated allocation measure.
It used the INT8 ConvRot DiT and INT4 ConvRot encoder with Forge offloading.
Speed and memory vary with dimensions, frames, components and adapters.
The still worker's Memory mode and Keep model controls do not govern native
video or GGUF; those routes use Forge memory management.


## VRAM profiles: 8 / 10 / 12 / 16 / 24 / 32 GiB

Open **H3 > Memory > VRAM profile**. **Auto** is the default; choose a smaller
profile to favor offloading. Both the native route and the dedicated still
worker understand these profiles. They are **soft weight-residency budgets,
not hard GPU peak limits**. Resolution, frames, steps and model precision
remain exactly as requested. Very large requests can still run out of memory.

| Profile | Native VAE tile batch cap | Memory behavior |
| --- | --- | --- |
| 8 / 10 / 12 / 16 | 1 | Partial GPU weights; CPU pixel canvas |
| 24 | 2 | Partial GPU weights; CPU pixel canvas |
| 32 | 4 | GPU pixel canvas; larger weight residency |
| Auto | Follows detected capacity | Detects GPU memory; worker also checks free memory |

The profiles leave approximately 2 GiB of inference/OS headroom and respect
larger existing Forge reservations. Native H3 restores Forge's prior memory
settings after each request. Completed conditioning weights are offloaded
before sampling; video VAE loading now accounts for decoding activations.
The overlap size, blending order and sampling math are preserved. Auto keeps
GPU text encoding available: forcing low/no-VRAM mode sent the entire large
encoder to CPU and was much slower. Explicit Still worker lowvram remains
available. Detection failures use conservative offloading rather than forcing
full residency. No Generate-time model download is added.

### Validation limits

All six profiles completed identical 384 x 256, 22-frame, 2-step native
functional requests on the **same physical RTX 5090 32 GiB card**. These are
neither quality examples nor physical 8–24 GiB compatibility tests. The first
8 GiB cold request observed **15,027 MiB total GPU usage**, demonstrating why
a soft profile is not an 8 GiB hard cap. Subsequent requests reused conditioning
and kernels; their timings cannot be compared as independent cold benchmarks.
The exact hardware/quantization combination must still be tested on smaller
cards. Large system RAM and a fast local drive remain necessary for offloading.

CPU validation: **138 main + 107 native = 245 tests**. Tests include all six
profile choices, free-memory pressure, invalid/detection-failure handling,
restoration after exceptions, unchanged dimensions and exact tile/canvas
values on deterministic tensor fixtures. They do not prove every GPU kernel,
quantization, LoKr/LoHA/DoRA adapter, or maximum workload fits a smaller card.

![H3 VRAM profiles](docs/assets/h3-vram-guide.svg)
