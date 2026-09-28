# H3 quantization support — September 28, 2026

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
| Text encoder | integer INT4 W4A4/W4A8 | No | Backend linear formats supported; H3 conversion unverified |
| Video VAE | FP16 | Yes · 5.21 GB | **Locally tested** |
| Video VAE | INT8 ConvRot | Yes · 2.81 GB | Official; not locally tested |
| Video VAE | INT4 / FP8 / NVFP4 | No | No supported H3 VAE file established |
| Any component | GGUF | No | Unsupported by this dedicated worker |

The tested combination is the user's pruned INT8 ConvRot DiT, NVFP4-AWQ text
encoder and FP16 video VAE. Real txt2img and experimental FL2VA reference runs
loaded these files. This does not validate every format in the table.

## Native backend formats

The pinned backend recognizes `float8_e4m3fn`, `float8_e5m2`, `mxfp8`,
`nvfp4`, `int8_tensorwise`, `convrot_w4a4`, and `asym_w4a8_int8` metadata.
The last two are the native integer INT4 routes: 4-bit weight/4-bit activation
and 4-bit weight/8-bit activation. Hardware kernels, correct conversion and H3
tensor compatibility are still required.

Detection accepts those metadata names without claiming inference success.
Unknown pickle files and GGUF remain blocked. Do not rename a GGUF or arbitrary
INT4 model to `.safetensors`.

## Speed and memory

File size is only a storage clue. Peak VRAM also includes text/image tokens,
reference latents, attention workspace, VAE work and allocator reserve. Lower
bit width may reduce transfer and residency, but kernel support can make a
smaller format slower.

On the RTX 5090 test host, the current files loaded at approximately 20.00 GB
for the DiT, 14.96 GB for the text encoder, and 4.97 GB for the VAE in separate
stages. A 1536×2048 run reached about 31.9 GB total GPU use. Compare formats
with the same prompt, seed, dimensions, steps, sampler, schedule and cold/warm
state; record load-to-first-step time, steps/second, peak dedicated/shared GPU
memory, final pixels and visual quality.

Sources: `resources/catalog.json` at pinned Comfy-Org revision
`bf92c4091e333e69b8ca1998e0a669f15cb0832b`, and the pinned backend files
`comfy/quant_ops.py`, `comfy/ops.py`, and `comfy/sd.py`.
