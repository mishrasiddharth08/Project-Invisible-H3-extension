# Forge Neo startup fixes

Measured locally on September 28, 2026: normal startup fell from 23.76 seconds to 20.66 seconds (one run each; warm caches and machine load affect results). No startup errors in the verification run. Existing Forge instance stayed running.

- H3 preload: obtain callback caller filenames without full stack inspection. Keeps native callback registration, naming and order; no Forge core file changes.
- ADetailer patch: reuse existing nonempty standard model files; download missing files only. Six focused tests passed.
- AutoLink patch: cache model fingerprints with file-change invalidation and independent metadata copies.
- Launcher patch: avoid implicitly enabling UV in fast mode, preventing the incompatible `pip freeze --all` diagnostics call. Explicit UV and full installation behavior remain available. Three mocked mode tests passed.

The H3 fix is included in the extension. The adjacent patches target separate local components and are not automatically installed by H3. Apply each patch from its corresponding project root after reviewing it and backing up the original. These patches have not been merged into the third-party upstream repositories.

Profiling identified redundant model network checks and callback stack inspection. Profiling timings include substantial instrumentation overhead and are not used as real startup speed measurements. H3 callback tests passed. This does not promise universal speed or quantization support.
