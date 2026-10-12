# comfyui-minimax-swap-tools

Helper nodes used by the MiniMax H3 swap / reference workflows.

| Node | Purpose |
|---|---|
| `MVEx_SubjectCropParallel` | Subject-tracked crop (MaskVidExperiments planner) with a threaded CPU resize path |
| `VHS_LoadVideoTrimSeek` | `VHS_LoadVideoTrim` with frame seeking instead of decoding skipped frames |
| `MiniMaxVideoEncodeCached` | AV latent encode, auto / single / dual GPU |
| `NKDAVLatentPreencoded`, `NKDAVLatentDualEncode` | `NKDAVLatent` variants (pre-encoded / two-GPU temporal VAE encode) |
| `SAM3TemporalMask` | Sparse SAM3 tracking with optical-flow mask reconstruction |
| `SAM3OptionalExclusion`, `ExclusionMaskSwitch`, `ExclusionMaskStrength`, `CropTargetMegapixels` | Optional exclusion ("keep") region and mask helpers |
| `CropParityBenchmark`, `VideoReadParityBenchmark`, `DualVAEEncodeBenchmark`, `LatentCacheProbe` | A/B parity benchmarks |

## Requirements

Runtime dependencies on other packs (resolved when a node runs, not at import):

- [ComfyUI-VideoHelperSuite fork](https://github.com/ShunL12324/ComfyUI-VideoHelperSuite) (`VHS_LoadVideoTrim`)
- [ComfyUI-NKD-Basic-Tools](https://github.com/Nekodificador/ComfyUI-NKD-Basic-Tools) (`NKDAVLatent`)
- [MaskVidExperiments](https://github.com/drozbay/MaskVidExperiments) (`MVEx_SubjectUncrop` pairs with the crop)
- ComfyUI core with MiniMax H3 and SAM3 support

Python: `numpy`, `scipy`, `opencv-python` (with `DISOpticalFlow_create`).

## Install

```
cd ComfyUI/custom_nodes
git clone https://github.com/ShunL12324/comfyui-minimax-swap-tools
```
