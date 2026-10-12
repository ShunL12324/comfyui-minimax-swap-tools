"""MiniMax H3 swap helper nodes: parallel subject crop, cached video read/encode, dual-GPU VAE
encode, exclusion masks and sparse SAM3 temporal masks.

V1 and V3 nodes are registered together through NODE_CLASS_MAPPINGS (ComfyUI loads a pack via
either NODE_CLASS_MAPPINGS or comfy_entrypoint, never both). This also keeps V3 schemas lazy:
several of them copy NKDAVLatent's schema, which only exists once ComfyUI-NKD-Basic-Tools loads.
"""
from . import dual_vae_encode, io_cache, mask_exclusion, temporal_mask
from .crop_parallel.benchmark import CropBenchmark
from .crop_parallel.nodes_subject_crop import MVEx_SubjectCropNode

NODE_CLASS_MAPPINGS = {
    # V3 nodes, keyed by their schema node_id
    "MVEx_SubjectCropParallel": MVEx_SubjectCropNode,
    "CropParityBenchmark": CropBenchmark,
    "MiniMaxVideoEncodeCached": io_cache.CachedVideoEncode,
    "NKDAVLatentPreencoded": io_cache.PreencodedAV,
    "LatentCacheProbe": io_cache.CacheProbe,
    "NKDAVLatentDualEncode": dual_vae_encode.DualAVLatent,
    "DualVAEEncodeBenchmark": dual_vae_encode.EncodeBenchmark,
    # V1 nodes
    "VHS_LoadVideoTrimSeek": io_cache.FastTrim,
    "VideoReadParityBenchmark": io_cache.ReadBenchmark,
    **mask_exclusion.NODE_CLASS_MAPPINGS,
    **temporal_mask.NODE_CLASS_MAPPINGS,
}
NODE_DISPLAY_NAME_MAPPINGS = dict(temporal_mask.NODE_DISPLAY_NAME_MAPPINGS)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
