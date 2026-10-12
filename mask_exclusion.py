"""Optional exclusion tracking and lazy mask selection."""
import logging

class ExclusionMaskSwitch:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"prompt": ("STRING", {"multiline": True}),
            "original": ("MASK", {"lazy": True}), "excluded": ("MASK", {"lazy": True})}}
    RETURN_TYPES = ("MASK",)
    FUNCTION = "select"
    CATEGORY = "mask/exclusion"

    def check_lazy_status(self, prompt, original=None, excluded=None):
        name = "excluded" if prompt.strip() else "original"
        value = excluded if prompt.strip() else original
        return [name] if value is None else []

    def select(self, prompt, original=None, excluded=None):
        return (excluded if prompt.strip() else original,)

class SAM3OptionalExclusion:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",), "model": ("MODEL",), "clip": ("CLIP",),
            "text": ("STRING", {"multiline": True, "default": ""}),
            "stride": ("INT", {"default": 1, "min": 1, "max": 16}),
            "detection_every_frames": ("INT", {"default": 6, "min": 1, "max": 120}),
            "flow_width": ("INT", {"default": 512, "min": 128, "max": 1024,"step":32}),
            "motion_compensation": ("BOOLEAN", {"default": True})}}
    RETURN_TYPES = ("MASK",)
    FUNCTION = "run"
    CATEGORY = "mask/exclusion"

    def run(self, images, model, clip, text="", stride=1, detection_every_frames=6,
            flow_width=512, motion_compensation=True):
        if not text.strip():
            import torch
            logging.info("Exclusion disabled: blank prompt; SAM tracking skipped")
            return (torch.zeros(images.shape[:3], dtype=torch.float32, device="cpu"),)
        import nodes
        condition = nodes.CLIPTextEncode().encode(clip, text)[0]
        return nodes.NODE_CLASS_MAPPINGS["SAM3TemporalMask"]().run(
            images, model, condition, stride, detection_every_frames, flow_width, motion_compensation)

NODE_CLASS_MAPPINGS = {"ExclusionMaskSwitch": ExclusionMaskSwitch,
                       "SAM3OptionalExclusion": SAM3OptionalExclusion}


class CropTargetMegapixels:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"megapixels": ("FLOAT", {"default": 0.3, "min": 0.01, "max": 16.0, "step": 0.01})}}
    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "convert"
    CATEGORY = "mask/exclusion"

    def convert(self, megapixels):
        if megapixels <= 0:
            raise ValueError("Head generation resolution must be positive")
        return (-float(megapixels),)

NODE_CLASS_MAPPINGS["CropTargetMegapixels"] = CropTargetMegapixels


class ExclusionMaskStrength:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"mask": ("MASK",), "strength": ("FLOAT", {"default": 0.85,"min":0.,"max":1.,"step":0.05})}}
    RETURN_TYPES = ("MASK",)
    FUNCTION = "scale"
    CATEGORY = "mask/exclusion"

    def scale(self, mask, strength):
        return (mask * strength,)

NODE_CLASS_MAPPINGS["ExclusionMaskStrength"] = ExclusionMaskStrength
