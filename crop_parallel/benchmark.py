"""Compare crop implementations without saving or displaying media."""
import gc
import json
import logging
from pathlib import Path
import sys
import time

import torch
from comfy_api.latest import io
from . import nodes_subject_crop as parallel


class CropBenchmark(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        schema = parallel.MVEx_SubjectCropNode.define_schema()
        schema.node_id = 'CropParityBenchmark'
        schema.display_name = 'Crop Timing and Parity'
        schema.outputs = []
        schema.is_output_node = True
        schema.not_idempotent = True
        return schema

    @classmethod
    def execute(cls, **kwargs):
        original = next(m for m in list(sys.modules.values()) if
                        str(getattr(m, '__file__', '')).endswith('/MaskVidExperiments/nodes_subject_crop.py'))
        report = []
        # Alternate order to reduce warm-cache bias. Retain only one pair.
        for repeat, order in enumerate(('serial parallel', 'parallel serial', 'serial parallel')):
            outputs = {}
            times = {}
            for name in order.split():
                module = original if name == 'serial' else parallel
                parallel._perf_sync()
                start = time.perf_counter()
                outputs[name] = module.MVEx_SubjectCropNode.execute(**kwargs)
                parallel._perf_sync()
                times[name] = time.perf_counter() - start
            a, b = outputs['serial'], outputs['parallel']
            equal = all(torch.equal(a[k][i], b[k][i])
                        for k in (0, 1) for i in range(a[k].shape[0]))
            equal = equal and a[2] == b[2] and a[3] == b[3]
            row = dict(repeat=repeat + 1, seconds=times, exact_equal=equal,
                       image_shape=list(a[0].shape))
            report.append(row)
            logging.info('[CROP_AB] %s', json.dumps(row))
            del a, b, outputs
            gc.collect()
            if not equal:
                raise RuntimeError('Crop parity failed; experiment stopped')
        Path('/workspace/seg-benchmark/crop-ab-results.json').write_text(json.dumps(report), encoding='utf-8')
        return io.NodeOutput()
