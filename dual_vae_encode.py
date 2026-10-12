"""Experimental two-device temporal VAE encoding with numerical verification."""
from concurrent.futures import ThreadPoolExecutor
import json
import logging
import math
from pathlib import Path
import time

import torch
import comfy.model_management
from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE
from comfy_api.latest import ComfyExtension, io

_replica = None
_replica_key = None


def sync():
    for index in range(2):
        torch.cuda.synchronize(index)


def replica_for(fsm, device):
    global _replica, _replica_key
    key = (id(fsm), str(device), next(fsm.parameters()).dtype)
    if key == _replica_key:
        return _replica
    started = time.perf_counter()
    # Recreate the stock architecture, preserving temporal and spatial rules.
    model = MiniMaxH3VideoVAE(clip_length=fsm.clip_length, token_drop=fsm.token_drop,
                            tile_size=fsm.tile_size, tile_overlap_min=fsm.tile_overlap_min,
                            tiling=fsm.tiling)
    model.to(device=device, dtype=key[2])
    model.load_state_dict(fsm.state_dict(), strict=True)
    # These normalization buffers are intentionally absent from state_dict.
    for name in ('pixel_mean', 'pixel_std'):
        setattr(model, name, getattr(fsm, name).to(device=device, dtype=key[2]).clone())
    model.eval()
    _replica, _replica_key = model, key
    sync()
    logging.info('[DUAL_ENCODE] replica setup %.3f s', time.perf_counter() - started)
    return model


def encode_dual(vae, images):
    if torch.cuda.device_count() < 2 or str(vae.device) not in ('cuda:0', 'cuda'):
        raise RuntimeError('This experiment requires the VAE on GPU 0 and a second GPU.')
    fsm = vae.first_stage_model
    if not isinstance(fsm, MiniMaxH3VideoVAE):
        raise TypeError('Only MiniMax H3 video VAE is supported.')
    original = fsm.encode_temporal

    def temporal(x, device):
        second = replica_for(fsm, torch.device('cuda:1'))
        count = math.ceil(x.shape[2] / fsm.clip_length)
        def run(rank, model, target):
            chunks = {}
            with torch.inference_mode(), torch.cuda.device(target):
                for index in range(rank, count, 2):
                    clip = x[:, :, index * fsm.clip_length:(index + 1) * fsm.clip_length].to(target)
                    if clip.shape[2] < fsm.clip_length:
                        padding = clip[:, :, -1:].repeat(1, 1, fsm.clip_length - clip.shape[2], 1, 1)
                        clip = torch.cat([clip, padding], dim=2)
                    chunks[index] = model._adaptive_encode(model._normalize_pixels(clip))
                torch.cuda.synchronize(target)
            return chunks
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(run, 1, second, torch.device('cuda:1'))
            chunks = run(0, fsm, device)
            other = pending.result()
        sync()
        compute = time.perf_counter() - start
        chunks.update({index: value.to(device) for index, value in other.items()})
        z = torch.cat([chunks[index] for index in range(count)], dim=2)
        if fsm.token_drop > 0:
            z = z[:, :, :-fsm.token_drop]
        sync()
        logging.info('[DUAL_ENCODE] chunks=%d compute=%.3f s transfer_assemble=%.3f s',
                     count, compute, time.perf_counter() - start - compute)
        return z

    fsm.encode_temporal = temporal
    try:
        return vae.encode(images)
    finally:
        fsm.encode_temporal = original


class VAEProxy:
    def __init__(self, vae):
        self.vae = vae

    def __getattr__(self, name):
        return getattr(self.vae, name)

    def encode(self, images):
        return encode_dual(self.vae, images)


class DualAVLatent(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        import nodes
        schema = nodes.NODE_CLASS_MAPPINGS['NKDAVLatent'].define_schema()
        schema.node_id = 'NKDAVLatentDualEncode'
        schema.display_name = 'AV Latent Dual Encode (Experimental)'
        return schema

    @classmethod
    def execute(cls, **kwargs):
        import nodes
        kwargs['video_vae'] = VAEProxy(kwargs['video_vae'])
        return nodes.NODE_CLASS_MAPPINGS['NKDAVLatent'].execute(**kwargs)


class EncodeBenchmark(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='DualVAEEncodeBenchmark', display_name='VAE Encode Parity',
                         inputs=[io.Image.Input('images'), io.Vae.Input('vae')],
                         outputs=[], is_output_node=True, not_idempotent=True)

    @classmethod
    def execute(cls, images, vae):
        report = []
        reference = None
        for label in ('single_cold', 'dual_cold', 'dual_warm', 'single_warm'):
            sync()
            for index in range(2):
                torch.cuda.reset_peak_memory_stats(index)
            start = time.perf_counter()
            out = encode_dual(vae, images) if label.startswith('dual') else vae.encode(images)
            sync()
            elapsed = time.perf_counter() - start
            if reference is None:
                reference = out.cpu()
            current = out.cpu()
            delta = (current.float() - reference.float()).abs()
            row = dict(mode=label, seconds=elapsed, shape=list(out.shape),
                       exact_equal=torch.equal(current, reference),
                       max_abs=delta.max().item(), mean_abs=delta.mean().item(),
                       peak_allocated_gib=[torch.cuda.max_memory_allocated(i) / 2**30 for i in range(2)],
                       peak_reserved_gib=[torch.cuda.max_memory_reserved(i) / 2**30 for i in range(2)])
            report.append(row)
            logging.info('[VAE_AB] %s', json.dumps(row))
            Path('/workspace/seg-benchmark/vae-ab-results.json').write_text(json.dumps(report), encoding='utf-8')
            del out, current, delta
        return io.NodeOutput()


class DualEncodeExtension(ComfyExtension):
    async def get_node_list(self):
        return [DualAVLatent, EncodeBenchmark]


def comfy_entrypoint():
    return DualEncodeExtension()
