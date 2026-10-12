"""Isolated video seeking and preencoded-latent experiments."""
import copy
import inspect
import json
import logging
import os
import queue
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

import torch
import nodes
from comfy_api.latest import ComfyExtension, io


def vhs_module():
    return sys.modules[nodes.NODE_CLASS_MAPPINGS['VHS_LoadVideoTrim'].__module__]


def fast_generator():
    module = vhs_module()
    source = inspect.getsource(module.cv_frame_generator)
    marker = '    while video_cap.isOpened():'
    assert source.count(marker) == 1
    # Reproduce the original rate-conversion arithmetic before seeking.
    insert = '''    if skip_first_frames > 0:
        source_index = 0
        while total_frame_count < skip_first_frames:
            if time_offset < target_frame_time:
                source_index += 1
                time_offset += base_frame_time
            if time_offset < target_frame_time:
                continue
            time_offset -= target_frame_time
            total_frame_count += 1
        if not video_cap.set(cv2.CAP_PROP_POS_FRAMES, source_index):
            raise RuntimeError("Frame seeking is unsupported for this source")
        if not video_cap.grab():
            raise RuntimeError("Cannot seek to requested frame")
'''
    namespace = dict(module.__dict__)
    exec(compile(source.replace(marker, insert + marker), '<exact_seek>', 'exec'), namespace)
    exec(compile(inspect.getsource(module.resized_cv_frame_gen), '<seek_resize>', 'exec'), namespace)
    return namespace['resized_cv_frame_gen']


def selected_source_frames(fps, force_rate, frame_load_cap, skip_first_frames, select_every_nth):
    """Replay VHS cv_frame_generator's time_offset walk: yields the source index of every output frame."""
    base_frame_time = 1 / fps
    target_frame_time = base_frame_time if force_rate == 0 else 1 / force_rate
    source, time_offset, total, evaluated, added = 0, target_frame_time, 0, -1, 0
    while True:
        if time_offset < target_frame_time:
            source += 1
            time_offset += base_frame_time
        if time_offset < target_frame_time:
            continue
        time_offset -= target_frame_time
        total += 1
        if total <= skip_first_frames:
            continue
        evaluated += 1
        if evaluated % select_every_nth != 0:
            continue
        yield source
        added += 1
        if frame_load_cap > 0 and added >= frame_load_cap:
            return


def ffmpeg_direct_load(module, video, force_rate, frame_load_cap, skip_first_frames, select_every_nth,
                       custom_width=0, custom_height=0, format='None', **_):
    """Same frames, metadata and outputs as VHS load_video (no VAE / meta_batch / resize), but decoded by
    an ffmpeg pipe and converted in multithreaded batches straight into one preallocated tensor.
    VHS converts frame by frame and then copies everything again through np.fromiter, which dominates.
    Returns None when the request needs the original VHS path."""
    cv2, np = module.cv2, module.np
    fmt = module.get_format(format)
    cap = cv2.VideoCapture(video)
    if not cap.isOpened() or not cap.grab():
        raise ValueError(f"{video} could not be loaded with cv.")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    _, first = cap.retrieve()
    cap.release()
    height, width = first.shape[:2]
    del first
    if total_frames <= 0 or fps <= 0:
        return None
    if module.target_size(width, height, custom_width, custom_height, fmt.get('dim', (1,))[0]) != (width, height):
        return None  # resize requested: keep VHS's lanczos path
    target_frame_time = 1 / fps if force_rate == 0 else 1 / force_rate
    capacity = int(total_frames / fps * force_rate) if force_rate != 0 else total_frames
    capacity //= select_every_nth
    if frame_load_cap != 0:
        capacity = min(frame_load_cap, capacity)

    wanted = selected_source_frames(fps, force_rate, frame_load_cap, skip_first_frames, select_every_nth)
    target = next(wanted)
    frame_bytes = width * height * 3
    cmd = [module.ffmpeg_path or 'ffmpeg', '-v', 'error', '-threads', os.environ.get('OMP_NUM_THREADS', '0'),
           '-i', video, '-map', '0:v:0', '-vf', rf'select=gte(n\,{target})', '-fps_mode', 'passthrough',
           '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-']
    images = torch.empty((capacity + 2, height, width, 3), dtype=torch.float32)
    extra, batch = [], []
    count, position = 0, target
    pbar = module.ProgressBar(capacity)

    def flush():
        nonlocal count
        u8 = torch.from_numpy(np.frombuffer(b''.join(batch), np.uint8).reshape(len(batch), height, width, 3))
        room = max(0, min(len(batch), images.shape[0] - count))
        if room:
            images[count:count + room].copy_(u8[:room]).div_(255)  # same arithmetic as VHS
        if room < len(batch):
            extra.append(u8[room:].float().div_(255))
        count += len(batch)
        pbar.update_absolute(min(count, capacity), capacity)
        batch.clear()

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=frame_bytes * 4)
    frames = queue.Queue(maxsize=96)

    def reader():  # overlap pipe reads (ffmpeg decode) with the batch conversion below
        while True:
            buf = proc.stdout.read(frame_bytes)
            frames.put(buf if len(buf) == frame_bytes else None)
            if len(buf) < frame_bytes:
                return

    threading.Thread(target=reader, daemon=True).start()
    try:
        while target is not None:
            buf = frames.get()
            if buf is None:
                break
            if position == target:
                batch.append(buf)
                target = next(wanted, None)
                if len(batch) == 32:
                    flush()
            position += 1
        if batch:
            flush()
    finally:
        proc.kill()
        while True:  # unblock the reader if it is waiting on a full queue
            try:
                frames.get_nowait()
            except queue.Empty:
                break
        proc.wait()
    kept = images[:min(count, images.shape[0])]
    images = torch.cat([kept] + extra) if extra else kept
    if len(images) == 0:
        raise RuntimeError("No frames generated")
    if 'frames' in fmt and len(images) % fmt['frames'][0] != fmt['frames'][1]:
        if len(fmt['frames']) > 2 and fmt['frames'][2]:
            raise RuntimeError(f"The number of frames loaded {len(images)}, does not match the requirements "
                               "of the currently selected format.")
        div, mod = fmt['frames'][:2]
        images = images[:(len(images) - mod) // div * div + mod]
    start_time = skip_first_frames * target_frame_time
    target_frame_time *= select_every_nth
    audio = module.lazy_get_audio(video, start_time, frame_load_cap * target_frame_time)
    video_info = {
        "source_fps": fps,
        "source_frame_count": total_frames,
        "source_duration": total_frames / fps,
        "source_width": width,
        "source_height": height,
        "loaded_fps": 1 / target_frame_time,
        "loaded_frame_count": len(images),
        "loaded_duration": len(images) * target_frame_time,
        "loaded_width": width,
        "loaded_height": height,
    }
    return (images, len(images), audio, video_info)


def fast_load(kwargs, decoder='ffmpeg'):
    module = vhs_module()
    kwargs = dict(kwargs)
    path = module.folder_paths.get_annotated_filepath(module.strip_path(kwargs['video']))
    kwargs['video'] = path
    rate = kwargs.get('force_rate') or module._source_fps(path)
    start, duration = kwargs.pop('start_time', 0) or 0, kwargs.pop('duration', 0) or 0
    kwargs['skip_first_frames'] = int(round(start * rate)) if rate else 0
    kwargs['frame_load_cap'] = int(round(duration * rate)) if rate and duration > 0 else 0
    if (decoder == 'ffmpeg' and kwargs.get('vae') is None and kwargs.get('meta_batch') is None
            and shutil.which(module.ffmpeg_path or 'ffmpeg')):
        result = ffmpeg_direct_load(module, **kwargs)
        if result is not None:
            return result
    return module.load_video(generator=fast_generator(), **kwargs)


class FastTrim:
    @classmethod
    def INPUT_TYPES(cls):
        return nodes.NODE_CLASS_MAPPINGS['VHS_LoadVideoTrim'].INPUT_TYPES()
    RETURN_TYPES = ('IMAGE', 'INT', 'AUDIO', 'VHS_VIDEOINFO')
    RETURN_NAMES = ('IMAGE', 'frame_count', 'audio', 'video_info')
    FUNCTION = 'load_video'
    CATEGORY = 'Experiments'

    def load_video(self, **kwargs):
        return fast_load(kwargs)

    @classmethod
    def IS_CHANGED(cls, video, **kwargs):
        return nodes.NODE_CLASS_MAPPINGS['VHS_LoadVideoTrim'].IS_CHANGED(video, **kwargs)


class ReadBenchmark(FastTrim):
    OUTPUT_NODE = True
    RETURN_TYPES = ()
    RETURN_NAMES = ()

    def load_video(self, **kwargs):
        report = []
        for repeat in range(2):
            outputs, times = {}, {}
            for mode in (('original', 'seek') if repeat == 0 else ('seek', 'original')):
                start = time.perf_counter()
                outputs[mode] = (nodes.NODE_CLASS_MAPPINGS['VHS_LoadVideoTrim']().load_video(**kwargs)
                                 if mode == 'original' else fast_load(kwargs))  # 'seek' = ffmpeg decoder
                times[mode] = time.perf_counter() - start
            a, b = outputs['original'], outputs['seek']
            equal = a[0].shape == b[0].shape and all(torch.equal(x, y) for x, y in zip(a[0], b[0]))
            metadata_equal = a[1] == b[1] and a[3] == b[3]
            row = dict(repeat=repeat + 1, seconds=times, exact_equal=equal,
                       metadata_equal=metadata_equal, frames=a[1])
            report.append(row)
            logging.info('[READ_AB] %s', json.dumps(row))
            Path('/workspace/seg-benchmark/read-ab-results.json').write_text(json.dumps(report), encoding='utf-8')
            if not equal or not metadata_equal:
                raise RuntimeError('Read parity failed; experiment stopped')
            del a, b, outputs
        return ()


def dual_module():
    return sys.modules[nodes.NODE_CLASS_MAPPINGS['NKDAVLatentDualEncode'].__module__]


class CachedVideoEncode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='MiniMaxVideoEncodeCached', display_name='Video Encode (Auto / Single / Dual)',
                         inputs=[io.Image.Input('images'), io.Vae.Input('vae'),
                                 io.Combo.Input('gpu_mode', options=['auto', 'single', 'dual', 'quad', 'octa'], default='auto', optional=True)],
                         outputs=[io.Latent.Output()])

    @classmethod
    def execute(cls, images, vae, gpu_mode='auto'):
        logging.info('[ENCODE_CACHE] Computing video latent')
        if gpu_mode in ('quad', 'octa'):  # shared gpu_mode widget with the loader; encoding only ever uses 2 GPUs
            gpu_mode = 'dual'
        available = torch.cuda.device_count() >= 2 and str(vae.device) in ('cuda:0', 'cuda')
        if gpu_mode == 'dual' and not available:
            raise RuntimeError('Dual video encoding requires two GPUs with the VAE on GPU 0')
        if gpu_mode == 'single' or not available:
            logging.info('[ENCODE_CACHE] Single GPU encoding')
            return io.NodeOutput({'samples': vae.encode(images[:, :, :, :3])})
        return io.NodeOutput({'samples': dual_module().encode_dual(vae, images[:, :, :, :3])})


class PreencodedAV(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        schema = nodes.NODE_CLASS_MAPPINGS['NKDAVLatent'].define_schema()
        schema.node_id = 'NKDAVLatentPreencoded'
        schema.display_name = 'AV Latent From Cached Video'
        schema.inputs.append(io.Latent.Input('video_latent'))
        return schema

    @classmethod
    def execute(cls, video_latent, **kwargs):
        class Proxy:
            def encode(self, images):
                return video_latent['samples']
        kwargs['video_vae'] = Proxy()
        return nodes.NODE_CLASS_MAPPINGS['NKDAVLatent'].execute(**kwargs)


class CacheProbe(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='LatentCacheProbe', inputs=[io.Latent.Input('latent'), io.Int.Input('trial', default=0)],
                         outputs=[], is_output_node=True)

    @classmethod
    def execute(cls, latent, trial):
        x = latent['samples']
        logging.info('[ENCODE_CACHE] probe=%d shape=%s', trial, list(x.shape))
        return io.NodeOutput()


class Extension(ComfyExtension):
    async def get_node_list(self):
        return [CachedVideoEncode, PreencodedAV, CacheProbe]


def comfy_entrypoint():
    return Extension()
