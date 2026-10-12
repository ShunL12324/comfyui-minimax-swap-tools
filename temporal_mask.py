"""Sparse SAM tracking with optical-flow mask reconstruction. Preserves every input frame in the output."""
import logging
import math
import time
import threading
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from comfy_extras.nodes_sam3 import SAM3_VideoTrack
from comfy.ldm.sam3.tracker import unpack_masks
import comfy.model_management

_FLOW_THREADS_LOCK=threading.RLock()


def sample_indices(count, stride):
    if count < 1 or stride < 1:
        raise ValueError('Frame count and stride must be positive')
    indices = list(range(0, count, stride))
    if indices[-1] != count-1:
        indices.append(count-1)
    return indices


def reconstruct_masks(images, masks, indices, use_flow=True):
    """Interpolate sparse masks, optionally compensating for image motion."""
    count, height, width, _ = images.shape
    mh, mw = masks.shape[-2:]
    gray = []
    device = comfy.model_management.get_torch_device()
    if use_flow:
        for frame in images:
            small_rgb=cv2.resize(frame[...,:3].detach().cpu().numpy(),(mw,mh),interpolation=cv2.INTER_LINEAR)
            rgb=(small_rgb.clip(0,1)*255).astype(np.uint8)
            gray.append(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY))
    yy, xx = np.mgrid[:mh,:mw].astype(np.float32)
    flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST) if use_flow else None
    small_outputs=[]
    left = 0
    for frame_idx in range(count):
        comfy.model_management.throw_exception_if_processing_interrupted()
        while left+1<len(indices) and indices[left+1]<=frame_idx:
            left+=1
        right=min(left+1,len(indices)-1)
        if frame_idx==indices[left] or left==right:
            mask=masks[left]
        else:
            alpha=(frame_idx-indices[left])/(indices[right]-indices[left])
            l,r=masks[left],masks[right]
            if use_flow:
                fl=flow.calc(gray[frame_idx],gray[indices[left]],None)
                fr=flow.calc(gray[frame_idx],gray[indices[right]],None)
                l=cv2.remap(l,xx+fl[...,0],yy+fl[...,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
                r=cv2.remap(r,xx+fr[...,0],yy+fr[...,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
            mask=l*(1-alpha)+r*alpha
        small_outputs.append(mask)
    output=torch.empty((count,height,width),dtype=torch.float32,device='cpu')
    for start in range(0,count,16):
        batch=torch.from_numpy(np.stack(small_outputs[start:start+16])).to(device)[:,None]
        resized=F.interpolate(batch,size=(height,width),mode='bilinear',align_corners=False)[:,0]
        output[start:start+len(batch)].copy_(resized.cpu())
        del batch,resized
    return output


class SAM3TemporalMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'images':('IMAGE',),'model':('MODEL',),'conditioning':('CONDITIONING',),
                'stride':('INT',{'default':4,'min':1,'max':16}),
                'detection_every_frames':('INT',{'default':6,'min':1,'max':120}),
                'flow_width':('INT',{'default':256,'min':128,'max':1024,'step':32}),
                'motion_compensation':('BOOLEAN',{'default':True})}}
    RETURN_TYPES=('MASK',)
    FUNCTION='run'
    CATEGORY='image/segmentation'

    def run(self,images,model,conditioning,stride=4,detection_every_frames=6,
            flow_width=256,motion_compensation=True):
        started_at=time.perf_counter()
        indices=sample_indices(len(images),stride)
        selected=images[indices]
        selected_at=time.perf_counter()
        data=SAM3_VideoTrack.execute(selected,model,conditioning=conditioning,detection_threshold=0.5,
               max_objects=0,detect_interval=max(1,math.ceil(detection_every_frames/stride)))[0]
        tracked_at=time.perf_counter()
        packed=data['packed_masks']
        mh=max(16,round(flow_width*images.shape[1]/images.shape[2]))
        if packed is None or packed.shape[1] == 0:
            masks=np.zeros((len(indices),mh,flow_width),np.float32)
        else:
            chunks=[]
            for start in range(0,len(packed),8):
                union=packed[start:start+8,0].clone()
                for j in range(1,packed.shape[1]):union.bitwise_or_(packed[start:start+8,j])
                expanded=unpack_masks(union).float()[:,None]
                chunks.append(F.interpolate(expanded,size=(mh,flow_width),mode='bilinear',align_corners=False)[:,0].cpu().numpy())
            masks=np.concatenate(chunks)
        logging.info('Temporal mask: %s input frames, %s inferred frames, stride %s',len(images),len(indices),stride)
        unpacked_at=time.perf_counter()
        # Large host CPU counts can make small optical-flow jobs slower.
        # Restore the process setting after this bounded operation.
        with _FLOW_THREADS_LOCK:
            previous_threads=cv2.getNumThreads()
            try:
                cv2.setNumThreads(min(previous_threads,4))
                output=reconstruct_masks(images,masks,indices,motion_compensation)
            finally:
                cv2.setNumThreads(previous_threads)
        self.last_profile={'select':selected_at-started_at,'track':tracked_at-selected_at,
                           'unpack':unpacked_at-tracked_at,'reconstruct':time.perf_counter()-unpacked_at}
        return (output,)


NODE_CLASS_MAPPINGS={'SAM3TemporalMask':SAM3TemporalMask}
NODE_DISPLAY_NAME_MAPPINGS={'SAM3TemporalMask':'SAM3 Temporal Mask'}
