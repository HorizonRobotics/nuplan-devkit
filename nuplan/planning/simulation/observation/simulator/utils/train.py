import os
import numpy as np
import torch

def to_device(data, device='cuda:0'):
    if isinstance(data, torch.Tensor): 
        return data.to(device, non_blocking=True)
    elif isinstance(data, dict):
        for k in data.keys():
            data[k] = to_device(data[k], device)

        return data
    elif isinstance(data, (tuple, list)):
        for i in range(len(data)):
            data[i] = to_device(data[i], device)

        return data
    
    return data


def set_requires_grad(model, value):
    for p in model.parameters():
        p.requires_grad = value

def load_model_checkpoint(path, model):
    if os.path.exists(path):
        ckpt = torch.load(path, map_location='cpu')   
        model.load_state_dict(ckpt['state_dict'])
    return model

def resize(imgs, sz=256):
    return torch.nn.functional.interpolate(imgs, size=sz)


def to_numpy(t, flipy=False, uint8=True, i=0):
    out = t[:]
    if len(out.shape) == 4:
        out = out[i]
    out = out.detach().permute(1, 2, 0) # HWC
    out = out.flip([0]) if flipy else out
    out = out.detach().cpu().numpy()
    out = (out.clip(0, 1)*255).astype(np.uint8) if uint8 else out
    return out

def get_module(path):
    import pydoc

    m = pydoc.locate(path)
    assert m is not None, f'{path} not found'

    return m