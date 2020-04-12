import os
from glob import glob
import torch

def get_ckpt_path(base_dir, step):
    if step is None:
        return get_recent_ckpt_path(base_dir)
    ckpt_name = 'ckpt_{:08d}.pt'.format(step)
    files = glob(os.path.join(base_dir, "*.pt"))
    for f in files:
        if ckpt_name in f:
            return f, step
    raise Exception("Did not find {}.".format(ckpt_name))

def get_recent_ckpt_path(base_dir):
    files = glob(os.path.join(base_dir, "*.pt"))
    files.sort()
    if len(files) == 0:
        return None, None
    max_step = max([f.rsplit('_', 1)[-1].split('.')[0] for f in files])
    paths = [f for f in files if max_step in f]
    if len(paths) == 1:
        return paths[0], int(max_step)
    else:
        raise Exception("Multiple most recent ckpts {}".format(paths))

def to_tensor(x, device):
    return torch.tensor(x, dtype=torch.float32).to(device)
