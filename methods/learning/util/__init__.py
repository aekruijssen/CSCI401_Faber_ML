import numpy as np

def str2bool(v):
    return v.lower() in ('true', '1')


def str2intlist(value):
    if not value:
        return value
    else:
       return [int(num) for num in value.split(',')]


def str2list(value):
    if not value:
        return value
    else:
        return [num for num in value.split(',')]

def hex2rgb(h):
    return np.array([int(h.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)]) / 255