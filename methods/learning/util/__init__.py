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


class AttrDict(dict):
    __setattr__ = dict.__setitem__

    def __getattr__(self, attr):
        # Take care that getattr() raises AttributeError, not KeyError.
        # Required e.g. for hasattr(), deepcopy and OrderedDict.
        try:
            return self.__getitem__(attr)
        except KeyError:
            raise AttributeError("Attribute %r not found" % attr)

    def __getstate__(self):
        return self

    def __setstate__(self, d):
        self = d
