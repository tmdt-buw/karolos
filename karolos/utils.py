import os
import random

import numpy as np
import torch


def unwind_dict_values(element):
    if type(element) is not dict:
        return element.flatten()

    values = []
    for key, value in element.items():
        values.append(unwind_dict_values(value))

    if len(values):
        return np.concatenate(values, axis=-1)
    else:
        return np.array([])


def unwind_space_shapes(space):
    if space.shape is not None:
        return [space.shape]

    shapes = []
    for _, space_ in space.spaces.items():
        shapes += unwind_space_shapes(space_)

    return shapes


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
