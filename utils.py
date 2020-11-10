import numpy as np


def unwind_dict_values(element):
    if type(element) is not dict:
        return element.flatten()

    values = []
    for key, value in element.items():
        values.append(unwind_dict_values(value))

    return np.concatenate(values, axis=-1)


def unwind_space_shapes(space):
    if space.shape is not None:
        return [space.shape]

    shapes = []
    for _, space_ in space.spaces.items():
        shapes += unwind_space_shapes(space_)

    return shapes
