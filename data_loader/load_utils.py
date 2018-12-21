import pickle as pkl
import os

"""
This module defines helper functions to load and save data.
"""

def save_obj(obj, name):
    """
    Shortcut function to save an object as pkl
    Args:
        obj: object to save
        name: filename of the object
    """
    with open(name + '.pkl', 'wb') as f:
        pkl.dump(obj, f)


def load_obj(name):
    """
    Shortcut function to load an object from pkl file
    Args:
        name: filename of the object
    Returns:
        obj: object to load
    """
    with open(name + '.pkl', 'rb') as f:
        return pkl.load(f)

def try_to_load_as_pickled_object(filepath):
    """
    This is a defensive way to write pickle.load, allowing for very large files on all platforms
    """
    max_bytes = 2**31 - 1
    input_size = os.path.getsize(filepath)
    bytes_in = bytearray(0)
    with open(filepath, 'rb') as f_in:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f_in.read(max_bytes)
    obj = pkl.loads(bytes_in)
    return obj
