import pickle as pkl

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
        pkl.dump(obj, f, pkl.HIGHEST_PROTOCOL)


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