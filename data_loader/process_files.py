import sys
import os.path

# To import from sibling directory ../utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import tensorlayer as tl
import numpy as np
from data_loader.load_utils import save_obj, load_obj


def process_all_files(list_indices, outputname1='patches.pkl', outputname2='labels_patches'):
    """This function is used to merge several data files from
    several teacher runs.

    Args:
        list_indices:it will merge all the patches_index and labels_patches_index for all index in the list
        outputname: filename where to save the merged data

    Example:
        list_indices = [0,1000,2000,3000, ...]')
    """
    patches = load_obj('data/patches_' + str(list_indices[0]) + '.pkl')
    labels = load_obj('data/labels_patches_' + str(list_indices[0]) + '.pkl')
    for d in list_indices:
        tmp1 = load_obj('data/patches_' + str(list_indices[0]))
        patches = np.append(patches, tmp1, axis=0)
        tmp2 = load_obj('data/labels_patches_' + str(list_indices[0]))
        labels = np.append(labels, tmp2, axis=0)
    save_obj(patches, 'data/patches')
    save_obj(labels, 'data/labels_patches')