import sys
import os.path

# To import from sibling directory ../utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import tensorlayer as tl
import numpy as np
from data_loader.load_utils import save_obj, try_to_load_as_pickled_object
import argparse


def process_all_files(list_indices):
    """This function is used to merge several data files from
    several teacher runs.

    Args:
        list_indices:it will merge all the patches_index and labels_patches_index for all index in the list
        outputname: filename where to save the merged data

    Example:
        list_indices = [0,1000,2000,3000, ...]')
    """
    patches = try_to_load_as_pickled_object('data/patches_' + str(list_indices[0]) + '.pkl')
    labels = try_to_load_as_pickled_object('data/labels_patches_' + str(list_indices[0]) + '.pkl')
    for i in list_indices[1:]:
        tmp1 = try_to_load_as_pickled_object('data/patches_' + str(i) + '.pkl')
        patches = np.append(patches, tmp1, axis=0)
        tmp2 = try_to_load_as_pickled_object('data/labels_patches_' + str(i) + '.pkl')
        labels = np.append(labels, tmp2, axis=0)
        print("loaded data " + str(i))
    return patches, labels
    #save_obj(patches, 'data/patches')
    #save_obj(labels, 'data/labels_patches')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--list', nargs='+', help='<Required> Set flag', required=True)
    args = parser.parse_args()
    indices = args.list if args.list else [0]
    process_all_files(indices)