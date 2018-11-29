import numpy as np
from skimage.segmentation import slic
from skimage.transform import resize
import os
from data_generator import load_batch
import warnings

# To remove future warning from being printed out
warnings.simplefilter(action='ignore', category=FutureWarning)


def box(seg, i):
    xind = np.nonzero(seg.ravel('C') == i)
    [xmax, _] = np.unravel_index(np.max(xind), seg.shape, order = 'C')
    [xmin, _] = np.unravel_index(np.min(xind), seg.shape, order = 'C')
    yind = np.nonzero(seg.ravel('F') == i)
    [_, ymax] = np.unravel_index(np.max(yind), seg.shape, order = 'F')
    [_, ymin] = np.unravel_index(np.min(yind), seg.shape, order = 'F')
    return np.array([xmax, ymax, xmin, ymin])

def patch_cat(gt, SLIC, i, thres1, thres2):
    num = np.sum(gt[SLIC == i] > 125)
    denom = gt[SLIC == i].size
    size_true = np.sum(gt > 125)
    if float(num)/float(denom)>thres1:
        return 1
    else:
        if float(size_true) > 0 and float(num)/float(size_true) > thres2:
            return 1
        else: return 0

def xpatchify(img, SLIC, boxed, i):
    [inda, indb] = np.nonzero(SLIC!=i)
    imtemp = np.copy(img)
    imtemp[inda,indb,:] = 0
    x_temp = imtemp[int(boxed[2]):int(boxed[0]),
                 int(boxed[3]):int(boxed[1])]
    x_train = resize(x_temp, (80,80))
    return(x_train)

def get_labeled_patches(imgs, gts, n_segments = 100, thres1 = 0.2, thres2 = 0.2):
    """
    Get all the patches from the set of images.
    :param imgs: images
    :param gts: masks
    :param n_segments: max number of patches for image
    :param thres1:
    :param thres2:
    :return: patches: list of patches, size [len(img), n_patches_per_image, 80,80]
    :return: labels: list of labels per each patch, size [len(img), n_patches_per_image]
    """
    n = len(imgs)
    SLIC_list = np.asarray([slic(imgs[i, :], n_segments, compactness=20, sigma=10) for i in range(len(imgs))])

    # initialise boxes
    boxes = np.empty((n, 0)).tolist()
    # run box function to find all superpixel patches sizes
    for i in range(n):
        [boxes[i].append(box(SLIC_list[i, :], j)) for j in range(np.max(SLIC_list[i, :]))]

    patches = np.empty((n, 0)).tolist()
    # populating x_train
    for i in range(n):
        for j in range(np.max(SLIC_list[i, :])):
            patches[i].append(xpatchify(imgs[i, :], SLIC_list[i, :], boxes[i][j], j))

    #labels
    labels = np.empty((n, 0)).tolist()
    for j in range(n):
        [labels[j].append(patch_cat(gts[j, :], SLIC_list[j, :], i, thres1, thres2)) for i in range(np.max(SLIC_list[j, :]))]

    return patches, labels

if __name__ == "__main__":
    path = './data/'
    pimg = 'train_sample/'
    pgt = 'train_maps/'
    nfiles = len(os.listdir(path + pimg))
    imgs, gts = load_batch(path, pimg, pgt, nfiles, 2)
    get_labeled_patches(imgs, gts)