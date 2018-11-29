import os
import numpy as np
from matplotlib import image as mpimg

'''
GIVEN EXAMPLE

class DataGenerator:
    def __init__(self, config):
        self.config = config
        # load data here
        self.input = np.ones((500, 784))
        self.y = np.ones((500, 10))

    def next_batch(self, batch_size):
        idx = np.random.choice(500, batch_size)
        yield self.input[idx], self.y[idx]
'''

def load_image(infilename):
    """ Reads images """
    data = mpimg.imread(infilename)
    return data


def load_batch(path, pimg, pgt, nfiles, batch_size=1000):
    # sample randomly
    randomise = np.random.choice(nfiles, size=batch_size, replace=False)
    # generate file lists
    print('Reading file names ..')
    filelist = []
    filelist = [os.listdir(path + pimg)[i] for i in randomise]
    gtlist = ['gt_' + filelist[i] for i in range(len(filelist))]
    print('read')
    # initialise datasets
    imgs = []
    gts = []
    # read files
    print('Reading ', batch_size, ' files...')
    i = 0
    while i < batch_size:
        name = path + pimg + filelist[i]
        gtname = path + pgt + gtlist[i]
        if name.endswith(".jpg"):
            i += 1
            imgs.append(load_image(name))
            gts.append(load_image(gtname))

    imgs = np.asarray(imgs)
    gts = np.asarray(gts)
    print('Read ', i, ' files.')
    print('Check: img size', imgs.shape, '\tgt size', gts.shape)
    return imgs, gts


if __name__ == "__main__":
    path = './data/'
    pimg = 'train_sample/'
    pgt = 'train_maps/'
    nfiles = len(os.listdir(path + pimg))
    load_batch(path, pimg, pgt, nfiles,2)
