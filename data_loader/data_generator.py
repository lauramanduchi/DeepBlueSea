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

path = '/Users/margheritarosnati/Documents/DS/2018-2/DL/DeepBlueSea/data/'
pimg = 'train_sample/'
pgt = 'train_maps/'
nfiles = len(os.listdir(path + pimg))


def load_image(infilename):
    """ Reads images """
    data = mpimg.imread(infilename)
    return data


def load_batch(path, pimg, pgt, nfiles, batch_size=1000):
    # sample randomly
    randomise = np.random.choice(nfiles, size=batch_size, replace=False)

    # generate file lists
    # code design choice: generating a priori filelist was quite slow, hence doing it here
    print('Reading file names ..')
    filelist = []
    for i in randomise:
        filelist = filelist + [os.listdir(path + pimg)[i]]
    gtlist = ['gt_' + filelist[i] for i in range(batch_size)]
    print('read')
    # initialise datasets
    imgs = []
    gts = []

    # read files
    print('Reading ', batch_size, ' files...')
    for i in range(batch_size):
        name = path + pimg + filelist[i]
        if name[-4:] == ".jpg":
            imgs.append(load_image(name))
        gtname = path + pgt + gtlist[i]
        if gtname[-4:] == ".jpg":
            gts.append(load_image(gtname))
    # print(len(imgs)) #debug
    imgs = np.asarray(imgs)
    gts = np.asarray(gts)
    print('read')
    print('Check: img size', imgs.shape, '\tgt size', gts.shape)
    return [imgs, gts]
    # TODO: return these for next step in pipeline


if __name__ == "__main__":
    load_batch(path, pimg, pgt, nfiles)
