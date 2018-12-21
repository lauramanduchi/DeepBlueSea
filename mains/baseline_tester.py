import tensorflow as tf
import os
import sys

# To import from sibling directory ../utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from data_loader.baseline_generator import DataGenerator
from models.baseline_model import BaselineModel
from trainers.baseline_trainer import BaselineTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.utils import get_args
from data_loader.preprocessing import *
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    # parse -c and put the wright config file ex. "-c configs/baseline.json"

    try:
        args = get_args()
        config = process_config(args.config)

    except:
        print("missing or invalid arguments")
        exit(0)

    # LOADING DATA AND PREPROCESSING
    path = './data/'
    pimg = 'test_sample/'
    nfiles = len(os.listdir(path + pimg))
    b_size = 10
    imgs = load_test_batch(path, pimg, nfiles, batch_size=b_size)
    list_patches, SLIC_list = get_patches(imgs)
    print("Got the patches.")

    # flatten the data
    patches_flat = []
    [patches_flat.append(patch) for patches_img in list_patches for patch in patches_img]
    patches_flat = np.array(patches_flat)


    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])
    # create tensorflow session
    sess = tf.Session()
    # create your data generator
    data = DataGenerator(config)
    # create an instance of the model you want
    model = BaselineModel(config)
    # create tensorboard logger
    logger = Logger(sess, config)
    # create trainer and pass all the previous components to it
    trainer = BaselineTrainer(sess, model, data, config, logger)
    # load model if exists
    model.load(sess)
    print("Model exists already, if you want to retrain it delete it first!")
    # here you train your model
    #trainer.train()

    # TESTING
    # load model if exists
    model.load(sess)

    feed_dict = {model.x: patches_flat, model.is_training: False}
    pred = sess.run([model.pred], feed_dict=feed_dict)
    labels_flat_copy = pred[0].copy()

    for index in range(b_size):
        slic_np = np.array(SLIC_list[index])
        values = slic_np.flatten()
        nl = len(set(values))
        label = labels_flat_copy[:nl]
        patches1 = []
        i = 0
        for l in label:
            if l == 1:
                patches1.append(i)
            i += 1
        labels_flat_copy = labels_flat_copy[nl:]

        mm = imgs[index].copy()
        for i in patches1:
            mm[slic_np == i] = 255
        fig, ax = plt.subplots(1, 2, figsize=(10, 10), sharex=True, sharey=True)
        ax[0].imshow(mark_boundaries(mm, SLIC_list[index]))
        ax[1].imshow(mark_boundaries(imgs[index], SLIC_list[index]))
        plt.show()


if __name__ == '__main__':
    main()