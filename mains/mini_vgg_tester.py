import os
import sys

import tensorflow as tf

# To import from sibling directory ../utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from models.mini_vgg_model import MiniVGG
from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args
from data_loader.preprocessing import *


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
    path = 'data/'
    pimg = 'train_sample/'
    nfiles = len(os.listdir(path + pimg))
    b_size = 1
    imgs = load_test_batch(path, pimg, nfiles, batch_size=b_size)
    print("Got the images.")

    # flatten the data
    imgs_flat = []
    [imgs_flat.append(img) for img in imgs]
    imgs_flat = np.array(imgs_flat)

    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])
    # create tensorflow session
    sess = tf.Session()
    # create an instance of the model you want
    model = MiniVGG(config)

    # TESTING
    # load model if exists
    model.load(sess)

    feed_dict = {model.x: imgs_flat, model.is_training: False}
    feats = sess.run([model.layer_deconv1], feed_dict=feed_dict)
    print(feats.shape)


if __name__ == '__main__':
    main()
