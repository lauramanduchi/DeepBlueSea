import os
import sys

import tensorflow as tf

# To import from sibling directory ../utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from models.mini_vgg_model import MiniVGG
from utils.config import process_config
from utils.logger import Logger
from utils.utils import get_args


# NOTE: this file can
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
    """
    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])
    """
    # create tensorflow session
    sess = tf.Session()
    """
    # create your data generator
    data = DataGenerator(config)
    
    """

    # create an instance of the model you want
    model = MiniVGG(config)
    # create tensorboard logger
    logger = Logger(sess, config)
    """
    # create trainer and pass all the previous components to it
    trainer = BaselineTrainer(sess, model, data, config, logger)
    #load model if exists
    model.load(sess)
    print("Model exists already, if you want to retrain it delete it first!")
    #here you train your model
    trainer.train()

    #TESTING
    #load model if exists
    model.load(sess)
    """


if __name__ == '__main__':
    main()
