import os
import sys
import tensorflow as tf

# To import from sibling directory ../utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from models.mini_vgg_model import MiniVGG
from trainers.mini_vgg_trainer import BaselineTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.utils import get_args
from data_loader.mini_vgg_pretraining import DataGenerator


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

    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])
    # create tensorflow session
    sess = tf.Session()
    # create your data generator
    print("Generating data ...")
    data = DataGenerator(config)
    print("data generated")
    # create an instance of the model you want
    print("Creating model...")
    model = MiniVGG(config)
    print("done")
    # create tensorboard logger
    logger = Logger(sess, config)
    # create trainer and pass all the previous components to it
    print("Creating trainer")
    trainer = BaselineTrainer(sess, model, data, config, logger)
    print("done")
    #load model if exists
    model.load(sess)
    print("Model exists already, if you want to retrain it delete it first!")
    #here you train your model
    trainer.train()

    #TESTING
    #load model if exists
    model.load(sess)


if __name__ == '__main__':
    main()
