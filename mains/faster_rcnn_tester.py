import tensorflow as tf
import os
import sys

# To import from sibling directory ../utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from data_loader.generator import DataGenerator
from models.faster_rcnn_model import FasterRcnnModel
from trainers.faster_rcnn_trainer import FasterRcnnTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.utils import get_args


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    # parse -c and put the wright config file ex. "-c configs/baseline.json"

    config = process_config("../configs/faster_rcnn.json")

    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])
    # create tensorflow session
    sess = tf.Session()
    # create your data generator
    data = DataGenerator(config)
    # create an instance of the model you want
    model = FasterRcnnModel(config)
    # create tensorboard logger
    logger = Logger(sess, config)
    # create trainer and pass all the previous components to it
    trainer = FasterRcnnTrainer(sess, model, data, config, logger)
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
