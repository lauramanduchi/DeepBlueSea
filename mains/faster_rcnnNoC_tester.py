import tensorflow as tf
import os
import sys
import numpy as np
# To import from sibling directory ../utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from data_loader.faster_rcnn_test_loader import DataGenerator
from models.faster_rcnn_noC_model import FasterRcnnModelNoC
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.utils import get_args


# Note: at the moment, this code will save A LOT of files to your computer
# modify as required

def main():
    args = get_args()
    config = process_config(args.config)
    print(args.config)

    print("we're in business")
    # create the experiments dirs
    #create_dirs([config.summary_dir, config.checkpoint_dir])
    # create tensorflow session
    sess = tf.Session()
    # create your data generator
    print('Starting Data Gen')
    data = DataGenerator(config)
    # create an instance of the model you want
    model = FasterRcnnModelNoC(config)
    # load model
    model.load(sess)

    n = int(100/ config.batch_size)
    for i in range(n):
        print("I get this far")
        batch_x, batch_y_class, batch_y_reg = data.one_batch(n, i)
        feed_dict = {model.x: batch_x,
                     model.y_map: batch_y_class,
                     model.y_reg: batch_y_reg,
                     model.is_training: False}
        class_score, reg_score = sess.run([model.class_scores, model.reg_scores], feed_dict=feed_dict)
        print("I get this far too")
        np.save(config.test_data_path + 'y_class' + str(i), batch_y_class)
        np.save(config.test_data_path + 'class_score' + str(i), class_score)
        np.save(config.test_data_path + 'reg_score' + str(i), reg_score)
    
    """
    batch_x, batch_y_class, batch_y_reg = data.one_batch()
    feed_dict = {model.x: batch_x,
                 model.y_map: batch_y_class,
                 model.y_reg: batch_y_reg,
                 model.is_training: False}
    class_score, reg_score = sess.run([model.class_scores, model.reg_scores], feed_dict=feed_dict)
    print("I get this far too")
    np.save(config.test_data_path + 'y_class' + str(i), batch_y_class)
    np.save(config.test_data_path + 'class_score' + str(i), class_score)
    np.save(config.test_data_path + 'reg_score' + str(i), reg_score)
    """
    
if __name__ == '__main__':
    main()
