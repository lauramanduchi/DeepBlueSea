import tensorflow as tf
import os
import sys
import json
import numpy as np
# To import from sibling directory ../utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from data_loader.faster_rcnn_test_loader import DataGenerator
from models.faster_rcnn_noC_model import FasterRcnnModelNoC
from utils.config import process_config
from utils.utils import get_args
from testing.nms import *

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

    prediction_dict = {}

    if bool(config.debug):
        n = int(2/ config.batch_size)
    else:
        n = int(100/config.batch_size)

    for i in range(n):
        batch_x, batch_y_class, batch_y_reg, filenames = data.one_batch(n, i)
        filenames = list(filenames)
        feed_dict = {model.x: batch_x,
                     model.y_map: batch_y_class,
                     model.y_reg: batch_y_reg,
                     model.is_training: False}
        class_score, reg_score = sess.run([model.class_scores, model.reg_scores], feed_dict=feed_dict)

        for b in range(config.batch_size):
            class_score_b = class_score[b,:,:,:,:]
            reg_score_b = reg_score[b,:,:,:,:]
            print('Applying NMS to image {}'.format(filenames[b]))
            predictions = boxes_per_image(class_score_b, reg_score_b, n_proposals=2000, nms_thres=0.2, debug=0)
            prediction_dict[filenames[b]] = predictions

    dump_path = 'predictions/' + config.exp_name+'.json'
    with open(dump_path, 'w') as fp:
        json.dump(prediction_dict, fp)
        print('Wrote to ' + dump_path)




    
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
