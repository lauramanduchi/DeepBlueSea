import numpy as np
import os
import sys
# To import from sibling directory ../utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from data_loader.load_utils import load_obj
from sklearn.model_selection import train_test_split

class DataGenerator:
    def __init__(self, config):
        self.config = config

        # load data here
        try:
            input = load_obj('data/patches')
            y = load_obj('data/labels_patches')
        except:
            print("There is no data. Save it first (run data_loader/preprocessing.py)!! ")
            exit(0)

        self.input, self.input_dev, self.y, self.y_dev = train_test_split(input,
                                                                          y,
                                                                          test_size=self.config.val_split)

    def next_batch(self, batch_size):
        idx = np.random.choice(len(self.input), batch_size)
        yield self.input[idx], self.y[idx]

    def next_batch_dev(self, batch_size):
        idx = np.random.choice(len(self.input_dev), batch_size)
        yield self.input_dev[idx], self.y_dev[idx]
