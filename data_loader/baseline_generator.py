import numpy as np
import os
import sys
# To import from sibling directory ../utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from data_loader.load_utils import load_obj
from data_loader.load_utils import try_to_load_as_pickled_object
from sklearn.model_selection import train_test_split
from data_loader.process_files import process_all_files

class DataGenerator:
    def __init__(self, config):
        self.config = config

        # load data here
        #input = try_to_load_as_pickled_object('./data/patches.pkl')
        #y = try_to_load_as_pickled_object('./data/labels_patches.pkl')
        print("\nloading the data")
        input, y = process_all_files([0,1000,2000,3000,4000,5000,6000,7000,8000,9000])
        print("\ndata loaded")

        self.input, self.input_dev, self.y, self.y_dev = train_test_split(input,
                                                                          y,
                                                                          test_size=self.config.val_split)

    def next_batch(self, batch_size):
        idx = np.random.choice(len(self.input), batch_size)
        yield self.input[idx], self.y[idx]

    def next_batch_dev(self, batch_size):
        idx = np.random.choice(len(self.input_dev), batch_size)
        yield self.input_dev[idx], self.y_dev[idx]
