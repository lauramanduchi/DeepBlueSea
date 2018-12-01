import numpy as np
import os
import sys
# To import from sibling directory ../utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from data_loader.load_utils import load_obj

class DataGenerator:
    def __init__(self, config):
        self.config = config
        a = load_obj('../data/patches')
        b = load_obj('../data/labels_patches')
        self.input = a
        self.y = b
        
        # load data here
        # Upload history of filters used from traffic_cat

        # try:
        #     self.input = load_obj('../data/patches')
        #     self.y = load_obj('../data/labels_patches')

        # Download history from traffic_cat

        # except:
        #     print("There is no data. Save it first (run data_loader/preprocessing.py)!! ")
        #     exit(0)

    def next_batch(self, batch_size):
        idx = np.random.choice(len(self.input), batch_size)
        yield self.input[idx], self.y[idx]
