import numpy as np

class DataGenerator:
    def __init__(self, config):
        self.config = config
        
        # load data here
        # Upload history of filters used from traffic_cat
        try:
            self.input = load_obj('../data/patches')
            self.y = load_obj('../data/labels_patches')
        # Download history from traffic_cat
        except:
            print("There is no data. Save it first (run data_loader/preprocessing.py)!! ")

    def next_batch(self, batch_size):
        idx = np.random.choice(500, batch_size)
        yield self.input[idx], self.y[idx]
